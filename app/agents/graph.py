import asyncio
import logging
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from app.cost import record_call_cost
from app.llm import get_async_client
# Import direto das funções core do MCP server (mesmo processo, sem protocolo MCP).
# run_query já carrega o guard SELECT-only + LIMIT e conecta pela role read-only.
from mcp_servers.postgres_mcp_server import get_schema, run_query

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5
SUPERVISOR_MODEL = "claude-haiku-4-5"  # routing é tarefa leve: modelo barato basta
SQL_MODEL = "claude-haiku-4-5"          # SQL simples: Haiku dá conta
SYNTHESIZER_MODEL = "claude-haiku-4-5"  # transformar linha de resultado em frase: idem

# Frase de quando nenhum worker rodou (pergunta fora de escopo, ou o circuit breaker
# cortou antes de qualquer resultado). Mora aqui, e não na API, porque quem escreve a
# resposta final agora é o synthesizer — o /ask só lê a última mensagem.
NO_ANSWER = "Não consegui responder essa pergunta com os dados disponíveis."

# Structured output canônico (mesmo padrão da extração de seguros): expõe UM tool
# cujo input_schema é o JSON Schema do Pydantic e força tool_choice pra ele. O modelo
# é OBRIGADO a devolver argumentos que batem com o schema.
_DECISION_TOOL = "route_decision"
_SQL_TOOL = "emit_sql"

SUPERVISOR_SYSTEM = (
    "Você é o supervisor de um grafo de agentes que responde perguntas sobre seguros "
    "residenciais. Sua função é rotear: olhe o histórico da conversa e decida o próximo "
    "passo.\n\n"
    "Workers disponíveis:\n"
    "  - sql_worker: responde perguntas sobre o banco de dados de seguros via SQL "
    "(tabelas de apólices, coberturas, perigos, exclusões).\n"
    "  - END: encerre quando a pergunta já estiver respondida pelo resultado de um "
    "worker. Se a última mensagem já traz o dado que responde a pergunta, escolha END.\n\n"
    "Responda SEMPRE chamando a tool route_decision."
)

SQL_SYSTEM = (
    "Você traduz uma pergunta em UMA query SQL (PostgreSQL) sobre o schema fornecido. "
    "Gere apenas UM SELECT que responda a pergunta — sem comentários, sem cercas de "
    "markdown, sem ponto e vírgula. Use exatamente os nomes de tabela/coluna do schema. "
    "Responda SEMPRE chamando a tool emit_sql."
)

SYNTHESIZER_SYSTEM = (
    "Você recebe a pergunta de um usuário e o resultado cru de uma query SQL. Devolva "
    "UMA frase em pt-BR que responda a pergunta usando esse resultado. Não invente "
    "nenhum dado além do que está no resultado — se ele não responder a pergunta, diga "
    "isso em uma frase. Responda só com a frase, sem preâmbulo e sem repetir o SQL."
)


# --- 1. State: the record that travels through the graph ---
class State(TypedDict):
    iterations: int
    next: str
    messages: Annotated[list, add_messages]


# --- 2. Supervisor decision: `next` é ENUM = o cinto de segurança ---
class SupervisorDecision(BaseModel):
    next: Literal["sql_worker", "END"]  # enum = the belt: no invalid worker can be returned
    reasoning: str                       # one line of why, for the message history


class SqlQuery(BaseModel):
    sql: str  # structured output = só o SQL, sem cercas de markdown pra limpar


# --- 2.5 Custo por chamada: BEST EFFORT, nunca derruba a resposta ---
async def _record_cost(agent_name: str, resp) -> None:
    """Grava 1 linha de cost_event pra chamada que acabou de voltar.

    Duas decisões:

    - **`resp.model`, não o alias que mandamos.** Enviamos "claude-haiku-4-5"; a API
      cobra e devolve o id versionado ("claude-haiku-4-5-20251001"), que é a chave do
      pricing.json. Gravar o alias faria o custo cair no fail-loud do cost_usd — ou,
      pior, num preço de outra versão do modelo.
    - **Best effort.** Quando chegamos aqui a chamada já foi paga e a resposta já está
      na mão. Falha ao gravar custo (banco fora, modelo sem preço) vira log e segue:
      perder a observabilidade de uma chamada é MUITO mais barato do que devolver 500
      num /ask que já custou dinheiro e já tem resposta.

    É chamado ANTES do parse do tool_use, e não depois: um payload malformado também
    foi cobrado, e o custo dele não pode sumir junto com a exceção do parse.
    """
    try:
        await record_call_cost(
            agent_name=agent_name,
            model=resp.model,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
    except Exception as exc:  # noqa: BLE001 — best effort é o ponto: nada aqui pode subir
        logger.warning("falha ao gravar custo de %s: %s", agent_name, exc)


# --- 3. Supervisor node: decide de verdade, via LLM com structured output ---
async def supervisor(state: State) -> dict:
    i = state["iterations"] + 1

    client = get_async_client()
    resp = await client.messages.create(
        model=SUPERVISOR_MODEL,
        max_tokens=512,
        system=SUPERVISOR_SYSTEM,
        tools=[
            {
                "name": _DECISION_TOOL,
                "description": "Registra a decisão de roteamento do supervisor.",
                "input_schema": SupervisorDecision.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": _DECISION_TOOL},
        messages=[
            {"role": "user" if m.type == "human" else "assistant", "content": m.content}
            for m in state["messages"]
        ],
    )
    await _record_cost("supervisor", resp)

    tool_use = next(b for b in resp.content if b.type == "tool_use")
    decision = SupervisorDecision.model_validate(tool_use.input)

    print(f"[supervisor] iteration {i} -> decided: {decision.next} ({decision.reasoning})")
    return {
        "iterations": i,
        "next": decision.next,
        "messages": [AIMessage(content=decision.reasoning, name="supervisor")],
    }


# --- 4. SQL worker (single-pass): pergunta -> SQL -> run_query. Sem loop ReAct ainda. ---
async def sql_worker(state: State) -> dict:
    # 1. Pega a pergunta do usuário (a última HumanMessage do histórico).
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    # 2. Schema numa chamada só — dá os nomes de tabela/coluna pro modelo. Se a conexão
    #    falhar aqui, curto-circuita: sem schema não dá pra gerar SQL, então volta o erro
    #    como mensagem (o supervisor enxerga e encerra) em vez de estourar o grafo.
    #    get_schema é psycopg3 SÍNCRONO: roda numa thread pra não travar o event loop.
    try:
        schema = await asyncio.to_thread(get_schema)
    except Exception as exc:  # noqa: BLE001 — qualquer falha de DB vira mensagem, não crash
        return {"messages": [AIMessage(content=f"SQL error (schema): {exc}", name="sql_worker")]}

    # 3. LLM gera UMA query, via structured output (devolve {"sql": "..."}).
    client = get_async_client()
    resp = await client.messages.create(
        model=SQL_MODEL,
        max_tokens=1024,
        system=SQL_SYSTEM,
        tools=[
            {
                "name": _SQL_TOOL,
                "description": "Emite a query SQL que responde a pergunta.",
                "input_schema": SqlQuery.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": _SQL_TOOL},
        messages=[
            {
                "role": "user",
                "content": (
                    f"Schema:\n{schema}\n"
                    f"Pergunta: {question}\n\n"
                    "Gere o SELECT que a responde."
                ),
            }
        ],
    )
    await _record_cost("sql_worker", resp)

    tool_use = next(b for b in resp.content if b.type == "tool_use")
    sql = SqlQuery.model_validate(tool_use.input).sql

    # 4. Executa pelo guard + role read-only. run_query nunca levanta: erros voltam como
    #    texto, então o supervisor os enxerga em vez do grafo estourar. Síncrono também
    #    (mesma conexão psycopg3), então vai pra thread igual ao get_schema.
    rows = await asyncio.to_thread(run_query, sql)

    print(f"[sql_worker] SQL: {sql}")
    return {"messages": [AIMessage(content=f"SQL: {sql}\nResult: {rows}", name="sql_worker")]}


# --- 4.5 Synthesizer: último nó SEMPRE. Vira o resultado cru em frase. ---
async def synthesizer(state: State) -> dict:
    """Escreve a resposta final em linguagem natural.

    Roda em todo caminho que termina o grafo, inclusive quando nenhum worker rodou —
    por isso é ele, e não o supervisor, quem produz a última mensagem. A API só precisa
    ler `messages[-1]`.

    Sem chamada de LLM quando não há resultado de worker: não há o que sintetizar, e
    pagar uma chamada só pra escrever "não sei" seria queimar dinheiro à toa.
    """
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )
    resultado = next(
        (
            m.content
            for m in reversed(state["messages"])
            if isinstance(m, AIMessage) and m.name == "sql_worker"
        ),
        None,
    )

    if resultado is None:
        print("[synthesizer] sem resultado de worker -> fallback estático")
        return {"messages": [AIMessage(content=NO_ANSWER, name="final")]}

    # Texto livre, não structured output: a saída é UMA frase em prosa, e forçar uma
    # tool aqui só embrulharia uma string em JSON sem ganhar nada.
    frase = None
    try:
        client = get_async_client()
        resp = await client.messages.create(
            model=SYNTHESIZER_MODEL,
            max_tokens=512,
            system=SYNTHESIZER_SYSTEM,
            messages=[
                {
                    "role": "user",
                    "content": f"Pergunta: {question}\n\nResultado da query:\n{resultado}",
                }
            ],
        )
        await _record_cost("synthesizer", resp)
        frase = "".join(b.text for b in resp.content if b.type == "text").strip()
    except Exception as exc:  # noqa: BLE001 — o dado já está na mão; não morrer no último nó
        logger.warning("synthesizer falhou (%s); caindo no texto cru do worker", exc)

    # `or resultado` cobre os dois modos de falha (exceção e resposta vazia): o texto
    # cru é feio, mas é uma resposta REAL e já paga — melhor que estourar o grafo.
    print("[synthesizer] frase final gerada")
    return {"messages": [AIMessage(content=frase or resultado, name="final")]}


# --- 5. Conditional edge: routes by reading State (with fail-closed fallback) ---
def route(state: State) -> str:
    if state["iterations"] >= MAX_ITERATIONS:  # mechanical guard: does not ask the LLM
        print("[route] circuit breaker -> END")
        return END
    nxt = state["next"]
    if nxt == "END":  # decisão legítima do supervisor de encerrar (o enum devolve a string "END")
        return END
    if nxt != "sql_worker":  # suspenders: enum should prevent this, but if it slips → END
        print(f"[route] invalid next '{nxt}' -> END (fail closed)")
        return END
    return nxt


# --- 6. Build the graph ---
builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("sql_worker", sql_worker)
builder.add_node("synthesizer", synthesizer)

builder.set_entry_point("supervisor")
# Onde antes o route() ia direto pro END, agora passa pelo synthesizer — o mapa traduz
# o valor devolvido pelo route (que continua sendo END) no nó de saída. É por isso que
# route() não precisou mudar: a decisão dele é a mesma, só o destino é outro.
builder.add_conditional_edges("supervisor", route, {
    "sql_worker": "sql_worker",
    END: "synthesizer",
})
builder.add_edge("sql_worker", "supervisor")   # worker returns to the supervisor
builder.add_edge("synthesizer", END)           # synthesizer é sempre o último nó

graph = builder.compile()


# --- 7. Run (só quando executado direto; importar o módulo não roda o grafo) ---
if __name__ == "__main__":
    final = asyncio.run(graph.ainvoke({
        "iterations": 0,
        "next": "",
        "messages": [HumanMessage(content="How many perils are there?")],
    }))
    print("FINAL STATE:", final)
