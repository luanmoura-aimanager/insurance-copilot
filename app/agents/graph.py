from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from app.llm import get_client
# Import direto das funções core do MCP server (mesmo processo, sem protocolo MCP).
# run_query já carrega o guard SELECT-only + LIMIT e conecta pela role read-only.
from mcp_servers.postgres_mcp_server import get_schema, run_query

MAX_ITERATIONS = 5
SUPERVISOR_MODEL = "claude-haiku-4-5"  # routing é tarefa leve: modelo barato basta
SQL_MODEL = "claude-haiku-4-5"          # SQL simples: Haiku dá conta

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


# --- 3. Supervisor node: decide de verdade, via LLM com structured output ---
def supervisor(state: State) -> dict:
    i = state["iterations"] + 1

    client = get_client()
    resp = client.messages.create(
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
    tool_use = next(b for b in resp.content if b.type == "tool_use")
    decision = SupervisorDecision.model_validate(tool_use.input)

    print(f"[supervisor] iteration {i} -> decided: {decision.next} ({decision.reasoning})")
    return {
        "iterations": i,
        "next": decision.next,
        "messages": [AIMessage(content=decision.reasoning, name="supervisor")],
    }


# --- 4. SQL worker (single-pass): pergunta -> SQL -> run_query. Sem loop ReAct ainda. ---
def sql_worker(state: State) -> dict:
    # 1. Pega a pergunta do usuário (a última HumanMessage do histórico).
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    # 2. Schema numa chamada só — dá os nomes de tabela/coluna pro modelo.
    schema = get_schema()

    # 3. LLM gera UMA query, via structured output (devolve {"sql": "..."}).
    client = get_client()
    resp = client.messages.create(
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
    tool_use = next(b for b in resp.content if b.type == "tool_use")
    sql = SqlQuery.model_validate(tool_use.input).sql

    # 4. Executa pelo guard + role read-only. run_query nunca levanta: erros voltam como
    #    texto, então o supervisor os enxerga em vez do grafo estourar.
    rows = run_query(sql)

    print(f"[sql_worker] SQL: {sql}")
    return {"messages": [AIMessage(content=f"SQL: {sql}\nResult: {rows}", name="sql_worker")]}


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

builder.set_entry_point("supervisor")
builder.add_conditional_edges("supervisor", route, {
    "sql_worker": "sql_worker",
    END: END,
})
builder.add_edge("sql_worker", "supervisor")  # worker returns to the supervisor

graph = builder.compile()


# --- 7. Run (só quando executado direto; importar o módulo não roda o grafo) ---
if __name__ == "__main__":
    final = graph.invoke({
        "iterations": 0,
        "next": "",
        "messages": [HumanMessage(content="How many perils are there?")],
    })
    print("FINAL STATE:", final)
