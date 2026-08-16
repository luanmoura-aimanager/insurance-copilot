import asyncio
import logging
from typing import TypedDict, Annotated, Literal
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from pydantic import BaseModel
from sqlalchemy import func, select

from app.agents.context import get_client_name, get_request_id
from app.cost import record_call_cost
from app.llm import get_async_client
from app.models import PolicyDocument
from app.rag.search import (
    MAX_DISTANCE_PADRAO,
    contar_documentos_pesquisaveis,
    search_clauses,
)
# Import direto das funções core do MCP server (mesmo processo, sem protocolo MCP).
# run_query já carrega o guard SELECT-only + LIMIT e conecta pela role read-only.
from mcp_servers.postgres_mcp_server import ERRO_PREFIXO, get_schema, run_query

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 5
SUPERVISOR_MODEL = "claude-haiku-4-5"  # routing é tarefa leve: modelo barato basta
SQL_MODEL = "claude-haiku-4-5"          # SQL simples: Haiku dá conta
SYNTHESIZER_MODEL = "claude-haiku-4-5"  # transformar linha de resultado em frase: idem

# Quantas cláusulas o rag_worker recupera por pergunta. 5 é o mesmo default de
# `search_clauses`: o suficiente pra cobrir seguradoras diferentes dizendo a mesma coisa,
# e pouco o bastante pra caber no contexto do synthesizer sem virar despejo de texto.
RAG_K = 5

# --- As QUATRO frases finais possíveis, e elas NÃO são intercambiáveis ---
#
# Cada uma responde a uma pergunta diferente do usuário sobre o que aconteceu, e trocá-las
# entre si é mentir de um jeito plausível:
#
#   NO_ANSWER      "eu deveria conseguir, mas falhei"  — nenhum worker chegou a produzir
#                  resultado, e nada estourou. Com o grafo single-hop isso só acontece se
#                  o supervisor devolver `END` ou um valor inválido, ou se o circuit
#                  breaker cortar (hoje inalcançável). É um problema NOSSO de ROTEAMENTO,
#                  e dá pra tentar de novo.
#   NADA_RELEVANTE "eu procurei e o corpus não tem"    — a busca rodou e todos os vizinhos
#                  ficaram acima do limiar de relevância. É um fato sobre o CORPUS.
#   FORA_DE_ESCOPO "eu não trato desse assunto"        — o supervisor classificou a
#                  pergunta como de outro domínio, antes de gastar qualquer busca. É um
#                  fato sobre o SISTEMA, e reformular a pergunta não muda nada.
#   FALHA_INTERNA  "algo quebrou do nosso lado"        — um worker levantou exceção (banco
#                  fora do ar, API de embedding recusando, payload malformado). É um fato
#                  sobre a INFRAESTRUTURA: nada a ver com a pergunta, e tentar de novo
#                  mais tarde é a única ação útil do lado do usuário. A diferença prática
#                  com NO_ANSWER é o que o operador tem pra investigar — a FALHA_INTERNA
#                  sempre tem um traceback correspondente no log, o NO_ANSWER não.
#
# Só UMA das quatro declara a base (o rodapé da seção 2.7): a NADA_RELEVANTE, porque ela é
# a única que afirma algo sobre o CORPUS — e sem dizer em quantas apólices procurou, ela lê
# como "isso não existe" em vez de "não achei nas que eu tenho". Nas outras três nada
# chegou a ser consultado, e declarar base ali afirmaria uma verificação que não houve.
#
# As quatro moram aqui, e não na API, porque quem escreve a resposta final é o synthesizer
# — o /ask só lê a última mensagem.
NO_ANSWER = "Não consegui responder essa pergunta com os dados disponíveis."
NADA_RELEVANTE = (
    "Não encontrei nenhuma cláusula relevante sobre isso nas condições gerais indexadas."
)
FORA_DE_ESCOPO = (
    "Só respondo perguntas sobre as condições gerais de seguro residencial — coberturas, "
    "perigos cobertos, exclusões e franquias."
)
FALHA_INTERNA = (
    "Tive uma falha interna ao consultar as condições gerais. Tente de novo em alguns "
    "instantes."
)

# O nome da mensagem que MARCA uma falha de worker, e é a marca ser o `name` que importa:
# a alternativa era um prefixo mágico no conteúdo (`"SQL error: ..."`, como era antes), e
# aí quem lê a mensagem tem que reconhecer a string — um `startswith` frágil que uma
# cláusula recuperada com o texto errado poderia acionar por acidente, e que se perde na
# primeira vez que alguém reescreve a frase. O `name` é campo estruturado da própria
# AIMessage: não colide com conteúdo nenhum e é o mesmo mecanismo que já distingue
# `sql_worker`/`rag_worker`/`final`. Fica FORA de `WORKERS` de propósito — uma falha não é
# resultado de worker, e é justamente essa distinção que o synthesizer lê.
WORKER_ERROR = "worker_error"

# A chave, em `AIMessage.additional_kwargs`, onde o worker declara SOBRE O QUE ele
# respondeu — e é o worker quem declara porque é ele quem consultou. É o MESMO caminho que
# já carrega o resultado (a própria mensagem marcada), e é campo ESTRUTURADO pelo mesmo
# motivo de `WORKER_ERROR` ser o `name` e não um prefixo no conteúdo: a alternativa seria o
# synthesizer contar as linhas de `_formatar_hits` de volta, o que amarra a declaração de
# base ao formato de um texto que existe pra ser lido por um LLM. Não é campo do State de
# propósito: isto é um fato sobre UMA consulta, não sobre o grafo, e todo nó futuro teria
# que carregar um campo que não é do domínio dele.
#
# **A ausência da chave significa "não declaro base", e esse default é DELIBERADO.** O
# desenho anterior era o inverso — quem não dissesse nada ganhava o rodapé do corpus —, e
# aí a saída insegura era a padrão: um worker novo que esquecesse de preencher afirmaria,
# em toda resposta, que o corpus inteiro foi lido. Rodapé a menos é uma resposta sem
# procedência; rodapé a mais é uma afirmação falsa de cobertura, que é o único erro que
# este rodapé nunca pode cometer.
BASE = "base_declarada"
BASE_CORPUS = "corpus"          # "respondi agregando sobre as tabelas inteiras"
BASE_RECUPERADO = "recuperado"  # "respondi lendo estas k cláusulas de d documentos"
BASE_PESQUISADO = "pesquisado"  # "procurei em tudo que a busca alcança, e não achei"

# Structured output canônico (mesmo padrão da extração de seguros): expõe UM tool
# cujo input_schema é o JSON Schema do Pydantic e força tool_choice pra ele. O modelo
# é OBRIGADO a devolver argumentos que batem com o schema.
_DECISION_TOOL = "route_decision"
_SQL_TOOL = "emit_sql"

SUPERVISOR_SYSTEM = (
    "Você é o supervisor de um grafo de agentes que responde perguntas sobre CONDIÇÕES "
    "GERAIS de seguro residencial registradas na SUSEP. O corpus descreve PRODUTOS "
    "(o que cada seguradora cobre e exclui), não apólices de clientes — não há preço, "
    "nem dado de cliente, nem sinistro individual.\n\n"
    "Sua função é CLASSIFICAR a pergunta, uma única vez, ANTES de qualquer trabalho: "
    "quem responde depois é o worker que você escolher, e a resposta final é escrita "
    "por outro nó. Você não volta a ser chamado — não há 'próximo passo' pra decidir "
    "depois, e nada do que o worker devolver passa por você.\n\n"
    "  - sql_worker: perguntas de ESTRUTURA, que se respondem contando, filtrando ou "
    "comparando campos categóricos (seguradoras, coberturas, perigos, tipo de franquia). "
    "Ex.: 'quantas seguradoras cobrem vendaval?', 'quais coberturas não têm franquia?', "
    "'liste os perigos da cobertura de incêndio'.\n"
    "  - rag_worker: perguntas de TEOR, que se respondem lendo o texto de uma cláusula. "
    "Ex.: 'em que situações o roubo não é coberto?', 'o que a apólice diz sobre danos "
    "elétricos?', 'chuva que entra por janela aberta é coberta?'.\n"
    "  - unsupported: a pergunta é de OUTRO ASSUNTO — outro ramo de seguro (auto, vida, "
    "saúde, viagem), preço/cotação, ou nada a ver com seguro. Ex.: 'quanto custa meu "
    "seguro?', 'seguro de vida cobre suicídio?', 'qual a capital da França?'.\n\n"
    "ATENÇÃO — o erro mais fácil de cometer aqui: uma pergunta cuja resposta é 'NÃO "
    "COBRE' continua sendo do escopo. 'Enchente é coberta?' se responde com a cláusula "
    "de exclusão de enchente, que existe no corpus. 'Não está coberto' é uma RESPOSTA, "
    "não uma pergunta fora de assunto. Use unsupported só quando o ASSUNTO for outro, "
    "nunca porque você suspeita que a cobertura não existe.\n\n"
    "REGRA DE DESEMPATE: na dúvida entre unsupported e um worker, escolha rag_worker. "
    "Recusar por engano custa a resposta certa a quem tinha uma pergunta legítima; "
    "buscar à toa custa frações de centavo e devolve 'não encontrei'.\n\n"
    # A instrução diz o que NÃO fazer e para. A versão anterior explicava também o que
    # o valor produz ("faz o sistema devolver 'não consegui responder' sem nem tentar")
    # — e isso, na última posição antes do fecho, descrevia pro modelo uma saída de um
    # token pra escapar de responder, competindo com a regra de desempate três linhas
    # acima, que existe justamente pra evitar recusa. Proibir não precisa do prêmio.
    "O schema da tool aceita um quarto valor, 'END', que existe só por compatibilidade: "
    "ignore-o. Toda pergunta cai em uma das três classificações acima.\n\n"
    "Responda SEMPRE chamando a tool route_decision."
)

SQL_SYSTEM = (
    "Você traduz uma pergunta em UMA query SQL (PostgreSQL) sobre o schema fornecido. "
    "Gere apenas UM SELECT que responda a pergunta — sem comentários, sem cercas de "
    "markdown, sem ponto e vírgula. Use exatamente os nomes de tabela/coluna do schema. "
    "Responda SEMPRE chamando a tool emit_sql."
)

SYNTHESIZER_SYSTEM = (
    "Você recebe a pergunta de um usuário e o resultado cru de um worker: linhas de uma "
    "query SQL, ou trechos de cláusulas recuperados das condições gerais. Devolva UMA "
    "frase em pt-BR que responda a pergunta usando esse resultado. Não invente nenhum "
    "dado além do que está no resultado — se ele não responder a pergunta, diga isso em "
    "uma frase. Responda só com a frase, sem preâmbulo, sem repetir o SQL e sem repetir "
    "os identificadores técnicos (chunk/doc/dist) que acompanham as cláusulas."
)


# --- 1. State: the record that travels through the graph ---
class State(TypedDict):
    iterations: int
    next: str
    messages: Annotated[list, add_messages]


# Os destinos que são NÓS de verdade. "unsupported" fica de fora de propósito: ele é uma
# classificação, não um worker — sai pelo synthesizer como qualquer outro fim de grafo.
WORKERS = frozenset({"sql_worker", "rag_worker"})


# --- 2. Supervisor decision: `next` é ENUM = o cinto de segurança ---
class SupervisorDecision(BaseModel):
    # enum = o cinto: o modelo não consegue rotear pra um worker que não existe.
    #
    # `"END"` continua no enum mesmo tendo saído do prompt (o supervisor não encerra mais
    # nada — os workers vão direto pro synthesizer). Ele fica como FAIL-SAFE: um modelo
    # que devolvesse `END` num schema sem essa opção quebraria a validação e derrubaria
    # o request; com ela, cai no mesmo caminho do valor inválido — synthesizer, que
    # escreve `NO_ANSWER` porque não há resultado de worker nenhum.
    next: Literal["sql_worker", "rag_worker", "unsupported", "END"]
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


# --- 2.6 Falha de worker: traceback pro operador, frase genérica pro usuário ---
def _falha_de_worker(worker: str) -> dict:
    """Registra a exceção corrente e devolve a mensagem MARCADA como falha.

    Chamar só de dentro de um `except`: `logger.exception` é o que anexa o traceback,
    e é ele que faz o log valer a pena — o texto da exceção não vai a lugar nenhum além
    daqui.

    **Nada da exceção entra no conteúdo da mensagem, e isso é o ponto da função.** Antes,
    os dois workers devolviam `f"SQL error: {exc}"` / `f"RAG error: {exc}"`, e esse texto
    seguia como qualquer outro resultado: ia pro prompt do synthesizer (que podia
    parafraseá-lo de volta pro usuário) e, no caminho de degradação do synthesizer
    (`frase or resultado`), virava *literalmente* a resposta do `/ask`. O que vazava por
    ali não era ruído inofensivo — `psycopg.OperationalError` carrega host, porta, usuário
    e nome de banco, e um erro de driver carrega caminhos internos. Detalhe de
    infraestrutura é informação do OPERADOR, e o canal do operador é o log.

    `request_id`/`client` vêm dos ContextVars e são as MESMAS chaves de `cost_event`: é o
    que liga este traceback às linhas de custo daquele `/ask` (e à pergunta, e ao cliente).
    Sem elas, com dois requests concorrentes, sobra um traceback solto — e como o usuário
    recebe uma frase genérica, este log é a ÚNICA descrição do que aconteceu.
    """
    logger.exception(
        "%s falhou — devolvendo falha interna. request_id=%s client=%s",
        worker,
        get_request_id(),
        get_client_name(),
    )
    # O conteúdo é genérico de propósito (o synthesizer nem o lê: ele curto-circuita pela
    # marca do `name`), mas nomeia o worker, porque quem lê o histórico do grafo num trace
    # precisa saber QUAL nó caiu — e o nome do nó não é dado da exceção.
    return {
        "messages": [
            AIMessage(content=f"{worker}: falha interna.", name=WORKER_ERROR)
        ]
    }


# --- 2.7 A resposta declara sua BASE, e quem monta isso é o CÓDIGO ---
#
# O sufixo é DETERMINÍSTICO: números contados em Python, colados numa string fixa. Nada
# disso passa pelo modelo, e o prompt do synthesizer não menciona base nenhuma — de
# propósito. Pedir pro LLM declarar a base seria pedir pra ele *afirmar quantos documentos
# consultamos*, que é exatamente o tipo de número que ele inventa com confiança quando não
# está no payload; e quando está, ele ainda pode arredondar, omitir ou reescrever. Uma
# declaração de cobertura errada é pior do que nenhuma: ela dá ao usuário uma razão falsa
# para confiar na resposta. Por isso a frase vem do modelo e o rodapé vem daqui.


async def _base_do_corpus(session) -> int:
    """Quantos PRODUTOS de condições gerais existem na base, agora.

    É a base da rota SQL, que agrega sobre as tabelas do domínio — todas as linhas de
    `policy_document` participam de um `count`/`GROUP BY`, embeddadas ou não. A rota RAG
    tem base MENOR e conta por outra função (`contar_documentos_pesquisaveis`).

    **`distinct susep_process`, e NÃO `count(*)`.** O grão de `policy_document` é
    `(susep_process, version)` — está no `UniqueConstraint` da tabela —, e
    `scripts/susep_harvest.py --all-versions` existe justamente pra baixar o histórico
    completo de cada processo. Num corpus assim, três versões do mesmo produto virariam
    "3 apólices" num rodapé que o usuário lê como três seguradoras diferentes. O produto
    é o processo SUSEP; a versão é a mesma apólice registrada de novo.
    """
    return await session.scalar(
        select(func.count(func.distinct(PolicyDocument.susep_process)))
    )


def _com_base(frase: str, sufixo: str | None) -> str:
    """Cola a declaração de base na frase — o ÚNICO ponto do módulo que monta isso.

    Separador único (linha em branco) porque o rodapé é um bloco à parte, não uma oração
    da resposta: quem lê tem que conseguir ver onde a resposta acaba e onde começa a
    afirmação sobre a cobertura dela. Sem sufixo devolve a frase INTACTA — nem separador
    nem espaço sobrando: quando a contagem falha (ver `_sufixo_do_corpus`), o usuário
    recebe uma resposta normal, e não uma resposta com um rodapé pela metade.
    """
    if not sufixo:
        return frase
    return f"{frase}\n\n{sufixo}"


async def _contando(rotulo: str, conta) -> int | None:
    """Roda uma contagem de base numa sessão própria. **Best effort, como `_record_cost`.**

    Sessão própria e import local, como `record_call_cost` e o `rag_worker`: o nó não
    conhece a borda HTTP e não recebe session por parâmetro, e resolver `SessionLocal` na
    hora da chamada é o que faz o `monkeypatch.setattr("app.db.SessionLocal", ...)` dos
    testes pegar.

    Se a contagem falhar, devolve `None` e a resposta sai SEM rodapé em vez de estourar:
    quando chegamos aqui a pergunta já foi paga e já tem resposta, e trocar uma resposta
    boa por um 500 (ou por uma FALHA_INTERNA) só porque um `count` não pôde ser lido
    destruiria o que já se comprou. Silenciar é a única saída que não mente — um rodapé
    com número errado é pior do que rodapé nenhum.
    """
    from app.db import SessionLocal

    try:
        async with SessionLocal() as session:
            return await conta(session)
    except Exception as exc:  # noqa: BLE001 — best effort é o ponto: nada aqui pode subir
        logger.warning("não consegui contar a base (%s): %s", rotulo, exc)
        return None


async def _sufixo_do_corpus() -> str | None:
    """"Este acervo tem N apólices" — a base da rota SQL.

    **Fala do ACERVO, não do que a query cobriu, e a diferença é o motivo da frase ser
    esta.** A versão anterior dizia "N apólice(s) analisada(s)", e isso era falso na
    pergunta mais comum que existe: o supervisor manda perguntas de FILTRO pro sql_worker
    ("quais coberturas a seguradora X oferece?"), o SELECT lê um documento, e o rodapé
    afirmava que 30 tinham sido analisadas. Saber o escopo real exigiria interpretar o SQL
    que o modelo escreveu — e o `consultou` do worker só distingue "rodou" de "deu erro",
    nunca "cobriu quanto".

    "Corpus de N" é verdadeiro para QUALQUER query, e continua entregando o que o rodapé
    existe pra entregar: a escala do acervo, que sozinha o usuário não tem como conhecer.
    As outras duas bases seguem afirmando cobertura, porque nelas nós sabemos exatamente o
    que foi lido — k cláusulas de d documentos, ou o conjunto inteiro que a busca alcança.

    "apólice(s)" e não "condições gerais": a palavra tecnicamente certa pro corpus é CG,
    mas o rodapé é lido por quem perguntou, não por quem modelou o schema.
    """
    n = await _contando("corpus", _base_do_corpus)
    return None if n is None else f"Base: corpus de {n} apólice(s)."


async def _sufixo_pesquisado() -> str | None:
    """"Procurei em N apólices e não achei" — a base do NADA_RELEVANTE.

    **Não é o mesmo número do corpus, e a diferença é exatamente o que essa frase afirma.**
    A busca semântica só alcança chunk com vetor do modelo atual, então um documento
    extraído e ainda não embeddado (ou um `--remodel` interrompido) está na base e fora do
    alcance. Declarar o corpus aqui diria "procurei nas 40" quando 30 foram procuradas — e
    numa frase cujo trabalho inteiro é dizer "não achei NAS QUE EU TENHO", esse é o pior
    lugar possível pra inflar o número. Quem conta é `contar_documentos_pesquisaveis`, que
    mora junto do filtro da busca justamente pra que os dois não possam divergir.
    """
    n = await _contando("pesquisável", contar_documentos_pesquisaveis)
    return None if n is None else f"Base: {n} apólice(s) pesquisada(s)."


async def _sufixo_da_resposta(msg: AIMessage) -> str | None:
    """A base que ESTA mensagem declara — e nenhuma, se ela não declarar nada.

    Três bases, porque as três saídas cobrem coisas diferentes: o `sql_worker` agrega
    sobre as tabelas (base = corpus), o `rag_worker` responde lendo k cláusulas de d
    documentos (declarar o corpus ali implicaria que as apólices foram lidas inteiras
    quando algumas cláusulas foram), e o "não achei" fala do que a busca ALCANÇA.

    Lê a declaração do `additional_kwargs` da própria mensagem, e nunca o `name` do worker:
    é o worker que sabe se chegou a consultar (um `run_query` que voltou erro NÃO declara
    base) e o que a consulta cobriu. Sem declaração, sem rodapé — ver o comentário de
    `BASE` pra por que o default é esse.

    **Este é o ÚNICO caminho pra um rodapé.** Nenhuma saída pode chamar
    `_sufixo_do_corpus`/`_sufixo_pesquisado` por conta própria: fazer isso reintroduz o
    fail-open que `BASE` existe pra fechar — a saída passaria a declarar uma base fixa
    independentemente do que o worker de fato consultou, que é como o "não achei" ia
    declarar o corpus inteiro no dia em que a busca ganhasse recorte por documento
    (`search_clauses` já aceita `document_ids`).
    """
    base = msg.additional_kwargs.get(BASE)
    if not isinstance(base, dict):
        return None
    tipo = base.get("tipo")
    if tipo == BASE_CORPUS:
        return await _sufixo_do_corpus()
    if tipo == BASE_PESQUISADO:
        return await _sufixo_pesquisado()
    if tipo == BASE_RECUPERADO:
        # O `try` cobre SÓ os dois subscritos, e essa estreiteza é o ponto: eles leem um
        # dicionário montado em OUTRO nó, então um `KeyError` aqui abortaria o grafo no
        # último nó de um request que já tem resposta. Envolver também as chamadas acima
        # (que `_contando` já protege inteiras) faria um defeito real no caminho do corpus
        # sair como "declaração malformada", mandando o operador caçar um payload de
        # worker que não existe.
        try:
            return (
                f"Base: {base['clausulas']} cláusula(s) de {base['documentos']} apólice(s)."
            )
        except Exception as exc:  # noqa: BLE001 — nada aqui pode derrubar o último nó
            logger.warning(
                "declaração de base recuperada malformada (%s); resposta sai sem rodapé", exc
            )
    return None


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

    if decision.next == "END":
        # O prompt manda explicitamente nunca escolher END; chegar aqui é o modelo
        # ignorando a instrução, e o preço é alto e SILENCIOSO: nenhum worker roda e o
        # usuário recebe NO_ANSWER ("não consegui responder") pra uma pergunta que o
        # sistema talvez soubesse responder — indistinguível, de fora, de uma falha de
        # infra. O fail-safe do enum evita o 500; este WARNING é o que evita que ele
        # aconteça sem ninguém ficar sabendo.
        # O `request_id` é o que torna o log utilizável: é a mesma chave de
        # `cost_event.request_id`, então dá pra ligar este WARNING às linhas de custo,
        # ao cliente e à pergunta daquele /ask. Sem ele, com dois requests concorrentes,
        # o operador vê um aviso solto e continua sem saber QUAL resposta foi essa —
        # que é exatamente a ambiguidade que este log existe pra desfazer.
        logger.warning(
            "supervisor devolveu END (proibido pelo prompt) — nenhum worker vai rodar. "
            "request_id=%s client=%s reasoning: %s",
            get_request_id(),
            get_client_name(),
            decision.reasoning,
        )

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

    # O nó INTEIRO roda dentro do try, não só o `get_schema`: qualquer degrau daqui pra
    # baixo pode estourar (conexão de banco, API da Anthropic fora, resposta sem bloco
    # `tool_use`, payload que não valida) e nenhum deles pode subir. Uma exceção que
    # escapa de um nó do LangGraph aborta o grafo, e o `/ask` responde 500 — sem resposta
    # nenhuma pro usuário, e num request que já pode ter pago o supervisor.
    try:
        # 2. Schema numa chamada só — dá os nomes de tabela/coluna pro modelo.
        #    get_schema é psycopg3 SÍNCRONO: roda numa thread pra não travar o event loop.
        schema = await asyncio.to_thread(get_schema)

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

        # 4. Executa pelo guard + role read-only. Contrato do MCP server: erro de QUERY
        #    volta como TEXTO (SQL inválido é resultado — o modelo errou, e o synthesizer
        #    precisa poder dizer isso), erro de CONEXÃO **levanta** e cai no `except` deste
        #    nó. Não confie na versão antiga desta linha ("run_query nunca levanta"): ela é
        #    falsa desde que a divisão entrou, e o `except` largo abaixo depende do
        #    contrário — tirá-lo acreditando nela devolve o 500 numa senha RO rotacionada.
        #    Síncrono também (mesma conexão psycopg3), então vai pra thread igual ao
        #    get_schema.
        rows = await asyncio.to_thread(run_query, sql)
    except Exception:  # noqa: BLE001 — nada sobe daqui: ver _falha_de_worker
        return _falha_de_worker("sql_worker")

    # **Erro de query não declara base.** `run_query` devolve como TEXTO tanto as linhas
    # quanto as recusas do guard e o erro do Postgres (é o contrato: SQL inválido é
    # resultado, o synthesizer precisa poder contar isso). Só que "houve mensagem" não é
    # "houve consulta": num `UndefinedColumn` o SELECT não leu apólice nenhuma, e um
    # rodapé "Base: corpus de N apólice(s)" embaixo da paráfrase desse erro afirmaria
    # uma verificação que não houve. O prefixo vem da CONSTANTE do produtor, não de um
    # literal escrito aqui, pra que os dois lados não possam divergir.
    consultou = not rows.startswith(ERRO_PREFIXO)
    if not consultou:
        # Sem este log, uma resposta SEM rodapé tem três causas indistinguíveis de fora
        # (query recusada, contagem falhou, worker não declarou) e o usuário não recebe
        # sinal nenhum. As outras duas já logam; esta era a que faltava — e é a que
        # aparece em massa quando, por exemplo, a role RO perde SELECT numa tabela.
        logger.info(
            "run_query não consultou (sem declaração de base). request_id=%s client=%s",
            get_request_id(),
            get_client_name(),
        )
    print(f"[sql_worker] SQL: {sql}")
    return {
        "messages": [
            AIMessage(
                content=f"SQL: {sql}\nResult: {rows}",
                name="sql_worker",
                additional_kwargs=({BASE: {"tipo": BASE_CORPUS}} if consultou else {}),
            )
        ]
    }


# --- 4.2 RAG worker: recupera cláusulas por similaridade. Sem LLM próprio. ---
def _formatar_hits(hits) -> str:
    """As cláusulas recuperadas, uma por linha, com a origem colada no texto.

    `chunk_id` e `document_id` viajam junto porque é o que torna a CITAÇÃO possível: são
    eles que ligam a frase de volta à cláusula e ao documento SUSEP de onde ela saiu — o
    motivo de as FKs do arco de `clause_chunk` existirem. A distância entra pra que a
    leitura do log (e do histórico do grafo) mostre *quão* perto o hit estava, que é a
    diferença entre uma resposta bem apoiada e um vizinho que passou raspando no limiar.

    **O `.strip()` no texto ficou DEFENSIVO, e o histórico importa pra ninguém removê-lo
    achando que é enfeite.** Ele foi escrito quando o grafo era cíclico: a mensagem do
    worker voltava pro supervisor como turno `assistant` FINAL, e a API recusa (400) um
    turno assistente final terminando em espaço em branco — `clause_chunk.text` guarda a
    cláusula como a extração devolveu (PDF lido por LLM, que termina em `\\n` com
    facilidade) e a R2a só rejeita texto EM BRANCO, não normaliza o resto. Com o grafo
    single-hop essa mensagem só é lida pelo synthesizer, dentro de um turno `user`, onde
    espaço no fim é inofensivo. Fica porque volta a ser load-bearing no dia do multi-hop
    — que é justamente quando ninguém vai lembrar — e porque texto normalizado é o certo
    de qualquer jeito.
    """
    return "\n".join(
        f"[chunk {h.chunk_id} | doc {h.document_id} | dist {h.distance:.3f}] {h.text.strip()}"
        for h in hits
    )


async def rag_worker(state: State) -> dict:
    """Busca semântica sobre `clause_chunk` e devolve as cláusulas como mensagem.

    Ao contrário dos outros nós, este NÃO chama a Anthropic: a única chamada paga é o
    embedding da pergunta, dentro de `search_clauses` (que também grava o próprio
    `cost_event` como `rag_search`). O trabalho de virar isso em frase é do synthesizer.

    O limiar é aplicado AQUI, e não é opcional: `search_clauses` sempre devolveria `k`
    vizinhos — sem corte, uma pergunta de outro assunto que escapou do supervisor volta
    com 5 cláusulas de seguro residencial e o synthesizer as transforma, confiante, numa
    resposta errada com citação. Com o corte, "nada relevante" é uma saída possível.

    **O corte é feito no nó, não passado como `max_distance`, pra que o descarte seja
    VISÍVEL.** É a mesma operação (a R3a já corta depois do `LIMIT`), mas passando o
    limiar pra dentro da busca as distâncias reprovadas somem, e são exatamente elas que
    dizem se `MAX_DISTANCE_PADRAO` está no lugar certo — o número segue decidido e não
    medido, então uma passada que devolve vazio precisa registrar *por quanto*. Sem isso,
    "não achei" e "achei e cortei a 0,61" ficam indistinguíveis no log.
    """
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    # Sessão PRÓPRIA, como `record_call_cost`: o nó do grafo não conhece a borda HTTP e
    # não recebe session por parâmetro. O import é local pelo mesmo motivo do de
    # `app.cost` — resolve `SessionLocal` no momento da chamada, que é o que faz o
    # `monkeypatch.setattr("app.db.SessionLocal", ...)` dos testes pegar.
    from app.db import SessionLocal

    # O corpo INTEIRO no try, como no sql_worker, e pelo mesmo motivo: qualquer exceção que
    # escape de um nó do LangGraph aborta o grafo e devolve 500. A busca é só o degrau mais
    # provável — o que vem depois dela também toca dado que vem do banco (`h.distance` num
    # `f"{...:.3f}"` estoura com TypeError se algum dia vier `None`, e nada no dataclass
    # `Hit` proíbe isso), e não há razão pra um nó ter duas políticas de erro. Cobrir só a
    # busca era exatamente a assimetria que o sql_worker corrigiu.
    try:
        async with SessionLocal() as session:
            hits = await search_clauses(session, question, k=RAG_K)

        relevantes = [h for h in hits if h.distance <= MAX_DISTANCE_PADRAO]
        print(
            f"[rag_worker] {len(relevantes)}/{len(hits)} cláusula(s) dentro do limiar "
            f"{MAX_DISTANCE_PADRAO}"
        )
        if not relevantes:
            if hits:
                # O único registro de quanto faltou. Enquanto o limiar não for calibrado,
                # é este número que diz se ele está apertado demais ou se a pergunta era
                # mesmo de outro mundo.
                logger.info(
                    "busca RAG sem hits: melhor distância %.3f > limiar %.2f",
                    hits[0].distance,
                    MAX_DISTANCE_PADRAO,
                )
            # Mensagem EXPLÍCITA, nunca string vazia: uma mensagem em branco no histórico é
            # indistinguível de "o worker não rodou", e o synthesizer cairia no NO_ANSWER
            # ("falhei") em vez de dizer a verdade ("procurei e não achei").
            #
            # E ela DECLARA a base, como qualquer outra saída de worker: "procurei em N e
            # não achei" é afirmação sobre o que foi varrido, e quem varreu é este nó. O
            # synthesizer não pode escolher essa base sozinho — no dia em que a busca usar
            # `document_ids` (a assinatura já aceita), o recorte é conhecido aqui e não lá,
            # e um sufixo global embaixo de uma busca de um documento diria "procurei em
            # 30" com 1 procurado.
            return {
                "messages": [
                    AIMessage(
                        content=NADA_RELEVANTE,
                        name="rag_worker",
                        additional_kwargs={BASE: {"tipo": BASE_PESQUISADO}},
                    )
                ]
            }
        return {
            "messages": [
                AIMessage(
                    content=f"Cláusulas recuperadas:\n{_formatar_hits(relevantes)}",
                    name="rag_worker",
                    # A base viaja com a recuperação, e é contada sobre `relevantes` — o que
                    # sobrou DEPOIS do corte —, nunca sobre `hits`. Contar antes inflaria o
                    # rodapé com cláusulas que o synthesizer nunca viu: "5 cláusulas de 3
                    # apólices" numa resposta escrita em cima de uma só é uma afirmação de
                    # cobertura falsa, e é justamente a que faz o usuário confiar mais.
                    additional_kwargs={
                        BASE: {
                            "tipo": BASE_RECUPERADO,
                            "clausulas": len(relevantes),
                            "documentos": len({h.document_id for h in relevantes}),
                        }
                    },
                )
            ]
        }
    except Exception:  # noqa: BLE001 — mesmo contrato do sql_worker: erro vira mensagem
        # Falha de infra vira mensagem MARCADA (o grafo não estoura no meio de um request
        # que já pode ter gasto em outros nós), e o texto da exceção fica no log. Aqui ele
        # era especialmente ruim de vazar: uma `voyageai.error.AuthenticationError` fala
        # de chave de API, e um erro de conexão do asyncpg fala do host do Postgres.
        return _falha_de_worker("rag_worker")


# --- 4.5 Synthesizer: último nó SEMPRE. Vira o resultado cru em frase. ---
async def synthesizer(state: State) -> dict:
    """Escreve a resposta final em linguagem natural.

    Roda em todo caminho que termina o grafo, inclusive quando nenhum worker rodou —
    por isso é ele, e não o supervisor, quem produz a última mensagem. A API só precisa
    ler `messages[-1]`.

    Sem chamada de LLM em NENHUM dos quatro caminhos estáticos (falha interna, nada
    encontrado, fora de escopo, nenhum worker): não há o que sintetizar, e pagar uma
    chamada só pra reescrever uma frase que já está pronta seria queimar dinheiro à toa.

    É também aqui que a resposta ganha o rodapé que DECLARA SUA BASE — montado em código,
    nunca pelo modelo (ver a seção 2.7). Quem consultou declara; quem não consultou, não.
    """
    question = next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )

    # **Só o que ESTE turno produziu.** A fatia começa depois da última `HumanMessage`, que
    # é a mesma pergunta lida acima — o que está antes dela foi respondido em outro turno.
    # Hoje é no-op: o `/ask` monta um State novo, com uma `HumanMessage` só, por request.
    # Vira load-bearing quando o histórico atravessar turnos (a superfície de WhatsApp), e
    # o modo de falha é do tipo que ninguém liga a esta função: uma falha de worker do
    # turno 1 faria o turno 5 responder FALHA_INTERNA sobre um resultado perfeitamente
    # bom — e mandaria o operador procurar um traceback de OUTRO request. Vale igual pros
    # resultados: sem a fatia, um turno sem worker nenhum sintetizaria em cima da resposta
    # anterior, com a confiança de quem acabou de consultar.
    i_pergunta = max(
        (i for i, m in enumerate(state["messages"]) if isinstance(m, HumanMessage)),
        default=-1,
    )
    turno = state["messages"][i_pergunta + 1:]

    # Os resultados de worker, do mais recente pro mais antigo. Fixar `sql_worker` faria
    # uma resposta de RAG ser sintetizada em cima de um SQL antigo do mesmo histórico —
    # ou, sem SQL nenhum, cair no NO_ANSWER com as cláusulas certas na mensagem anterior.
    # `WORKERS` não inclui `WORKER_ERROR`: falha não é resultado, e é por isso que ela
    # não aparece aqui nem pode ser escolhida como payload de síntese.
    #
    # Guardamos as MENSAGENS, não só o `.content`: é no `additional_kwargs` delas que o
    # worker declarou o que a resposta cobre (ver `BASE`), e a base tem que sair da mesma
    # mensagem que virou payload — não de "a última que passou por aqui".
    resultados = [
        m for m in reversed(turno) if isinstance(m, AIMessage) and m.name in WORKERS
    ]
    # As falhas de worker, reconhecidas pelo `name` — nunca por prefixo do conteúdo. O
    # conteúdo delas não é lido em lugar nenhum: o que importa é que existiu falha. É
    # `any`, e não "a mais recente": com dois workers no turno, um que caiu invalida a
    # afirmação do outro (ver a ordem das frases abaixo), então a ordem entre eles não muda
    # nada — o que importa é o turno ter tido falha.
    falhou = any(isinstance(m, AIMessage) and m.name == WORKER_ERROR for m in turno)

    # A busca rodou e não achou nada dentro do limiar: a frase do worker já É a resposta
    # final. Mas ela só vale como final se for o ÚNICO resultado — se um sql_worker
    # respondeu antes, a resposta está lá, e dizer "não encontrei nada" jogaria fora o
    # dado certo. Daí separar o que é resultado SUBSTANTIVO do que é "procurei e nada".
    substantivos = [m for m in resultados if m.content != NADA_RELEVANTE]

    # **Nenhuma frase estática vale quando há trabalho a apresentar** — só entramos aqui
    # sem nada substantivo na mão. Dentro, a ordem é da mais específica sobre o que
    # aconteceu pra menos, e cada degrau tem um motivo:
    #
    #  1. FALHA_INTERNA na frente de tudo. Se um worker caiu, qualquer outra frase mente
    #     sobre a causa: "não achei nada no corpus" e "não trato desse assunto" são
    #     afirmações sobre o corpus e sobre o escopo, e nenhuma das duas foi verificada
    #     quando a consulta nem chegou a rodar. FORA_DE_ESCOPO seria a pior, porque manda
    #     o usuário embora por causa de um banco fora do ar.
    #  2. NADA_RELEVANTE só quando a busca de fato rodou (é o que `resultados` não-vazio
    #     significa neste ponto: todos os resultados eram NADA_RELEVANTE).
    #  3. FORA_DE_ESCOPO depois dos resultados, e não antes. Com o grafo single-hop `next`
    #     e os resultados não podem se contradizer (o supervisor classifica ANTES de
    #     qualquer worker), então hoje essa ordem é redundante — e fica porque o multi-hop
    #     a torna load-bearing de novo: com o supervisor decidindo depois de cada worker,
    #     nada o impede de classificar como `unsupported` na segunda passada, lendo
    #     justamente o "não encontrei" do rag_worker. Invertida, uma pergunta legítima
    #     recebia "só respondo sobre seguro residencial" DEPOIS de a busca ter sido paga.
    #     O teste chama o synthesizer direto, com o estado que o multi-hop reintroduz —
    #     via /ask o caso é inalcançável hoje, e um teste por /ask passaria sem exercitar
    #     nada.
    #  4. NO_ANSWER é o resto: ninguém trabalhou e nada quebrou — falha de roteamento.
    #
    # **Só uma das quatro declara base, e a regra é: só declara quem CONSULTOU.**
    #  - FALHA_INTERNA: a consulta não chegou a rodar. Um "Base: corpus de N apólice(s)"
    #    embaixo de "tive uma falha interna" afirmaria uma verificação que não houve — e
    #    ainda por cima daria um ar de completude à única frase que significa o contrário.
    #  - FORA_DE_ESCOPO: o supervisor classificou ANTES de gastar qualquer coisa; nada foi
    #    aberto, e a frase é sobre o sistema, não sobre o corpus.
    #  - NO_ANSWER: ninguém trabalhou (falha de roteamento). Mesma razão.
    #  - NADA_RELEVANTE: esta SIM. É a única afirmação sobre o CORPUS, e sem a base ela lê
    #    como "isso não existe" quando o que ela quer dizer é "não achei nas que eu tenho".
    #    O rodapé é o que separa as duas leituras, e a diferença é enorme pra quem pergunta
    #    sobre uma cobertura que só existe numa apólice que ainda não foi indexada. Ele usa
    #    a base PESQUISÁVEL, não a do corpus — ver `_sufixo_pesquisado`.
    if not substantivos:
        if falhou:
            print("[synthesizer] falha de worker -> frase fixa, sem LLM")
            return {"messages": [AIMessage(content=FALHA_INTERNA, name="final")]}
        if resultados:
            print("[synthesizer] busca sem hits -> frase do worker, sem LLM")
            # A base sai da MENSAGEM (a mais recente), pelo mesmo caminho de toda outra
            # saída — nunca de uma chamada direta a `_sufixo_pesquisado` aqui, que voltaria
            # a fixar uma base global independente do que o worker varreu.
            frase = _com_base(NADA_RELEVANTE, await _sufixo_da_resposta(resultados[0]))
            return {"messages": [AIMessage(content=frase, name="final")]}
        if state.get("next") == "unsupported":
            print("[synthesizer] fora de escopo -> frase fixa, sem LLM")
            return {"messages": [AIMessage(content=FORA_DE_ESCOPO, name="final")]}
        print("[synthesizer] sem resultado de worker -> fallback estático")
        return {"messages": [AIMessage(content=NO_ANSWER, name="final")]}
    msg_resultado = substantivos[0]
    resultado = msg_resultado.content

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
                    # "Resultado do worker", não "da query": com o rag_worker no grafo o
                    # payload tanto pode ser linha de SELECT quanto cláusula recuperada,
                    # e rotular cláusula como resultado de query faz o modelo escrever
                    # "a consulta retornou..." sobre uma consulta que não existiu.
                    "content": f"Pergunta: {question}\n\nResultado do worker:\n{resultado}",
                }
            ],
        )
        await _record_cost("synthesizer", resp)
        frase = "".join(b.text for b in resp.content if b.type == "text").strip()
    except Exception as exc:  # noqa: BLE001 — o dado já está na mão; não morrer no último nó
        logger.warning("synthesizer falhou (%s); caindo no texto cru do worker", exc)

    # `or resultado` cobre os dois modos de falha (exceção e resposta vazia): o texto
    # cru é feio, mas é uma resposta REAL e já paga — melhor que estourar o grafo.
    #
    # O rodapé entra nos DOIS casos, e não só na frase bonita: a base é uma afirmação sobre
    # o que foi consultado, e o que foi consultado não muda quando o synthesizer cai. A
    # degradação já entrega um texto feio; entregá-lo também sem procedência seria tirar
    # justamente a informação que torna um despejo cru interpretável.
    #
    # **LIMITAÇÃO CONHECIDA, e ela vence no dia do multi-turno:** o rodapé é gravado DENTRO
    # do `content` da mensagem `final`, ou seja, dentro do histórico. Hoje isso é inócuo (o
    # `/ask` monta um State novo por request, e nada relê a resposta anterior), mas na
    # superfície de WhatsApp o supervisor converte o histórico inteiro em turnos e o número
    # que este desenho mantém FORA do contexto do modelo volta pra dentro dele como texto
    # de assistente — de onde dá pra parafrasear, arredondar ou carregar pra um turno em
    # que worker nenhum rodou. A saída é o rodapé viajar fora do `content` (em
    # `additional_kwargs`, com a junção na borda HTTP), e ela é da fatia do multi-turno
    # porque é lá que se decide o que o supervisor enxerga do histórico.
    print(f"[synthesizer] resposta final {'sintetizada' if frase else 'CRUA (degradou)'}")
    final = _com_base(frase or resultado, await _sufixo_da_resposta(msg_resultado))
    return {"messages": [AIMessage(content=final, name="final")]}


# --- 5. Conditional edge: routes by reading State (with fail-closed fallback) ---
def route(state: State) -> str:
    # O circuit breaker FICA, e hoje é inalcançável pelo caminho normal: com os workers
    # indo direto pro synthesizer, `route` roda uma vez só por request e `iterations`
    # nunca passa de 1. Ele não é resto de código — é o que segura um ciclo reintroduzido
    # por engano (uma aresta de worker apontando de volta pro supervisor volta a ser
    # possível numa linha), e guarda que nunca dispara continua sendo guarda. O teste
    # dele força `iterations` direto no State justamente por isso.
    if state["iterations"] >= MAX_ITERATIONS:  # mechanical guard: does not ask the LLM
        print("[route] circuit breaker -> END")
        return END
    nxt = state["next"]
    if nxt in WORKERS:
        return nxt
    # "END" (fail-safe do enum) e "unsupported" saem os dois pelo mesmo lugar: o
    # synthesizer. A diferença entre eles é só a FRASE, e quem a escolhe é o synthesizer
    # lendo `state["next"]` — route() decide o CAMINHO, não o texto.
    if nxt not in ("END", "unsupported"):  # suspenders: o enum deveria impedir, mas se escapar → END
        print(f"[route] invalid next '{nxt}' -> END (fail closed)")
    return END


# --- 6. Build the graph ---
builder = StateGraph(State)
builder.add_node("supervisor", supervisor)
builder.add_node("sql_worker", sql_worker)
builder.add_node("rag_worker", rag_worker)
builder.add_node("synthesizer", synthesizer)

builder.set_entry_point("supervisor")
# O route() devolve END pro que não é worker, e o mapa traduz esse END no nó de saída —
# `unsupported` não aparece aqui de propósito, ele já virou END lá dentro.
builder.add_conditional_edges("supervisor", route, {
    "sql_worker": "sql_worker",
    "rag_worker": "rag_worker",
    END: "synthesizer",
})

# **O grafo é SINGLE-HOP, e a terminação é ESTRUTURAL.** Os workers vão direto pro
# synthesizer; nenhuma aresta volta pro supervisor. Antes voltavam, e a medição num
# /ask real mostrou o preço: o supervisor re-classificava a pergunta a cada volta,
# reroteava pro MESMO worker e só parava no circuit breaker — 10 chamadas de LLM onde
# bastavam 3, 8.790 tokens de supervisor contra 4.204 do worker que fez o trabalho, e
# na rota RAG 4 embeddings pagos com 3 jogados fora. A instrução de prompt "encerre se
# um worker já respondeu" reduzia a chance disso, não o custo do caso ruim: pedir pro
# modelo não repetir é mais frágil do que não lhe dar a chance.
#
# O que se perde é o multi-hop — uma pergunta composta ("quantas seguradoras cobrem
# vendaval E o que a cláusula diz da franquia?") hoje é respondida por um worker só.
# Isso é FATIA PRÓPRIA, e a razão de não voltar a aresta é justamente essa: multi-hop
# exige uma regra de parada explícita (quem decide que já basta, e com base em quê),
# e herdar o ciclo "de graça" é como se chegou às 10 chamadas.
builder.add_edge("sql_worker", "synthesizer")
builder.add_edge("rag_worker", "synthesizer")
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
