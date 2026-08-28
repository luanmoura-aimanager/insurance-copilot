import asyncio
import json
import logging
from uuid import uuid4

# Configurado ANTES dos outros imports pra que o `basicConfig` que o FastMCP dispara (via
# `app.agents.graph` -> `mcp_servers.postgres_mcp_server`, algumas linhas abaixo) vire no-op,
# e pra que o que for logado DURANTE os imports restantes já saia formatado. O porquê
# completo — e por que a ordem NÃO é o que decide quem vence — está em app/logging_config.py.
from app.logging_config import configure_logging, resumo_do_erro, traceback_da_cadeia

configure_logging()

from fastapi import (  # noqa: E402
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import PlainTextResponse
from starlette.datastructures import MutableHeaders
from starlette.requests import ClientDisconnect
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, DisconnectionError, InterfaceError
from sqlalchemy.exc import TimeoutError as SATimeoutError

from app import inbox
from app.agents.answer import final_answer
from app.agents.context import get_request_id, reset_request_context, set_request_context
from app.agents.graph import graph
from app.auth import require_client
from app.db import get_session
from app.limits import WEBHOOK_ANCORA, ask_client_limit, client_key, limiter, whatsapp_limit
from app.models import WhatsAppMessage
from app.whatsapp import (
    ASSINATURA_HEADER,
    MAX_BODY_BYTES,
    IncomingMessage,
    extract_messages,
    id_curto,
    mascarar_telefone,
    verify_challenge,
    verify_signature,
)

logger = logging.getLogger(__name__)

app = FastAPI()

# O slowapi lê o limiter de app.state e precisa do handler pra transformar o
# RateLimitExceeded (que é uma HTTPException 429) em resposta.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# O teto por IP é global e vive AQUI, no middleware, não num decorator de rota: o
# middleware roda antes do roteamento e das dependencies, então também conta os
# requests que morrem no 401 da auth. Como decorator ele nunca seria alcançado —
# require_client levanta antes — e dava pra martelar /ask com token inválido de graça.
app.add_middleware(SlowAPIMiddleware)


HEADER_REQUEST_ID = "x-request-id"


class RequestIdMiddleware:
    """Cunha um id por request, fixa nos ContextVars e devolve no `X-Request-Id`.

    **Middleware ASGI puro, e não `BaseHTTPMiddleware`.** O `BaseHTTPMiddleware` roda o app
    de baixo numa task filha, então o ContextVar que ele fixa vira uma CÓPIA: a leitura
    funciona, mas o `reset` do `finally` acontece num contexto diferente do endpoint — o
    tipo de semântica que quebra em silêncio. ASGI puro fixa a var no mesmo contexto que o
    resto do stack herda, e de quebra não acrescenta um task group + par de memory streams
    por request.

    Registrado DEPOIS do `SlowAPIMiddleware`, então fica por FORA dele (o Starlette roda o
    último adicionado como o mais externo): o id cobre também o 429 do teto por IP e o 401
    da auth, que são justamente as respostas em que quem reclama não tem outra pista.

    O id é sempre NOSSO — um `X-Request-Id` de entrada é texto escolhido pelo cliente, e ele
    iria parar em `cost_event.request_id` e no log. Com `--forwarded-allow-ips=*` no
    Procfile o header já é escolhido pelo cliente e não pelo proxy, então não haveria o que
    validar.

    Numa exceção NÃO tratada quem responde é o `ServerErrorMiddleware`, que é mais externo
    que qualquer user middleware — então **o 500 sai sem o header**, e não há como mudar isso
    daqui. O que dá pra garantir é o outro lado: o `except` abaixo registra a exceção AINDA
    DENTRO do contexto, antes do `reset` do `finally`. Sem ele, o único log daquele crash
    seria o `"Exception in ASGI application"` do uvicorn, emitido depois do reset e portanto
    **sem `request_id`** (verificado) — ou seja, a resposta que mais precisa ser rastreável
    seria a única sem header E sem correlação. O preço é uma linha a mais por crash, e ela é
    a que tem o id.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        # `client` entra como None: quem autentica é a dependency da rota, que roda bem
        # depois daqui. O `/ask` refixa o par assim que sabe quem perguntou.
        ctx = set_request_context(request_id, None)

        async def send_com_id(message):
            if message["type"] == "http.response.start":
                # `headers` é OPCIONAL no http.response.start (default `[]`), e
                # `MutableHeaders` indexa a chave sem guarda — um app ASGI montado aqui
                # dentro que a omitisse viraria KeyError DEPOIS de a resposta ter começado,
                # que é o pior ponto possível pra falhar.
                message.setdefault("headers", [])
                MutableHeaders(scope=message)[HEADER_REQUEST_ID] = request_id
            await send(message)

        try:
            await self.app(scope, receive, send_com_id)
        except Exception:
            # Dentro do contexto de propósito — ver o docstring. Re-levanta: quem transforma
            # isso em 500 é o ServerErrorMiddleware, e engolir aqui devolveria uma resposta
            # vazia no lugar do erro.
            logger.exception(
                "exceção não tratada em %s %s", scope.get("method"), scope.get("path")
            )
            raise
        finally:
            reset_request_context(ctx)


app.add_middleware(RequestIdMiddleware)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    iterations: int


# `_final_answer` MUDOU DE CASA: agora é `app.agents.answer.final_answer`. Ela passou a ter
# DOIS leitores — o /ask e a varredura do WhatsApp — e uma cópia divergiria no dia em que o
# nome da mensagem terminal do grafo mudasse. Mesmo argumento do `ERRO_PREFIXO` exportado pelo
# produtor e do `_filtro_pesquisavel` compartilhado. Ver o docstring lá, que explica também
# por que falha de worker sai 200 (aqui) e não vira silêncio (no WhatsApp).


# /health e /health/db ficam ABERTOS de propósito: é o healthcheck do Railway, que
# não manda Authorization — protegê-los derrubaria o deploy. Nenhum dos dois gasta
# LLM nem devolve dado do domínio.
#
# @limiter.exempt os tira também do teto GLOBAL por IP: o healthcheck bate neles em
# loop, e cota consumida por healthcheck viraria 429 pro Railway, que então derruba
# um serviço saudável. O exempt vem por DENTRO do @app.get pra que a rota registrada
# seja a função já marcada como isenta.
@app.get("/health")
@limiter.exempt
def health():
    return {"status": "ok"}


@app.get("/health/db")
@limiter.exempt
async def health_db(session=Depends(get_session)):
    await session.execute(text("SELECT 1"))
    return {"db": "ok"}


@app.post("/ask", response_model=AskResponse)
# Só a cota por CLIENTE fica aqui: ela depende de request.state.client_name, que só
# existe depois da auth. O teto por IP é global (middleware, acima).
#
# Os dois NÃO contam em dobro, e isso foi lido na fonte do slowapi 0.1.10:
# _check_request_limit só soma os default_limits quando `combined_defaults` é True,
# ou seja, quando todo limite da rota tem override_defaults=False. O .limit() usa
# override_defaults=True por padrão, então a passagem pelo decorator avalia APENAS a
# cota por cliente, e a passagem pelo middleware avalia APENAS o default por IP.
# (test_limite_por_ip_429 fixa isso: com 1/minute, o 1º request tem que passar —
# se contasse duas vezes, ele já sairia 429.)
@limiter.limit(ask_client_limit, key_func=client_key)
async def ask(
    request: Request,                                  # exigido pelo slowapi
    req: AskRequest,
    client: str = Depends(require_client),
) -> AskResponse:
    # require_client já gravou isso em request.state (antes do wrapper do slowapi);
    # repetimos aqui só pra deixar o vínculo explícito na leitura da rota.
    request.state.client_name = client

    # Contexto que os nós do grafo leem pra atribuir o custo: o id do request
    # (correlaciona as N chamadas de LLM de um mesmo /ask) e QUEM pediu. O reset no
    # finally é obrigatório — sem ele o valor sobreviveria ao request nesta task e
    # vazaria pro próximo que a reaproveitasse.
    #
    # O id vem do RequestIdMiddleware, não é cunhado aqui: ele já correlacionou o request
    # inteiro (o log de acesso, um 401, um 429) e é o mesmo que volta no `X-Request-Id`.
    # Cunhar outro neste ponto partiria em dois o que o `cost_event` e o traceback de
    # `_falha_de_worker` usam pra se encontrar. O `or` é cinto pra quem chame o endpoint
    # direto, fora do stack ASGI.
    ctx = set_request_context(get_request_id() or str(uuid4()), client)
    try:
        state = await graph.ainvoke({
            "iterations": 0,
            "next": "",
            "messages": [HumanMessage(content=req.question)],
        })
    finally:
        reset_request_context(ctx)
    return AskResponse(answer=final_answer(state), iterations=state["iterations"])


# ---------------------------------------------------------------------------
# Superfície do WhatsApp (Meta Cloud API) — RECEBER, GUARDAR e RESPONDER.
#
# W1: a rota prova que o evento veio da Meta (HMAC sobre o corpo cru). W2a: a mensagem de
# texto vira linha em `whatsapp_message`, deduplicada por `wamid`, e o 200 passa a significar
# "aceito com durabilidade" em vez de "li o que deu". W2b: depois do commit, a rota AGENDA a
# varredura (`app/inbox.py`) que chama o grafo e responde pela Cloud API.
#
# A rota em si continua não chamando o grafo nem enviando nada — ela só agenda. Quem faz o
# trabalho roda depois do 200, numa sessão própria, e pode ser disparado também pelo
# `scripts/process_pending.py`. Ver o docstring de `app/inbox.py`.
#
# É o primeiro endpoint público do projeto: a Meta não manda `Authorization`, então
# quem faz o papel do Bearer aqui é a assinatura HMAC do corpo (`app/whatsapp.py`).
# ---------------------------------------------------------------------------


async def _corpo_com_teto(request: Request) -> bytes | None:
    """Lê o corpo até `MAX_BODY_BYTES`; devolve None se passar disso.

    Substitui o `await request.body()` (que bufferiza o stream inteiro) porque aqui a
    leitura acontece ANTES de qualquer autenticação — o HMAC precisa dos bytes pra
    existir, então o que dá pra fazer não é verificar antes de ler, é limitar a
    leitura. O `Content-Length` sozinho não basta: em `Transfer-Encoding: chunked` ele
    nem vem, e é o cliente que o escolhe — daí contar também enquanto se lê.

    Três detalhes carregam peso, e os três são alcançáveis por qualquer anônimo.

    (1) O `int()` vai dentro de `try`: `"²".isdigit()` é **True** e `int("²")` levanta
    `ValueError` (`isdecimal` seria False, mas `try` não depende de eu ter escolhido o
    predicado certo). Um `Content-Length: ²` estourava 500 com traceback, antes da
    checagem de assinatura, na rota pública.

    (2) `ClientDisconnect` é capturado: `request.stream()` levanta quando o cliente
    aborta no meio do corpo, e nada tratava isso — 500 e traceback por request, na
    ÚNICA rota cujo contrato é nunca responder não-2xx. Pior no caso real: se o
    cliente da própria Meta expirar no meio da entrega, o 500 é o que empurra ela pra
    desabilitar a inscrição. Corpo incompleto vira corpo vazio, que não passa no HMAC.

    (3) `request._body` é preenchido no fim. `Request.stream()` marca o stream como
    consumido sem popular `_body`, então um `await request.body()` posterior levantaria
    `RuntimeError: Stream consumed` — e quem faria isso é justamente quem vier depois
    (uma dependency do FastAPI que leia o corpo, um middleware de log, a W2). Guardar o
    resultado deixa o `Request` no mesmo estado em que o `body()` o deixaria.
    """
    declarado = request.headers.get("content-length")
    if declarado is not None:
        try:
            if int(declarado) > MAX_BODY_BYTES:
                return None
        except ValueError:
            pass

    partes: list[bytes] = []
    tamanho = 0
    try:
        async for pedaco in request.stream():
            tamanho += len(pedaco)
            if tamanho > MAX_BODY_BYTES:
                return None
            partes.append(pedaco)
    except ClientDisconnect:
        logger.warning("webhook whatsapp: cliente desconectou no meio do corpo")
        partes = []

    raw = b"".join(partes)
    request._body = raw
    return raw


def _perguntas(mensagens: list[IncomingMessage]) -> list[IncomingMessage]:
    """As mensagens que viram linha: texto, com texto de verdade, sem wamid repetido.

    Só mensagem de TEXTO é persistida porque é ela que carrega a pergunta — áudio,
    imagem e botão ficam só no log, já que a W2b não teria o que perguntar ao grafo a
    partir deles.

    O teste é `m.text` e não `m.text is not None`: `tipo == "text"` com `text.body`
    ausente ou `""` sai da borda como texto vazio (ver `_mensagem` em app/whatsapp.py),
    e string vazia não tem conteúdo nenhum pra responder. Mesma truthiness do guard de
    `id`/`from` lá.

    O filtro para AÍ de propósito: o que tem conteúdo é gravado como veio, inclusive um
    "ok" de dois caracteres e um texto de 4.000 (o WhatsApp permite 4.096). Seria
    tentador espelhar os limites do `AskRequest` (3..500), mas eles são do `/ask`, não do
    canal — descartar na borda uma mensagem que o usuário de fato mandou é pior do que
    guardá-la: o remetente fica sem resposta E sem registro de que falou. Quem decide o
    que consegue responder, e como (recusar, pedir pra encurtar), é a W2b, com a linha
    na mão.

    A deduplicação por `wamid` DENTRO do lote é cinto: o `ON CONFLICT DO NOTHING`
    aguenta duplicata no mesmo statement, mas com o wamid repetido o `RETURNING` traria
    só uma das duas linhas, e a contagem de "novas" do log sairia menor do que foi.
    """
    vistos: set[str] = set()
    saida: list[IncomingMessage] = []
    for m in mensagens:
        if m.tipo != "text" or not m.text or m.wamid in vistos:
            continue
        vistos.add(m.wamid)
        saida.append(m)
    return saida


# Classes de SQLSTATE que uma REENTREGA tem chance de resolver. É por SQLSTATE, e não
# por classe de exceção do SQLAlchemy, porque **com asyncpg a classe não discrimina**:
# medido nesta base — `SELECT pg_sleep` estourando statement_timeout e um texto com NUL
# chegam os DOIS como `sqlalchemy.exc.DBAPIError`, e o primeiro é transitório enquanto o
# segundo nunca vai entrar. O `sqlalchemy.exc.OperationalError` (o palpite óbvio, e o que
# esta função checava antes) o driver asyncpg **não emite nunca**: o translator dele não
# tem entrada pra ele, então todo erro do servidor que não case por nome vira o
# `DBAPIError` genérico. Só o SQLSTATE separa os dois.
#
#   08 conexão      53 recursos insuficientes (too_many_connections)
#   57 intervenção do operador (admin_shutdown, cannot_connect_now, query_canceled)
#   40 rollback de transação (serialization_failure, deadlock_detected)
#
# É o mesmo instrumento que o `run_query` usa pra separar os limites da classe 54 do
# resto — lá pra escolher entre texto e exceção, aqui pra escolher entre 200 e 500.
_SQLSTATE_TRANSITORIO = ("08", "53", "57", "40")

# Falha de INFRAESTRUTURA que nem chega a virar erro do Postgres: com o banco fora do ar
# o asyncpg levanta `ConnectionRefusedError` CRU (medido) — o SQLAlchemy não embrulha,
# porque não é instância do Error do DBAPI. Este era o buraco mais grave da versão
# anterior: o caso mais provável de todos (banco indisponível) caía no ramo PERMANENTE e
# a mensagem era descartada com 200.
_ERROS_DE_INFRA = (OSError, ConnectionError, asyncio.TimeoutError)


def _vale_reentregar(exc: BaseException) -> bool:
    """Se a Meta mandar este lote de novo, tem chance de dar certo?

    Errar pra 500 num erro PERMANENTE faz a Meta reentregar, falhar igual, e depois de
    repetidas falhas **desabilitar a inscrição** — trocando a perda de UMA mensagem pela
    de TODAS as futuras. Errar pra 200 num TRANSITÓRIO descarta uma pergunta que teria
    entrado no próximo try. Nenhum dos dois lados é seguro por default, e é por isso que
    a classificação olha o SQLSTATE em vez de chutar pela classe.
    """
    if isinstance(exc, _ERROS_DE_INFRA):
        return True
    # Conexão morta no meio do statement (banco reiniciado) chega como InterfaceError;
    # SATimeoutError é o checkout do pool estourando, que passa quando o pool desafoga.
    if isinstance(exc, (InterfaceError, DisconnectionError, SATimeoutError)):
        return True
    if isinstance(exc, DBAPIError):
        # asyncpg põe o SQLSTATE no erro original; o `orig` do SQLAlchemy é ele.
        sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", None)
        if isinstance(sqlstate, str) and sqlstate[:2] in _SQLSTATE_TRANSITORIO:
            return True
        # Uma causa de infra pode vir embrulhada (o asyncpg levanta OSError e o
        # SQLAlchemy embrulha ao invalidar a conexão).
        if isinstance(getattr(exc, "orig", None), _ERROS_DE_INFRA):
            return True
    return False


# `_resumo_do_erro` MUDOU DE CASA: agora é `app.logging_config.resumo_do_erro`, importado no
# topo, ao lado do `traceback_da_cadeia` que a mesma regra produziu. Ela passou a ter dois
# chamadores (esta rota e a varredura do inbox) e a duplicação divergiria no dia em que uma
# delas endurecesse. Sem alias local: dar dois nomes à mesma função no mesmo arquivo é a
# duplicação que a mudança de casa foi feita pra remover.


async def _registrar_mensagens(session, perguntas: list[IncomingMessage]) -> set[str]:
    """INSERT idempotente do lote; devolve os `wamid` que entraram AGORA.

    `ON CONFLICT DO NOTHING` sobre o unique de `wamid` é o que torna a reentrega da Meta
    inofensiva — e o `RETURNING` só devolve linha de fato inserida, então o que não
    voltou é reentrega. É daí que sai o "nova" vs "duplicada" do log, sem uma segunda
    consulta.

    Um statement por entrega, de propósito: ou o lote inteiro entra, ou nada entra e a
    Meta reentrega tudo. Não existe estado pela metade pra W2b encontrar.

    NÃO commita — quem é dono da transação é a rota, que precisa distinguir "gravou" de
    "falhou" pra escolher entre 200 e 500.
    """
    stmt = (
        pg_insert(WhatsAppMessage)
        .values([
            {
                "wamid": m.wamid,
                "from_phone": m.from_phone,
                "text": m.text,
                "phone_number_id": m.phone_number_id,
            }
            for m in perguntas
        ])
        .on_conflict_do_nothing(index_elements=["wamid"])
        .returning(WhatsAppMessage.wamid)
    )
    return set((await session.execute(stmt)).scalars())


# O handshake de verificação, que a Meta dispara UMA vez ao cadastrar a URL.
#
# Fica de propósito SEM decorator de limite, ou seja, coberto pelo teto por IP do
# middleware — o oposto do POST logo abaixo. Ninguém legítimo chama isto em loop, e é
# o único ponto do sistema onde um segredo pode ser adivinhado por tentativa
# (`hub.verify_token`): ali o teto é proteção, não estorvo.
@app.get("/webhook/whatsapp", response_class=PlainTextResponse)
async def whatsapp_verify(
    # Os nomes de verdade têm ponto (`hub.mode`), que não é identificador Python.
    hub_mode: str | None = Query(None, alias="hub.mode"),
    hub_verify_token: str | None = Query(None, alias="hub.verify_token"),
    hub_challenge: str | None = Query(None, alias="hub.challenge"),
) -> PlainTextResponse:
    # `verify_challenge` PRIMEIRO: é a única chamada que lê configuração, e o `or`
    # curto-circuita. Na ordem inversa, um GET sem challenge num ambiente sem
    # WHATSAPP_VERIFY_TOKEN saía 403 ("a Meta mandou errado") escondendo um deploy
    # quebrado — o mesmo disfarce que a ordem dentro de `verify_signature` evita.
    #
    # E o teste é `not hub_challenge`, não `is None`: `hub.challenge=` (vazio) passava
    # e devolvia 200 com corpo vazio. A Meta compara o corpo com o challenge que
    # mandou, então isso é handshake FALHO reportado como sucesso — mais difícil de
    # depurar que um 403, e a rota inteira existe pra ecoar o challenge cru.
    if not verify_challenge(hub_mode, hub_verify_token) or not hub_challenge:
        # Sem o token e sem o challenge no log: um deles é segredo, o outro é eco.
        logger.warning("webhook whatsapp: verificação recusada (hub.mode=%r)", hub_mode)
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="verificação recusada")

    # O challenge só é devolvido DEPOIS de o token conferir, então isto não é um
    # refletor gratuito; e o text/plain mantém o eco inerte.
    return PlainTextResponse(hub_challenge)


@app.post("/webhook/whatsapp")
# Duas anotações, e a de cima existe pelo EFEITO COLATERAL, não pelo número: pôr o
# nome da rota em `limiter._route_limits` é o único mecanismo do slowapi 0.1.10 que a
# tira do `default_limits` por IP. É o espelho exato da landmine do /ask — mesma linha
# da lib, sentido oposto. O porquê inteiro (com nomes de função e linhas) está no
# docstring de `whatsapp_limit`, em app/limits.py.
#
# `per_method=True` NÃO é enfeite. O slowapi chaveia o balde pelo PATH
# (`key_style="url"`, então `_endpoint_key = request["path"]`) e a chave do `limits`
# inclui o valor do limite — então o balde do teto por IP no GET e o balde deste POST
# ficam byte a byte iguais sempre que os dois valores coincidem (o fixture do teste
# põe os dois em 100/minute, e "apertar o webhook até o teto" é movimento natural de
# operador). Reproduzido com os dois em 5/minute: 5 handshakes com token errado, de
# qualquer anônimo, faziam o POST assinado seguinte da Meta sair 429 — exatamente o
# desfecho que a âncora existe pra impedir. Com per_method o escopo vira
# "/webhook/whatsapp:POST" (`__evaluate_limits`) e os dois deixam de se tocar.
@limiter.limit(WEBHOOK_ANCORA, per_method=True)
@limiter.limit(whatsapp_limit, per_method=True)
async def whatsapp_webhook(request: Request, background: BackgroundTasks) -> Response:
    """Recebe um evento da Meta, confere a assinatura, GUARDA, e agenda a resposta.

    **O 200 daqui significa "aceito com durabilidade".** Na W1 significava só "li o que
    deu": o evento era logado e descartado. Agora ele é a promessa que autoriza a varredura
    a processar fora do request — a Meta considera a entrega concluída e nunca mais a
    reenvia.

    **A resposta NÃO sai daqui.** Depois do commit, a rota agenda `varrer_em_background`
    (`app/inbox.py`), que roda depois do 200, abre a própria sessão, chama o grafo e envia
    pela Cloud API. O 200 é independente do que a varredura fizer, e é assim que tem que
    ser: a promessa feita à Meta é durabilidade, não entrega — a entrega é retomável, e o
    `scripts/process_pending.py` é o backstop de quem morrer no meio.

    **Segue 200 em quase todo caminho pós-assinatura, e isso continua sendo escolha.** A
    Meta reenvia em não-2xx e desabilita a inscrição depois de falhas repetidas, e nada
    disso é coisa que retentativa conserte: payload ilegível, tipo sem texto, evento que
    nem é mensagem (`value.statuses`). O canal do operador aqui é o log, como na falha de
    worker do /ask.

    **A exceção é o INSERT, e ela é o oposto de tudo acima: 500 de propósito.** Se não
    conseguimos GUARDAR, um 200 seria mentira — a Meta não reenviaria e a pergunta
    sumiria sem que ninguém do lado do usuário soubesse. É a única falha aqui que uma
    retentativa de fato conserta, então é a única que vale devolver não-2xx. Isso só é
    seguro porque `wamid` é UNIQUE: o lote reentregue reentra pelo ON CONFLICT DO
    NOTHING sem duplicar. É a idempotência que autoriza o retry.

    A sessão é aberta **depois** da assinatura, com `SessionLocal()` em vez de
    `Depends(get_session)`, e a ordem é a decisão. Dependência do FastAPI resolve ANTES
    do corpo do handler, então com `Depends` um request forjado — que deve morrer no 403
    — já teria feito a app ler `DATABASE_URL` e construir a engine; num deploy sem banco
    configurado, todo POST saía **500 mudo**, inclusive os forjados, escondendo o 403 e
    quebrando a regra do módulo de que erro de configuração no webhook loga ERROR antes
    (o custo aqui é composto: a Meta desabilita a inscrição). Nada acontece antes de
    provarmos que o evento é da Meta.
    """
    # O corpo CRU, antes de qualquer parse: a assinatura cobre estes bytes exatos, e
    # um json.loads seguido de re-serialização mudaria o que está sendo verificado.
    # Com TETO: o HMAC só existe depois de ler os bytes, então até aqui quem manda é
    # um anônimo — sem limite, ele faz a app bufferizar o que quiser em memória.
    raw = await _corpo_com_teto(request)
    if raw is None:
        logger.warning("webhook whatsapp: corpo acima de %d bytes, recusado sem ler", MAX_BODY_BYTES)
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="corpo grande demais"
        )

    if not verify_signature(raw, request.headers.get(ASSINATURA_HEADER)):
        # Sem o corpo e sem a assinatura no log: é conteúdo de terceiro não
        # autenticado, e logar payload de quem falhou na porta é como se enche disco.
        logger.warning(
            "webhook whatsapp: assinatura inválida (header presente=%s, corpo=%d bytes)",
            ASSINATURA_HEADER in request.headers,
            len(raw),
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="assinatura inválida")

    # `extract_messages` já é defensivo por construção; este try é a SEGUNDA camada,
    # porque o 200 acima não pode depender de o parser estar perfeito.
    try:
        mensagens = extract_messages(json.loads(raw))
    except Exception:
        logger.exception("webhook whatsapp: payload não interpretado (%d bytes)", len(raw))
        mensagens = []

    for msg in mensagens:
        # Um try por MENSAGEM, não um em volta do laço: a Meta entrega em LOTE, e como
        # respondemos 200 ela nunca reentrega — uma falha na mensagem N engolindo as
        # N+1.. as perderia de vez, e o log ainda diria "payload não interpretado"
        # sobre um payload que foi interpretado quase todo.
        try:
            # O texto NUNCA vai pro log — só o tamanho dele. Telefone mascarado.
            logger.info(
                "webhook whatsapp: mensagem (id=%s, de=%s, tipo=%s, caracteres=%d)",
                id_curto(msg.wamid),
                mascarar_telefone(msg.from_phone),
                msg.tipo,
                len(msg.text or ""),
            )
        except Exception:
            logger.exception("webhook whatsapp: falha ao registrar mensagem (id=%s)", id_curto(msg.wamid))

    # A persistência fica FORA do `try` por mensagem acima, e é esse o ponto. Aquele
    # `try` existe pra que uma falha de OBSERVABILIDADE na mensagem N não engula as
    # N+1.. — ele cobre o log, e só. Durabilidade é a regra oposta: se ela falha, o
    # request inteiro tem que falhar. Um `except` largo em volta do INSERT reproduziria
    # exatamente o 200-mentiroso que esta fatia existe pra matar.
    perguntas = _perguntas(mensagens)

    if perguntas:
        # Import tardio e `SessionLocal` (não `Depends`): resolve a fábrica no momento da
        # CHAMADA, que é o que faz o `monkeypatch.setattr("app.db.SessionLocal", ...)`
        # dos testes pegar — o mesmo seam de `record_call_cost` e dos nós do grafo.
        from app.db import SessionLocal

        try:
            session = SessionLocal()
        except Exception as exc:
            # CONFIGURAÇÃO, não indisponibilidade: `SessionLocal()` só monta a fábrica —
            # o asyncpg conecta preguiçosamente, no `execute`. Então o que cai aqui é
            # `DATABASE_URL` ausente ou impossível de parsear, e banco fora do ar aparece
            # lá embaixo (classificado transitório, 500). O rótulo importa: apontar
            # "indisponível" num erro de config manda o operador olhar o Postgres em vez
            # do env.
            #
            # É incidente NOSSO e no webhook custa a integração inteira — sem esta linha
            # ele só descobre pela inscrição já desabilitada. Mesma regra (e mesmo nível)
            # do `_config` de app/whatsapp.py. Sem `exc_info`: a exceção de configuração
            # pode carregar a URL do banco, com senha.
            logger.error(
                "webhook whatsapp: banco não configurado, a entrega não pode ser guardada "
                "(%d mensagens, ids=%s) — %s",
                len(perguntas),
                ",".join(id_curto(m.wamid) for m in perguntas),
                resumo_do_erro(exc),
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="falha ao registrar a entrega",
            )

        try:
            novos = await _registrar_mensagens(session, perguntas)
            await session.commit()
        except Exception as exc:
            reentregar = _vale_reentregar(exc)
            # LOG PRIMEIRO, limpeza depois. A causa mais provável de uma falha aqui é a
            # conexão morta — que é exatamente o caso em que o `rollback()` também
            # levanta. Com a ordem invertida, a exceção do rollback escapava e levava
            # junto o traceback do erro ORIGINAL e o `detail` genérico: 500 mudo, no
            # incidente que mais precisa de diagnóstico.
            # `logger.error` + stack montado à mão, e NÃO `logger.exception`: este é o
            # único ponto do sistema que segura uma exceção nascida de um INSERT com PII
            # nos parâmetros, e `exc_info` imprimiria o texto da exceção inteiro. Ver
            # `_resumo_do_erro`. O stack (só os frames) fica, porque é o que diz ONDE.
            #
            # Os ids entram porque sem eles o ramo PERMANENTE — que devolve 200 e perde a
            # mensagem — deixaria como registro só um número: "3 mensagens perdidas", sem
            # nada que permita achar quem ficou sem resposta. `id_curto` casa com a linha
            # de INFO da W1 e não carrega telefone.
            logger.error(
                "webhook whatsapp: falha ao GRAVAR a entrega (%d mensagens, ids=%s) — %s [%s]\n%s",
                len(perguntas),
                ",".join(id_curto(m.wamid) for m in perguntas),
                "500, pra que a Meta reentregue"
                if reentregar
                else "200, e estas mensagens estão PERDIDAS: reentregar não conserta",
                resumo_do_erro(exc),
                traceback_da_cadeia(exc),
            )
            try:
                await session.rollback()
            except Exception as rollback_exc:
                logger.warning(
                    "webhook whatsapp: o rollback também falhou — %s",
                    resumo_do_erro(rollback_exc),
                )

            if reentregar:
                # `detail` genérico: a rota é pública e sem auth, e nomear o que quebrou
                # entregaria a um scanner o estado interno do deploy. Mesma regra do 500
                # de configuração em app/whatsapp.py — o operador é servido pelo log
                # acima, que carrega o traceback.
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="falha ao registrar a entrega",
                )
            # Permanente: cai fora do `if` e a rota devolve 200. Ver o comentário de
            # `_vale_reentregar` — insistir aqui desliga o webhook inteiro.
        else:
            # UMA linha por entrega, não uma por mensagem: a identificação por mensagem
            # já saiu no laço acima, e o que esta acrescenta é um bit por LOTE — quantas
            # eram reentrega. É esse número que distingue tráfego normal de uma tempestade
            # de retentativa da Meta, e ele vem do RETURNING (linha de fato inserida), não
            # de uma segunda consulta.
            #
            # Dentro de um `try` pelo mesmo motivo do laço acima, e aqui o motivo é mais
            # forte: estamos DEPOIS do commit, então uma exceção daqui trocaria uma
            # gravação bem-sucedida por um 500 — e a reentrega que esse 500 provoca cairia
            # no mesmo ponto, de novo, pra sempre.
            logger.info(
                "webhook whatsapp: entrega guardada (%d nova(s) de %d)",
                len(novos),
                len(perguntas),
            )

            # W2b: agenda a varredura que responde. TRÊS coisas decidem este lugar exato.
            #
            # (1) **Depois do commit bem-sucedido**, e só aqui: a varredura abre OUTRA
            #     sessão, então não enxergaria o que não foi commitado — é o mesmo fato que
            #     faz os testes do inbox lerem por uma conexão separada. No ramo de exceção
            #     não há o que varrer.
            # (2) **A task não pode capturar `session`**: quando ela roda, o `finally` abaixo
            #     já fechou a sessão e devolveu a conexão ao pool. Por isso ela não recebe
            #     nada — abre a sua.
            # (3) **Incondicional, não `if novos:`**. Uma REENTREGA (zero linhas novas) vira
            #     gatilho gratuito de recuperação: se a varredura anterior morreu no meio, a
            #     órfã `computed` sai agora. Se dependesse de linha nova, recuperar exigiria
            #     o `scripts/process_pending.py`, e ele é o backstop, não o caminho normal.
            #     Varredura sem nada a fazer custa um SELECT.
            #
            # A resposta já foi escrita quando a task roda (`Response.__call__` manda
            # `http.response.start`/`body` e só DEPOIS faz `await self.background()`), então
            # a Meta recebe o 200 na hora; o que fica pendurado é a chamada ASGI. Isso
            # também significa que, sob o `ASGITransport` dos testes, `await client.post(...)`
            # roda a varredura inteira — daí o no-op autouse em `tests/conftest.py`.
            background.add_task(inbox.varrer_em_background)
        finally:
            # O `Depends` fechava a sessão por nós; agora é nosso. Best effort: chegamos
            # aqui com a resposta já decidida, e um erro de fechamento não pode trocá-la.
            try:
                await session.close()
            except Exception:
                logger.warning("webhook whatsapp: falha ao fechar a sessão", exc_info=True)

    # O 200 sai aqui; a varredura agendada acima roda logo depois dele, fora do caminho da
    # resposta. Nada do que ela fizer pode mudar este status — a promessa que a Meta recebe é
    # "aceito com durabilidade", e ela já foi cumprida pelo commit.
    return Response(status_code=status.HTTP_200_OK)
