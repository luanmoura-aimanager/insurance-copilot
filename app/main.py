import json
import logging
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text

from app.agents.context import reset_request_context, set_request_context
from app.agents.graph import NO_ANSWER, graph
from app.auth import require_client
from app.db import get_session
from app.limits import WEBHOOK_ANCORA, ask_client_limit, client_key, limiter, whatsapp_limit
from app.whatsapp import (
    ASSINATURA_HEADER,
    MAX_BODY_BYTES,
    extract_messages,
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


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    iterations: int


def _final_answer(state: dict) -> str:
    """A frase do synthesizer, que é sempre o último nó do grafo.

    Não há mais o que varrer: todo caminho de saída passa pelo synthesizer (inclusive
    o que não roda worker nenhum), então a resposta é literalmente a última mensagem.
    O NO_ANSWER aqui é só cinto: se o histórico vier com outra coisa no fim, não
    devolvemos o raciocínio interno de um agente como se fosse resposta.

    **Falha de worker também sai 200, com a frase `FALHA_INTERNA`.** A superfície deste
    sistema é conversa (hoje `/ask`, amanhã WhatsApp): um 5xx entrega ao usuário uma tela
    de erro do framework ou um balão vazio — ele fica sem resposta E sem saber se vale
    tentar de novo. Uma frase que diz "falhei do meu lado, tente em instantes" é a
    informação que ele pode usar. O que o STATUS resolveria — alarme, dashboard,
    investigação — é necessidade do operador, e o operador é servido pelo LOG: cada uma
    dessas respostas tem um `logger.exception` com traceback, `request_id` e `client`
    (ver `_falha_de_worker` em `app/agents/graph.py`), que é estritamente mais do que um
    500 opaco carregaria. Trocar o diagnóstico do operador pela resposta do usuário seria
    piorar os dois lados.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.name == "final":
        return last.content
    return NO_ANSWER


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

    # Contexto que os nós do grafo leem pra atribuir o custo: um id novo por request
    # (correlaciona as N chamadas de LLM de um mesmo /ask) e QUEM pediu. O reset no
    # finally é obrigatório — sem ele o valor sobreviveria ao request nesta task e
    # vazaria pro próximo que a reaproveitasse.
    ctx = set_request_context(str(uuid4()), client)
    try:
        state = await graph.ainvoke({
            "iterations": 0,
            "next": "",
            "messages": [HumanMessage(content=req.question)],
        })
    finally:
        reset_request_context(ctx)
    return AskResponse(answer=_final_answer(state), iterations=state["iterations"])


# ---------------------------------------------------------------------------
# Superfície do WhatsApp (Meta Cloud API) — FATIA W1: só RECEBER.
#
# Estas duas rotas não chamam o grafo e não respondem no WhatsApp; elas provam que um
# evento chegou, que veio mesmo da Meta e que não foi adulterado. Responder é a W2.
#
# É o primeiro endpoint público do projeto: a Meta não manda `Authorization`, então
# quem faz o papel do Bearer aqui é a assinatura HMAC do corpo (`app/whatsapp.py`).
# ---------------------------------------------------------------------------


async def _corpo_com_teto(request: Request) -> bytes | None:
    """Lê o corpo até `MAX_BODY_BYTES`; devolve None se passar disso.

    `await request.body()` bufferiza o stream inteiro, e aqui isso é feito ANTES de
    qualquer autenticação (o HMAC precisa dos bytes pra existir). O `Content-Length`
    sozinho não basta — em `Transfer-Encoding: chunked` ele nem vem —, então o teto é
    conferido também enquanto se lê.
    """
    declarado = request.headers.get("content-length")
    if declarado is not None and declarado.isdigit() and int(declarado) > MAX_BODY_BYTES:
        return None

    partes: list[bytes] = []
    tamanho = 0
    async for pedaco in request.stream():
        tamanho += len(pedaco)
        if tamanho > MAX_BODY_BYTES:
            return None
        partes.append(pedaco)
    return b"".join(partes)


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
async def whatsapp_webhook(request: Request) -> Response:
    """Recebe um evento da Meta, confere a assinatura e registra o que dá pra ler.

    **Devolve 200 em todo caminho pós-assinatura, e isso é escolha.** A Meta reenvia
    em não-2xx e desabilita a inscrição depois de falhas repetidas — e um payload que
    não sabemos ler não é coisa que retentativa conserte, então responder não-2xx não
    compra nada e arrisca desligar o webhook. O canal do operador aqui é o log, como
    na falha de worker do /ask.
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
                "webhook whatsapp: mensagem (wamid=%s, de=%s, tipo=%s, caracteres=%d)",
                msg.wamid,
                mascarar_telefone(msg.from_phone),
                msg.tipo,
                len(msg.text or ""),
            )
        except Exception:
            logger.exception("webhook whatsapp: falha ao registrar mensagem (wamid=%s)", msg.wamid)

    # W2 herda daqui: a Meta pode REENTREGAR o mesmo `wamid` (é o que ela faz quando
    # não recebe 200 a tempo). Hoje isso é inofensivo porque só se loga; quando o
    # webhook chamar o grafo, cada reentrega vira uma rodada paga a mais — a
    # deduplicação por `wamid` é obrigatória lá, não aqui.
    return Response(status_code=status.HTTP_200_OK)
