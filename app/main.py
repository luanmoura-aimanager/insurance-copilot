import json
import logging
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse
from starlette.requests import ClientDisconnect
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DisconnectionError, InterfaceError, OperationalError
from sqlalchemy.exc import TimeoutError as SATimeoutError

from app.agents.context import reset_request_context, set_request_context
from app.agents.graph import NO_ANSWER, graph
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
# Superfície do WhatsApp (Meta Cloud API) — FATIA W2a: RECEBER E GUARDAR.
#
# Estas duas rotas ainda não chamam o grafo e não respondem no WhatsApp — isso é a W2b.
# O que a W2a acrescentou à W1 é DURABILIDADE: a mensagem de texto vira linha em
# `whatsapp_message`, deduplicada por `wamid`, e o 200 passa a significar "aceito com
# durabilidade" em vez de "li o que deu".
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


# Erros de banco que uma REENTREGA tem chance de resolver: conexão morta, banco
# reiniciando, pool estourado, statement timeout. É a mesma divisão que o `run_query`
# faz entre `OperationalError` (infra, sobe) e `ProgrammingError` (a query, vira texto)
# — só que aqui ela decide entre **500 e 200**, e a assimetria é grosseira.
#
# Errar pra 500 num erro PERMANENTE é o desfecho ruim: a Meta reentrega o mesmo lote,
# falha igual, e depois de repetidas falhas **desabilita a inscrição** — trocando a
# perda de UMA mensagem pela perda de TODAS as futuras, mais um recadastro manual no
# console dela. Por isso o default do desconhecido é 200 (barulhento, mas vivo), e só o
# que está nesta lista vira 500. Um `DataError` (texto com NUL, que é JSON válido e o
# Postgres não guarda) ou um `ValueError` do encoder do asyncpg não voltam daqui.
_ERROS_QUE_VALE_REENTREGAR = (
    OperationalError, InterfaceError, DisconnectionError, SATimeoutError,
)


def _vale_reentregar(exc: BaseException) -> bool:
    """Se a Meta mandar este lote de novo, tem chance de dar certo?"""
    return isinstance(exc, _ERROS_QUE_VALE_REENTREGAR)


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
async def whatsapp_webhook(request: Request) -> Response:
    """Recebe um evento da Meta, confere a assinatura e GUARDA o que dá pra responder.

    **O 200 daqui significa "aceito com durabilidade".** Na W1 significava só "li o que
    deu": o evento era logado e descartado. Agora ele é a promessa que autoriza a W2b a
    processar fora do request — a Meta considera a entrega concluída e nunca mais a
    reenvia.

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
        except Exception:
            # Banco não configurado é incidente NOSSO, e no webhook ele custa a
            # integração inteira: sem esta linha o operador só descobre pela inscrição já
            # desabilitada. Mesma regra (e mesmo nível) do `_config` de app/whatsapp.py.
            logger.error(
                "webhook whatsapp: banco indisponível para guardar a entrega (%d mensagens)",
                len(perguntas),
                exc_info=True,
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
            logger.exception(
                "webhook whatsapp: falha ao GRAVAR a entrega (%d mensagens) — %s",
                len(perguntas),
                "500, pra que a Meta reentregue"
                if reentregar
                else "200, e estas mensagens estão PERDIDAS: reentregar não conserta",
            )
            try:
                await session.rollback()
            except Exception:
                logger.warning("webhook whatsapp: o rollback também falhou", exc_info=True)

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
            # `_ERROS_QUE_VALE_REENTREGAR` — insistir aqui desliga o webhook inteiro.
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
            try:
                logger.info(
                    "webhook whatsapp: entrega guardada (%d nova(s) de %d)",
                    len(novos),
                    len(perguntas),
                )
            except Exception:
                logger.exception("webhook whatsapp: falha ao registrar o resumo da entrega")
        finally:
            # O `Depends` fechava a sessão por nós; agora é nosso. Best effort: chegamos
            # aqui com a resposta já decidida, e um erro de fechamento não pode trocá-la.
            try:
                await session.close()
            except Exception:
                logger.warning("webhook whatsapp: falha ao fechar a sessão", exc_info=True)

    # O que a W2b herda daqui: chamar o grafo e responder pela Cloud API. A dedup por
    # `wamid` que esta nota pedia deixou de ser herança — ela existe, é o unique de
    # `whatsapp_message`, e é o que torna a reentrega da Meta inofensiva.
    return Response(status_code=status.HTTP_200_OK)
