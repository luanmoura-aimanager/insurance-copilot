from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy import text

from app.agents.context import reset_request_context, set_request_context
from app.agents.graph import graph
from app.auth import require_client
from app.db import get_session
from app.limits import ask_client_limit, client_key, limiter

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

# Resposta quando nenhum worker rodou (supervisor foi direto pro END, ou o circuit
# breaker cortou antes de qualquer resultado): não inventamos resposta.
NO_ANSWER = "Não consegui responder essa pergunta com os dados disponíveis."


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str
    iterations: int


def _final_answer(state: dict) -> str:
    """Última mensagem do sql_worker no histórico.

    Varre DE TRÁS PRA FRENTE procurando o worker: messages[-1] seria o reasoning do
    supervisor (ele sempre roda por último, pra decidir o END), não a resposta.
    """
    for m in reversed(state["messages"]):
        if isinstance(m, AIMessage) and m.name == "sql_worker":
            return m.content
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
