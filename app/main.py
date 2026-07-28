from fastapi import Depends, FastAPI, Request
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from sqlalchemy import text

from app.agents.graph import graph
from app.auth import require_client
from app.db import get_session
from app.limits import ask_client_limit, ask_ip_limit, client_key, limiter

app = FastAPI()

# O slowapi lê o limiter de app.state e precisa do handler pra transformar o
# RateLimitExceeded (que é uma HTTPException 429) em resposta.
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

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
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/db")
async def health_db(session=Depends(get_session)):
    await session.execute(text("SELECT 1"))
    return {"db": "ok"}


@app.post("/ask", response_model=AskResponse)
# Os dois limites são registrados sob o mesmo nome de função, e o slowapi marca
# request.state._rate_limiting_complete depois da primeira checagem — então os dois
# são avaliados uma única vez, cada um com a sua chave (cliente e IP).
@limiter.limit(ask_client_limit, key_func=client_key)
@limiter.limit(ask_ip_limit, key_func=get_remote_address)
async def ask(
    request: Request,                                  # exigido pelo slowapi
    req: AskRequest,
    client: str = Depends(require_client),
) -> AskResponse:
    # require_client já gravou isso em request.state (antes do wrapper do slowapi);
    # repetimos aqui só pra deixar o vínculo explícito na leitura da rota.
    request.state.client_name = client
    state = await graph.ainvoke({
        "iterations": 0,
        "next": "",
        "messages": [HumanMessage(content=req.question)],
    })
    return AskResponse(answer=_final_answer(state), iterations=state["iterations"])
