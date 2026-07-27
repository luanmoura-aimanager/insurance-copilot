from fastapi import FastAPI, Depends
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field
from sqlalchemy import text

from app.agents.graph import graph
from app.db import get_session

app = FastAPI()

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/db")
async def health_db(session=Depends(get_session)):
    await session.execute(text("SELECT 1"))
    return {"db": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    state = await graph.ainvoke({
        "iterations": 0,
        "next": "",
        "messages": [HumanMessage(content=req.question)],
    })
    return AskResponse(answer=_final_answer(state), iterations=state["iterations"])
