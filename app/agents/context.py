"""Contexto do request (quem perguntou, e em qual request) visível de dentro do grafo.

Por que ContextVar e não um campo no State do grafo: quem grava o custo é o NÓ
(supervisor / sql_worker), mas o dado nasce na borda HTTP. Carregar `request_id` e
`client` pelo State obrigaria todo nó novo a arrastar campos que não são do domínio do
grafo — e o State é o contrato entre os nós, não o do request.

ContextVar é local à task do asyncio: cada request roda na sua task e as tasks filhas
(os nós) herdam uma CÓPIA do contexto no momento em que são criadas. Dois /ask
concorrentes não se enxergam, ao contrário do que aconteceria com uma global.

Default None de propósito: fora de um request (extração offline, `python -m
app.agents.graph`) não existe request_id nem cliente — e as colunas são nullable.
"""
from contextvars import ContextVar, Token

request_id_var: ContextVar[str | None] = ContextVar("request_id", default=None)
client_var: ContextVar[str | None] = ContextVar("client", default=None)


def set_request_context(request_id: str | None, client: str | None) -> tuple[Token, Token]:
    """Fixa o contexto e devolve os tokens de reset — passe-os pro reset no `finally`."""
    return request_id_var.set(request_id), client_var.set(client)


def reset_request_context(tokens: tuple[Token, Token]) -> None:
    """Desfaz o `set` restaurando o valor ANTERIOR.

    `reset(token)` em vez de `set(None)`: o token devolve o contexto ao estado exato de
    antes, sem assumir que antes era None.
    """
    request_id_token, client_token = tokens
    request_id_var.reset(request_id_token)
    client_var.reset(client_token)


def get_request_id() -> str | None:
    """Id do request atual (correlaciona todas as chamadas de LLM de um mesmo /ask)."""
    return request_id_var.get()


def get_client_name() -> str | None:
    """Nome do cliente autenticado — a identidade do token, nunca o token."""
    return client_var.get()
