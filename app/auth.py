"""
Autenticação por Bearer token COM IDENTIDADE.

Não é um segredo único global: `API_TOKENS` mapeia nome -> token, então cada cliente
tem seu próprio token. Isso é o que permite (a) revogar um cliente sem derrubar os
outros e (b) o rate limit por cliente em `app/limits.py` — sem identidade, o limite
só poderia ser por IP.

Formato do env: "nome:token,nome2:token2".
"""
import os
import secrets

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

# auto_error=False: queremos controlar o 401 (e o header WWW-Authenticate) na mão,
# em vez do 403 que o HTTPBearer devolve por padrão quando falta o header.
_bearer = HTTPBearer(auto_error=False)

_UNAUTHORIZED = "Token inválido ou ausente."


def _load_tokens() -> dict[str, str]:
    """Parseia API_TOKENS -> {token: nome}.

    O env é lido AQUI DENTRO, nunca no nível do módulo: `import app.main` precisa
    continuar funcionando sem env var nenhuma (import não pode explodir), e ler a
    cada request permite trocar tokens sem rebuild da imagem.
    """
    tokens: dict[str, str] = {}
    for entry in os.getenv("API_TOKENS", "").split(","):
        name, sep, token = entry.partition(":")
        if not sep:
            continue
        name, token = name.strip(), token.strip()
        if name and token:
            tokens[token] = name
    return tokens


def require_client(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """Valida o Bearer e devolve o NOME do cliente (nunca o token)."""
    tokens = _load_tokens()

    # Fail closed na CONFIGURAÇÃO: sem tokens cadastrados ninguém entra, e o erro é
    # 500 (não 401) de propósito — 401 aqui esconderia um deploy quebrado atrás de
    # uma resposta que parece "credencial errada do cliente".
    if not tokens:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_TOKENS não configurado",
        )

    if creds is None or creds.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )

    # compare_digest (tempo constante) em vez de `==`: comparação normal sai no
    # primeiro byte diferente e vaza o prefixo correto por timing. E iteramos TODOS
    # os tokens SEM break — um break vazaria a posição do token na lista.
    #
    # Comparamos em BYTES: compare_digest sobre str explode com TypeError se houver
    # caractere não-ASCII, e a credencial vem do header (o cliente escolhe o que
    # mandar). Em str, um token com acento viraria 500 em vez de 401.
    presented = creds.credentials.encode("utf-8")
    matched: str | None = None
    for token, name in tokens.items():
        if secrets.compare_digest(presented, token.encode("utf-8")):
            matched = name

    if matched is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Gravado aqui, e não no corpo da rota, por uma questão de ORDEM: o FastAPI
    # resolve as dependencies ANTES de chamar o endpoint, e o wrapper do slowapi
    # (que lê client_name pra montar a chave do limite) roda junto com o endpoint.
    # Se só o corpo gravasse, o limite por cliente nunca enxergaria o nome.
    request.state.client_name = matched
    return matched
