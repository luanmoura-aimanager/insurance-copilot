"""
Rate limit do /ask — dois limites empilhados, com chaves diferentes.

Por que dois: o limite por CLIENTE é o teto de consumo contratado (cada hop do
supervisor é uma chamada paga à Anthropic); o limite por IP é a proteção de burst,
que continua valendo mesmo se um token vazar e for usado de vários lugares.
"""
import logging
import os

from limits import parse_many
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

logger = logging.getLogger(__name__)

# Defaults conservadores; sobrescritos por env (o teste aperta os dois pra caber
# num teste rápido, sem sleep).
DEFAULT_CLIENT_LIMIT = "30/hour"
DEFAULT_IP_LIMIT = "10/minute"


def _limit_from_env(var: str, default: str) -> str:
    """Lê um limite do env, caindo pro default se a string não for parseável.

    Sem essa checagem o slowapi FALHA ABERTO: ele engole o ValueError do parse,
    loga, e segue sem nenhum limite naquela chave. Ou seja, um typo em
    ASK_RATE_LIMIT_CLIENT no Railway desligaria o teto de gasto em silêncio —
    exatamente o oposto do fail closed da autenticação.
    """
    raw = os.getenv(var, "").strip()
    if not raw:
        return default
    try:
        parse_many(raw)
    except ValueError:
        logger.warning("%s inválido (%r); usando o default %s", var, raw, default)
        return default
    return raw


def client_key(request: Request) -> str:
    """Chave do limite por cliente.

    `client_name` é gravado por `require_client`. O fallback pro IP existe porque o
    slowapi calcula a chave mesmo em rotas/caminhos onde a dependency não rodou —
    melhor cair num limite por IP do que numa chave vazia compartilhada por todos.
    """
    return getattr(request.state, "client_name", None) or get_remote_address(request)


def ask_client_limit() -> str:
    """Callable (não string): o slowapi reavalia a cada request, então o teste
    consegue apertar o limite via env DEPOIS do import do app.main."""
    return _limit_from_env("ASK_RATE_LIMIT_CLIENT", DEFAULT_CLIENT_LIMIT)


def ask_ip_limit() -> str:
    return _limit_from_env("ASK_RATE_LIMIT_IP", DEFAULT_IP_LIMIT)


limiter = Limiter(key_func=client_key)
