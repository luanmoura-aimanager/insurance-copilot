"""
Rate limit em duas camadas, com chaves E pontos de aplicação diferentes.

- Por **IP**: limite GLOBAL (`default_limits`), aplicado pelo `SlowAPIMiddleware`.
  Middleware roda ANTES das dependencies do FastAPI, então também conta requests que
  morrem na autenticação — sem isso dá pra martelar /ask com token inválido de graça,
  porque o 401 sai da dependency antes do decorator do endpoint ser alcançado.
- Por **CLIENTE**: cota por rota (decorator no /ask). Tem que continuar no decorator
  porque depende de `request.state.client_name`, que só existe DEPOIS da auth.

O key_func global é o IP justamente porque o `default_limits` herda ele (ver
`Limiter.__init__`); a chave por cliente é passada explicitamente no decorator.
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
    """Chave da cota por cliente.

    `client_name` é gravado por `require_client`, que roda antes do decorator; o
    fallback pro IP é só um cinto — chave vazia faria o slowapi pular o limite
    ("Skipping limit: Empty value found in parameters"), o que falharia aberto.
    """
    return getattr(request.state, "client_name", None) or get_remote_address(request)


def ask_client_limit() -> str:
    """Callable (não string): o slowapi reavalia a cada request, então o teste
    consegue apertar o limite via env DEPOIS do import do app.main.

    ATENÇÃO — isto é load-bearing além do teste: `_should_exempt` do middleware pula
    a rota inteira se ela estiver em `_route_limits`, e só limites ESTÁTICOS vão
    parar lá (os callables vão pra `_dynamic_route_limits`). Trocar este callable por
    uma string faria o middleware ignorar /ask e o teto por IP pararia de cobrir os
    401 — em silêncio. `test_teto_por_ip_cobre_request_sem_auth` é o que segura isso.
    """
    return _limit_from_env("ASK_RATE_LIMIT_CLIENT", DEFAULT_CLIENT_LIMIT)


def ask_ip_limit() -> str:
    return _limit_from_env("ASK_RATE_LIMIT_IP", DEFAULT_IP_LIMIT)


# key_func global = IP porque é a chave que o `default_limits` herda. O limite por IP
# vira `default_limits`, avaliado só pelo middleware; o decorator do /ask usa
# override_defaults=True (padrão do .limit()), então NÃO reavalia o default.
# Resultado: cada limite é contado uma vez por request, nunca em dobro — ver o
# comentário no /ask em app/main.py.
limiter = Limiter(key_func=get_remote_address, default_limits=[ask_ip_limit])
