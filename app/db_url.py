"""Normalização de URLs de Postgres para os drivers usados no projeto.

Um único ponto que resolve as variações que provedores gerenciados (ex.: Railway)
entregam em ``DATABASE_URL`` — ``postgres://`` vs ``postgresql://``, driver async
vs sync, e o parâmetro ``sslmode`` (que é do libpq/psycopg2 e o asyncpg rejeita).
"""

from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

# Parâmetros de query que só o libpq (psycopg2) entende; o asyncpg os rejeita.
_LIBPQ_ONLY_PARAMS = {"sslmode", "sslrootcert", "sslcert", "sslkey", "target_session_attrs"}


def normalize_url(url: str, driver: str) -> str:
    """Normaliza qualquer URL de Postgres para ``postgresql+<driver>://...``.

    ``driver`` é ``"asyncpg"`` (app) ou ``"psycopg2"`` (migrations Alembic).
    Só o *scheme* é reescrito — credenciais e nome do banco ficam intactos.
    Para asyncpg, remove parâmetros de query exclusivos do libpq (ex.: ``sslmode``),
    que fariam ``create_async_engine`` quebrar na primeira conexão.
    """
    scheme, netloc, path, query, fragment = urlsplit(url)

    # postgres:// -> postgresql:// ; qualquer +driver anterior é descartado.
    base = scheme.split("+", 1)[0]
    if base == "postgres":
        base = "postgresql"
    scheme = f"{base}+{driver}"

    if driver == "asyncpg" and query:
        query = urlencode(
            [(k, v) for k, v in parse_qsl(query, keep_blank_values=True)
             if k not in _LIBPQ_ONLY_PARAMS]
        )

    return urlunsplit((scheme, netloc, path, query, fragment))
