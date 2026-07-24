"""MCP server (stdio) sobre o schema Postgres do insurance-copilot.

Porta do sqlite-mcp-server do sql-agent, retargeteado de SQLite para o Postgres
das 5 tabelas do domínio. Expõe duas tools: get_schema() e run_query(sql).

Processo curto de stdio → psycopg **sync** (não asyncpg), uma conexão nova por
chamada, fechada ao fim (sem pool).
"""

import os
import re

from dotenv import load_dotenv
import psycopg
from mcp.server.fastmcp import FastMCP

from app.db_url import normalize_url

load_dotenv()

mcp = FastMCP("postgres-mcp-server")

# As 5 tabelas do domínio — a fronteira do que o worker SQL pode enxergar.
TABLES = ("policy_document", "coverage", "peril", "coverage_peril", "exclusion")


def _conninfo() -> str:
    """String de conexão libpq. Prefere a role read-only (`DATABASE_URL_RO`) quando
    presente — é a garantia real de segurança: mesmo se o filtro de texto do run_query
    for burlado, a role não tem permissão de escrever. Cai pro DATABASE_URL admin só se
    a RO não estiver configurada."""
    raw = os.environ.get("DATABASE_URL_RO") or os.environ["DATABASE_URL"]
    # normalize_url devolve `postgresql+psycopg://...`; o libpq (psycopg.connect)
    # só entende o scheme puro `postgresql://`, então tiramos o `+psycopg`.
    normalized = normalize_url(raw, "psycopg")
    return normalized.replace("postgresql+psycopg://", "postgresql://", 1)


def _connect():
    """Abre uma conexão nova (read-only). Cada tool fecha a sua."""
    return psycopg.connect(_conninfo())


# --- Core (funções puras): chamáveis por import direto (o worker SQL as usa no mesmo
# processo) E registradas como tools FastMCP mais abaixo. Uma lógica só, dois acessos. ---
def get_schema() -> str:
    """
    Retorna o schema das 5 tabelas do domínio: cada tabela e suas colunas (com tipo).
    """
    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_name = ANY(%s)
            ORDER BY table_name, ordinal_position
            """,
            (list(TABLES),),
        )
        rows = cursor.fetchall()
    finally:
        conn.close()

    # Agrupa por tabela preservando a ordem de TABLES.
    by_table: dict[str, list[tuple[str, str]]] = {t: [] for t in TABLES}
    for table_name, column_name, data_type in rows:
        by_table.setdefault(table_name, []).append((column_name, data_type))

    schema_str = ""
    for table in TABLES:
        schema_str += f"Table: {table}\n"
        cols = by_table.get(table, [])
        if not cols:
            schema_str += "  (tabela não encontrada no banco)\n"
        for column_name, data_type in cols:
            schema_str += f"  - {column_name} ({data_type})\n"
        schema_str += "\n"

    return schema_str


def run_query(sql: str) -> str:
    """
    Executa uma query read-only (SELECT) e retorna até 100 linhas.
    Rejeita statements empilhados e qualquer coisa que não comece com SELECT.
    """
    stripped = sql.strip()

    # 1. Rejeita statements empilhados: qualquer `;` que não seja o trailing.
    body = stripped
    if body.endswith(";"):
        body = body[:-1].rstrip()
    if ";" in body:
        return (
            "Error: multiple SQL statements are not allowed "
            "(only a single SELECT per call)."
        )

    # 2. Só SELECT.
    if not body.lower().startswith("select"):
        return "Error: only SELECT queries are allowed."

    # 3. Anexa LIMIT 100 se a query ainda não tiver um LIMIT.
    if not re.search(r"\blimit\b", body, re.IGNORECASE):
        body = f"{body} LIMIT 100"

    conn = _connect()
    try:
        cursor = conn.cursor()
        cursor.execute(body)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except psycopg.Error as exc:
        conn.close()
        return f"Error executing query: {exc}"
    finally:
        if not conn.closed:
            conn.close()

    if not rows:
        return f"Columns: {', '.join(columns)}\n(no rows returned)"

    total = len(rows)
    truncated = rows[:100]
    output = f"Columns: {', '.join(columns)}\nRows:\n"
    for row in truncated:
        output += " | ".join(str(value) for value in row) + "\n"
    # Um LIMIT do próprio caller (> 100) escapa do LIMIT 100 que anexamos, então
    # o fetch pode passar de 100 — reporta a truncagem em vez de escondê-la.
    if total > 100:
        output += f"\n(showing first 100 of {total} rows)"
    else:
        output += f"\n({total} rows)"
    return output


# --- Registra as funções core como tools FastMCP (a superfície do protocolo). Registrar
# por chamada (não por @decorator) preserva os nomes de módulo apontando pras funções
# puras, então `from mcp_servers.postgres_mcp_server import get_schema, run_query` devolve
# as funções chamáveis diretas — não os wrappers Tool. ---
mcp.tool()(get_schema)
mcp.tool()(run_query)


if __name__ == "__main__":
    mcp.run(transport="stdio")
