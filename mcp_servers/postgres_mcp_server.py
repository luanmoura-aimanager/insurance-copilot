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

# Alvo de FROM/JOIN. Não é um parser de SQL, e não precisa ser: subquery e CTE também
# usam FROM, então toda leitura de tabela passa por aqui. `FROM (SELECT ...)` não casa
# (o próximo char é `(`, não identificador), que é o certo — o que importa está dentro.
_FROM_JOIN_RE = re.compile(
    r"\b(?:from|join)\s+(?:only\s+)?([a-zA-Z_][\w$]*(?:\.[a-zA-Z_][\w$]*)?)",
    re.IGNORECASE,
)
# Nomes de CTE (`WITH x AS (`), que são referenciáveis mas não são tabelas.
_CTE_RE = re.compile(
    r"\b([a-zA-Z_][\w$]*)\s+as\s*(?:not\s+materialized\s+|materialized\s+)?\(",
    re.IGNORECASE,
)


# `OperationalError` cuja CAUSA é a query, não a infraestrutura — a exceção à regra do
# `except` do run_query, e ela existe porque a hierarquia do psycopg não separa as duas
# coisas. `statement_timeout` (57014) e a classe 54 inteira (`program_limit_exceeded`:
# query complexa demais, colunas/argumentos demais) são resultado direto do SQL que o
# modelo escreveu — um cartesian join num Postgres gerenciado, que costuma vir com
# `statement_timeout` configurado, chega aqui. Tratá-las como incidente pagaria o operador
# pra investigar infra por causa de uma query ruim, e tiraria do worker a chance de
# reportar o que aconteceu. Ficam de fora de propósito a classe 53
# (`insufficient_resources`: disk full, out of memory) e o resto da 57 (`admin_shutdown`,
# `crash_shutdown`): mesmo quando uma query pesada os provoca, o operador quer saber.
_ERROS_DA_QUERY = (
    psycopg.errors.QueryCanceled,          # 57014 — statement_timeout / cancel
    psycopg.errors.ProgramLimitExceeded,   # 54000
    psycopg.errors.StatementTooComplex,    # 54001
    psycopg.errors.TooManyColumns,         # 54011
    psycopg.errors.TooManyArguments,       # 54023
)


def _conninfo() -> str:
    """String de conexão libpq. Prefere a role read-only (`DATABASE_URL_RO`) quando
    presente — é a garantia real de segurança: mesmo se o filtro de texto do run_query
    for burlado, a role não tem permissão de escrever. Cai pro DATABASE_URL admin só se
    a RO não estiver configurada.

    Esse fallback é silencioso, e é por isso que a allowlist de tabela do run_query não
    é redundante com o `REVOKE SELECT ON clause_chunk`: num deploy sem DATABASE_URL_RO
    o worker conecta como admin e o REVOKE simplesmente não participa."""
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


def _tabelas_fora_da_allowlist(body: str) -> set[str]:
    """Nomes referenciados em FROM/JOIN que não são tabela de domínio nem CTE local.

    A allowlist é o par da omissão no `get_schema()`: sem ela, "o worker SQL não
    enxerga `clause_chunk`" era só uma tabela não *anunciada* — um
    `SELECT * FROM clause_chunk` escrito pelo LLM passava pelo filtro textual (que só
    olha o primeiro token) e o LIMIT 100 automático despejava 100 vetores de 1024
    floats no contexto da chamada seguinte, num projeto que cobra por chamada.

    O `REVOKE SELECT` da migration a4c91e5d7f28 cobre o mesmo buraco pelo lado do
    banco, mas só morde quando `DATABASE_URL_RO` está configurada — sem ela o
    `_conninfo()` cai pro DATABASE_URL admin e a role read-only não entra na história.
    Esta checagem vale nos dois casos, e é por isso que ela existe além do REVOKE.
    """
    ctes = {nome.lower() for nome in _CTE_RE.findall(body)}
    referenciadas = set()
    for nome in _FROM_JOIN_RE.findall(body):
        nome = nome.lower()
        # `public.coverage` é a mesma coverage; qualquer outro schema é suspeito e
        # cai na rejeição junto com os nomes desconhecidos.
        if nome.startswith("public."):
            nome = nome.split(".", 1)[1]
        referenciadas.add(nome)
    return referenciadas - set(TABLES) - ctes


def run_query(sql: str) -> str:
    """
    Executa uma query read-only (SELECT) e retorna até 100 linhas.
    Rejeita statements empilhados, qualquer coisa que não comece com SELECT, e
    qualquer leitura de tabela fora das 5 do domínio.

    **Erro de QUERY volta como texto; erro de CONEXÃO sobe como exceção.** Ver o
    comentário no `except`: o primeiro é resultado (o LLM escreveu SQL inválido e precisa
    ser informado), o segundo é incidente.
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

    # 3. Só as 5 tabelas do domínio. Sem isto o filtro acima aprova qualquer SELECT,
    #    inclusive em clause_chunk (vetores) e cost_event.
    fora = _tabelas_fora_da_allowlist(body)
    if fora:
        return (
            f"Error: table(s) not allowed: {', '.join(sorted(fora))}. "
            f"Only these tables can be queried: {', '.join(TABLES)}."
        )

    # 4. Anexa LIMIT 100 se a query ainda não tiver um LIMIT.
    if not re.search(r"\blimit\b", body, re.IGNORECASE):
        body = f"{body} LIMIT 100"

    conn = None
    try:
        conn = _connect()
        cursor = conn.cursor()
        cursor.execute(body)
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
    except psycopg.Error as exc:
        # **Falha de INFRAESTRUTURA sobe; erro de QUERY continua voltando como texto.**
        #
        # Antes tudo voltava como texto, e isso era o último caminho por onde detalhe de
        # infra chegava ao usuário: a mensagem de uma `OperationalError` de conexão é
        # `connection to server at "host" (ip), port 5432 failed: password authentication
        # failed for user "insurance_ro"` — host, porta, usuário e banco. Como o texto ia
        # como RESULTADO do worker, entrava no prompt do synthesizer e, no fallback dele
        # (`frase or resultado`), virava a resposta do /ask ao pé da letra.
        #
        # Deixar subir é mais do que não vazar: é a MESMA falha que o `get_schema()` já
        # propagava (ele não tem `except` nenhum), e as duas chamadas de banco deste worker
        # não podem tratar o mesmo `OperationalError` de dois jeitos. Quem recebe é o `try`
        # que envolve o `sql_worker` inteiro, que a transforma em `FALHA_INTERNA` + um
        # `logger.exception` com `request_id`/`client`. Também economiza a chamada de LLM
        # que o synthesizer gastaria pra parafrasear um erro de conexão.
        #
        # O texto continua sendo o certo pro erro de QUERY, e é o ponto de "run_query nunca
        # levanta" que de fato importava: o LLM escreveu SQL inválido, e o worker precisa
        # poder contar isso ao synthesizer em vez de estourar. Tudo que o modelo causa por
        # ESCREVER errado é `ProgrammingError` (`SyntaxError`, `UndefinedColumn`,
        # `UndefinedTable`, `InsufficientPrivilege`) — mas **`OperationalError` não é
        # sinônimo de infraestrutura**, e é por isso que `_ERROS_DA_QUERY` existe: uma query
        # que estoura o `statement_timeout` ou os limites de complexidade também cai lá, e
        # tratá-la como incidente pagaria o operador pra investigar um runaway query.
        # `sqlstate` é `None` numa falha de conexão (não há resposta do servidor), o que é
        # justamente o caso que tem que subir.
        #
        # No servidor stdio isso vira erro de tool em vez de texto de resultado, o que é
        # honesto: "o banco está fora" não é o resultado de uma query.
        if isinstance(exc, psycopg.OperationalError) and not isinstance(exc, _ERROS_DA_QUERY):
            raise
        return f"Error executing query: {exc}"
    finally:
        if conn is not None and not conn.closed:
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
