"""A allowlist de tabela do `run_query` — a fronteira do worker SQL, pelo lado do app.

Não toca no banco de propósito: a rejeição acontece *antes* do `_connect()`, então
estes testes exercitam exatamente o caminho que importa (a query nunca chega ao
Postgres) e rodam sem container.

O par desta camada é o `REVOKE SELECT ON clause_chunk FROM insurance_ro` (migration
a4c91e5d7f28), testado em `test_vector.py`. As duas existem porque nenhuma cobre o
caso da outra: o REVOKE não participa quando `DATABASE_URL_RO` não está configurada e
o `_conninfo()` cai pro DATABASE_URL admin; a allowlist não impediria uma role com
permissão de escrita de fazer estrago se o filtro de SELECT fosse burlado.
"""
import pytest

from mcp_servers.postgres_mcp_server import TABLES, run_query


@pytest.mark.parametrize(
    "sql,proibida",
    [
        ("SELECT * FROM clause_chunk", "clause_chunk"),
        ("select id, embedding from clause_chunk order by id", "clause_chunk"),
        # cost_event é governança, não domínio — também está fora.
        ("SELECT sum(cost_usd) FROM cost_event", "cost_event"),
        # Tabela de domínio no SELECT, proibida escondida numa subquery.
        ("SELECT (SELECT count(*) FROM clause_chunk) FROM peril", "clause_chunk"),
        # JOIN, não FROM.
        (
            "SELECT p.name FROM peril p JOIN clause_chunk c ON c.id = p.id",
            "clause_chunk",
        ),
        # O catálogo do Postgres é uma rota de fuga tão boa quanto qualquer outra.
        ("SELECT tablename FROM pg_catalog.pg_tables", "pg_catalog.pg_tables"),
    ],
)
def test_rejeita_tabela_fora_do_dominio(sql, proibida):
    """O motivo de existir: um `SELECT * FROM clause_chunk` escrito pelo LLM passava
    pelo filtro textual (que só olha o primeiro token) e o LIMIT 100 automático
    despejava 100 vetores de 1024 floats — ~1 MB de texto — no contexto da chamada
    seguinte, num projeto que mede custo por chamada."""
    resultado = run_query(sql)
    assert resultado.startswith("Error: table(s) not allowed")
    assert proibida in resultado


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT count(*) FROM peril",
        "SELECT * FROM public.coverage",
        "SELECT * FROM ONLY coverage",
        # Alias de tabela e join entre tabelas do domínio.
        "SELECT c.coverage_name FROM coverage c "
        "JOIN coverage_peril cp ON cp.coverage_id = c.id "
        "JOIN peril p ON p.id = cp.peril_id",
        # CTE: `x` é referenciável mas não é tabela — não pode ser confundido com uma
        # tabela proibida, senão a allowlist quebraria queries legítimas.
        "WITH x AS (SELECT id FROM coverage) SELECT count(*) FROM x",
        # Subquery no FROM: o alias `sub` também não é tabela.
        "SELECT count(*) FROM (SELECT id FROM exclusion) AS sub",
    ],
)
def test_aceita_as_tabelas_do_dominio(sql, monkeypatch):
    """A allowlist não pode ser um muro largo demais.

    Sem este lado, `return "Error"` incondicional passaria nos testes de rejeição e
    derrubaria o worker inteiro em silêncio. Não há banco aqui, então o que se afirma é
    que a query **passou da allowlist** — a execução falha depois, no connect.
    """
    monkeypatch.delenv("DATABASE_URL_RO", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/none")
    resultado = run_query(sql)
    assert not resultado.startswith("Error: table(s) not allowed")


def test_filtros_anteriores_continuam_valendo():
    """A allowlist entrou no meio da cadeia; os dois filtros originais seguem antes."""
    assert run_query("DROP TABLE peril").startswith("Error: only SELECT")
    assert run_query("SELECT 1; DROP TABLE peril").startswith(
        "Error: multiple SQL statements"
    )


def test_a_mensagem_de_erro_nao_vaza_o_schema_proibido():
    """Lista o que é permitido, não o que existe.

    O texto volta pro LLM e vira contexto da próxima chamada: enumerar as tabelas
    bloqueadas seria ensinar o caminho e pagar tokens pra isso.
    """
    resultado = run_query("SELECT * FROM clause_chunk")
    for tabela in TABLES:
        assert tabela in resultado
    assert "embedding" not in resultado
