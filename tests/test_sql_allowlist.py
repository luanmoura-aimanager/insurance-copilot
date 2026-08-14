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
import psycopg
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
        # CTE dentro de subquery: `y` é referenciável mas não é tabela — não pode ser
        # confundido com uma tabela proibida, senão a allowlist quebraria queries legítimas.
        #
        # É esta forma, e não `WITH x AS (...) SELECT ...`, que exercita o `_CTE_RE`: a
        # versão de topo nunca chega à allowlist (é barrada antes por não começar com
        # SELECT — ver test_cte_de_topo_e_rejeitada_pelo_filtro_de_select), então ela
        # passava neste teste sem provar nada sobre CTE nenhuma.
        "SELECT * FROM (WITH y AS (SELECT id FROM coverage) SELECT id FROM y) z",
        # Subquery no FROM: o alias `sub` também não é tabela.
        "SELECT count(*) FROM (SELECT id FROM exclusion) AS sub",
    ],
)
def test_aceita_as_tabelas_do_dominio(sql, monkeypatch):
    """A allowlist não pode ser um muro largo demais.

    Sem este lado, `return "Error"` incondicional passaria nos testes de rejeição e
    derrubaria o worker inteiro em silêncio. Não há banco aqui, então o que se afirma é
    que a query **passou da allowlist** — a execução falha depois, no connect.

    E falhar no connect agora significa `OperationalError` SUBINDO, não voltando como
    texto: falha de infraestrutura deixou de ser resultado de query (ver o `except` do
    `run_query`), justamente pra não vazar host/porta/usuário pela resposta do `/ask`.
    Chegar até lá é a prova que este teste procura, então a exceção conta como sucesso.
    """
    monkeypatch.delenv("DATABASE_URL_RO", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@127.0.0.1:1/none")
    try:
        resultado = run_query(sql)
    except psycopg.OperationalError:
        return   # passou da allowlist e morreu no connect — que é o esperado sem banco
    # As queries que não chegam ao connect (a CTE, barrada antes por não começar com
    # SELECT) caem aqui, e o que vale pra elas é a mesma coisa: não foi a allowlist.
    assert not resultado.startswith("Error: table(s) not allowed")


def test_cte_de_topo_e_rejeitada_pelo_filtro_de_select():
    """LIMITAÇÃO CONHECIDA, registrada: o worker SQL não pode usar `WITH ... SELECT`.

    O filtro 2 exige que a query COMECE com `select`, então uma CTE de topo morre ali —
    não na allowlist. Isso não é acidente que dá pra "consertar" removendo o filtro: no
    Postgres, `WITH x AS (...) DELETE FROM ...` é válido, então é justamente o
    `startswith("select")` que impede escrita disfarçada de CTE. Liberar `WITH` exige que
    a checagem de escrita passe a olhar o statement final, e isso é fatia própria.

    O teste existe porque a alternativa é pior: a CTE de topo estava na lista de queries
    "aprovadas" (passando pelo motivo errado — a asserção era só "não foi a allowlist"),
    e a doc anunciava suporte a CTE. Agora o comportamento real está fixado por teste.
    """
    resultado = run_query("WITH x AS (SELECT id FROM coverage) SELECT count(*) FROM x")
    assert resultado == "Error: only SELECT queries are allowed."


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
