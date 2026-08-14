from sqlalchemy import text

EXPECTED_TABLES = {
    "policy_document",
    "coverage",
    "peril",
    "coverage_peril",
    "exclusion",
    # A asserção é subconjunto (<=), então uma tabela que não esteja listada aqui
    # simplesmente deixa de ser coberta: dá pra remover o create_table dela da
    # migration e este teste continua verde. Por isso o schema cresce aqui junto.
    "cost_event",
    "clause_chunk",
}


async def test_migration_created_all_tables(db_session):
    """`alembic upgrade head` (run in the container fixture) created the schema."""
    result = await db_session.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public'"
        )
    )
    tables = {row[0] for row in result}
    assert EXPECTED_TABLES <= tables


def test_migration_in_process_nao_mexe_no_logging_do_pytest(db_url):
    """A migration roda DENTRO do processo do pytest — e não pode reconfigurar o logging.

    `fileConfig`, em `alembic/env.py`, é ação global no processo, e com os defaults faz
    duas estragas independentes: (1) marca `disabled = True` em todo logger já criado
    (`disable_existing_loggers`), e (2) SUBSTITUI os handlers da raiz, por causa do
    `[logger_root] handlers = console` do alembic.ini — e é exatamente na raiz que vive o
    `LogCaptureHandler` que alimenta o `caplog`, pelo tempo de cada teste.

    O sintoma das duas é o mesmo, e é o pior possível: teste de log que passa sem ver log
    nenhum. Quem paga a conta é `tests/test_falha_interna.py`, o único lugar que prova que
    uma falha de worker deixa traceback com `request_id`/`client` pro operador — a ÚNICA
    descrição do incidente, já que o usuário recebe uma frase genérica. A estraga (1) já
    aconteceu de verdade (os testes de log passavam sozinhos e falhavam na suíte completa);
    a (2) hoje não morde só porque o container sobe num teste que não usa `caplog`, o que
    é ordem alfabética de arquivo, não garantia.

    A defesa é o `configure_logger=False` que `tests/conftest.py` passa nos `attributes` da
    Config e o `env.py` respeita. Este teste repete a mesma chamada (o `upgrade` é no-op,
    o banco já está na head) e afirma as duas propriedades; sem a flag, fica vermelho.
    """
    import logging

    from alembic import command
    from alembic.config import Config

    sentinela = logging.Handler()
    logging.getLogger().addHandler(sentinela)
    try:
        cfg = Config("alembic.ini")
        cfg.attributes["configure_logger"] = False
        command.upgrade(cfg, "head")

        assert sentinela in logging.getLogger().handlers, "os handlers da raiz foram trocados"
        assert not logging.getLogger("app.agents.graph").disabled, "logger do app desligado"
    finally:
        logging.getLogger().removeHandler(sentinela)
