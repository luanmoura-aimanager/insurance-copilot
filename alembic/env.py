import os
from logging.config import fileConfig

from dotenv import load_dotenv
from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context

from app.models import Base
from app.db_url import normalize_url

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Read DATABASE_URL from the environment and normalize to a SYNC driver.
# The app runs async (asyncpg), but Alembic runs synchronously, so migrations use
# psycopg2. Railway may hand us postgres:// or postgresql://; normalize all cases.
load_dotenv()

config.set_main_option("sqlalchemy.url", normalize_url(os.environ["DATABASE_URL"], "psycopg2"))

# Interpret the config file for Python logging.
# This line sets up loggers basically.
#
# **`fileConfig` NÃO roda quando a migration é chamada de dentro de outro processo Python**
# — quem passa `configure_logger=False` é `tests/conftest.py`, que roda
# `command.upgrade()` no processo do pytest. Configurar logging é ação GLOBAL no processo,
# e aqui ela fazia duas estragas independentes, as duas silenciosas:
#
#   1. `disable_existing_loggers` (default True) marca `disabled = True` em TODO logger já
#      criado — a partir da fixture do container, `logging.getLogger("app.agents.graph")`
#      ficava desligado e qualquer `caplog` posterior não via registro nenhum.
#   2. `[logger_root] handlers = console` no alembic.ini SUBSTITUI os handlers da raiz — e
#      o `LogCaptureHandler` do pytest (o que alimenta o `caplog`) vive exatamente lá,
#      pelo tempo de cada teste. Ou seja: mesmo com o item 1 resolvido, o teste que por
#      acaso disparar a fixture do container perde o caplog no meio.
#
# As duas dão falso verde no mesmo lugar: os testes que provam que uma falha de worker
# deixa traceback pro operador — a ÚNICA descrição do incidente, já que o usuário recebe
# uma frase genérica (ver `_falha_de_worker` em `app/agents/graph.py`). O item 2 hoje só
# não morde porque o container sobe num teste que não usa caplog, o que é ordem alfabética
# de arquivo, não garantia. Não configurar nada é o único jeito de não mexer no logging de
# quem chamou; o `alembic upgrade head` do CLI (Procfile) não passa a flag e segue
# configurando normalmente, que é onde essa configuração serve pra algo.
#
# `disable_existing_loggers=False` fica como rede pra QUALQUER outro chamador in-process
# que não conheça a flag (um script, um notebook): sozinho não basta — ver item 2 — mas
# sem ele o estrago do item 1 é irreversível pro processo.
if config.config_file_name is not None and config.attributes.get("configure_logger", True):
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# Target schema for 'autogenerate' — every table that inherits from Base.
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
