"""add read-only role insurance_ro

Revision ID: 4b285ffad59b
Revises: fbb0f178ab3a
Create Date: 2026-07-22 15:53:33.121626

"""
import os
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4b285ffad59b'
down_revision: Union[str, Sequence[str], None] = 'fbb0f178ab3a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Cria a role read-only `insurance_ro`.

    O worker SQL do MCP conecta com essa role: mesmo que o filtro textual
    "só SELECT" falhe, o Postgres barra qualquer escrita no nível de privilégio.
    """
    conn = op.get_bind()
    # Role admin que o Alembic usa para criar tabelas — ancora os default
    # privileges nela em vez de hardcodar (dev = `insurance`, Railway = outra).
    admin = conn.exec_driver_sql("SELECT current_user").scalar()

    # 2a. cria a role (login; senha vem de env var — nunca hardcodar segredo real)
    ro_password = os.environ.get("RO_ROLE_PASSWORD", "changeme_dev_only")
    op.execute(f"CREATE ROLE insurance_ro LOGIN PASSWORD '{ro_password}'")

    # 2b. deixa conectar + enxergar o schema
    op.execute(
        "GRANT CONNECT ON DATABASE "
        + conn.exec_driver_sql("SELECT current_database()").scalar()
        + " TO insurance_ro"
    )
    op.execute("GRANT USAGE ON SCHEMA public TO insurance_ro")

    # 2c. tabelas já existentes (snapshot de agora)
    op.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO insurance_ro")

    # 2d. tabelas futuras criadas pela role admin (regra permanente)
    op.execute(
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {admin} IN SCHEMA public "
        "GRANT SELECT ON TABLES TO insurance_ro"
    )


def downgrade() -> None:
    """Remove a role e todos os grants, na ordem inversa."""
    conn = op.get_bind()
    admin = conn.exec_driver_sql("SELECT current_user").scalar()
    op.execute(
        f"ALTER DEFAULT PRIVILEGES FOR ROLE {admin} IN SCHEMA public "
        "REVOKE SELECT ON TABLES FROM insurance_ro"
    )
    op.execute("REVOKE ALL ON ALL TABLES IN SCHEMA public FROM insurance_ro")
    op.execute("REVOKE ALL ON SCHEMA public FROM insurance_ro")
    op.execute("DROP ROLE IF EXISTS insurance_ro")
