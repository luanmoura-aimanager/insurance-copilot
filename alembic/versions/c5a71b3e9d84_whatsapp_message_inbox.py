"""whatsapp_message: inbox do webhook (+ revoke do insurance_ro)

Revision ID: c5a71b3e9d84
Revises: a4c91e5d7f28
Create Date: 2026-08-23 10:12:44.918273

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c5a71b3e9d84'
down_revision: Union[str, Sequence[str], None] = 'a4c91e5d7f28'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# Cópia deliberada do helper da a4c91e5d7f28 (migration não importa migration): role é
# objeto de CLUSTER, não de banco, e a cadeia de migrations só garante objetos do banco.
# Um `pg_dump` restaurado num cluster novo (pg_dump não emite roles) chega aqui sem
# `insurance_ro` e o REVOKE mata o `alembic upgrade head` do Procfile — ou seja, o web
# process não sobe. O Postgres não tem `REVOKE ... IF EXISTS` pra role.
def _if_role_exists(statement: str) -> str:
    return f"""
    DO $$
    BEGIN
        IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'insurance_ro') THEN
            {statement}
        END IF;
    END
    $$;
    """


def upgrade() -> None:
    """Cria a fila de entrada do WhatsApp e a esconde do worker SQL.

    Escrita à mão pelo mesmo motivo das duas anteriores: o autogenerate enxerga a tabela
    e os índices, mas não emite o REVOKE — e o REVOKE aqui não é higiene, é o que impede
    PII de vazar pro contexto de uma chamada de LLM (ver o passo 3).
    """
    # 1. A tabela. `wamid` UNIQUE é a chave de idempotência: é ela que faz a reentrega
    #    da Meta (e o retry que o 500 do webhook provoca de propósito quando não
    #    consegue gravar) reentrar pelo ON CONFLICT DO NOTHING sem duplicar linha.
    op.create_table('whatsapp_message',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('wamid', sa.String(), nullable=False),
    sa.Column('from_phone', sa.String(), nullable=False),
    sa.Column('text', sa.String(), nullable=False),
    sa.Column('phone_number_id', sa.String(), nullable=True),
    sa.Column('status', sa.String(), server_default='pending', nullable=False),
    sa.Column('received_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.Column('processed_at', sa.DateTime(), nullable=True),
    sa.CheckConstraint("status IN ('pending', 'answered', 'failed')", name='ck_whatsapp_message_status'),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('wamid', name='uq_whatsapp_message_wamid')
    )

    # 2. Pra varredura de pendentes da W2b. Declarado também no __table_args__ do
    #    modelo: índice que existe só aqui vira `drop_index` no próximo autogenerate.
    op.create_index('ix_whatsapp_message_status', 'whatsapp_message', ['status'])

    # 3. O REVOKE, e ele NÃO é decorativo. A 4b285ffad59b deixou um
    #    `ALTER DEFAULT PRIVILEGES ... GRANT SELECT ON TABLES TO insurance_ro`
    #    permanente, então toda tabela nova criada pelo admin JÁ NASCE legível pela role
    #    do worker SQL. Sem isto, o worker que responde sobre apólices passaria a
    #    enxergar conversa de usuário, e um `SELECT * FROM whatsapp_message` escrito pelo
    #    LLM despejaria telefone e texto de terceiro no contexto da chamada seguinte.
    #    Mesmo argumento do clause_chunk, dano diferente: lá era custo, aqui é PII.
    #
    #    A outra camada (independente desta) é a allowlist de tabela do `run_query`, que
    #    já rejeita a tabela sem mudança de código — nenhuma das duas cobre o caso da
    #    outra: o REVOKE não participa de nada quando DATABASE_URL_RO não está setada e
    #    o `_conninfo()` cai pro DATABASE_URL admin.
    op.execute(_if_role_exists("REVOKE SELECT ON whatsapp_message FROM insurance_ro;"))


def downgrade() -> None:
    """Derruba a tabela — e com ela o inbox inteiro.

    Sem GRANT de volta: o objeto some, e os privilégios sobre ele vão junto (ao
    contrário da a4c91e5d7f28, onde a tabela sobrevivia ao downgrade).

    Sem DELETE prévio, e sem eufemismo sobre o que isso significa: ao contrário do
    `clause_chunk`, este dado NÃO é derivado. Um chunk se reconstrói a partir do texto
    que continua em `exclusion.clause_text`; uma pergunta de usuário não se reconstrói de
    lugar nenhum, e a Meta não reenvia o que já recebeu 200. Um downgrade aqui descarta
    perguntas — inclusive as que ainda não foram respondidas.
    """
    op.drop_index('ix_whatsapp_message_status', table_name='whatsapp_message')
    op.drop_table('whatsapp_message')
