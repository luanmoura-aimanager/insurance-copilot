"""clause_chunk: ponteiro polimórfico -> arco exclusivo (+ revoke do insurance_ro)

Revision ID: a4c91e5d7f28
Revises: d3f8a1c60b47
Create Date: 2026-08-06 18:05:12.447301

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a4c91e5d7f28'
down_revision: Union[str, Sequence[str], None] = 'd3f8a1c60b47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Troca (source, source_id) por (exclusion_id, coverage_id) com FK de verdade.

    Escrita à mão, como a d3f8a1c60b47: o autogenerate não emite nem o REVOKE nem a
    ordem correta de drop/add em volta do check.

    Por que DROP COLUMN é seguro aqui: `clause_chunk` está vazia em todo ambiente —
    a R1 entregou só o armazenamento, e nada chunka nem embeda ainda (R2/R3). Não há
    dado a migrar, então não existe passo de backfill. Se um dia houver, este é o
    ponto que precisaria de UPDATE ... FROM antes dos drops.
    """
    # 1. Fora o desenho antigo. O unique some junto porque (source, source_id,
    #    chunk_index) não impunha idempotência nenhuma: source_id era nullable e o
    #    Postgres trata NULLs como distintos, então duplicatas entravam sem reclamar.
    op.drop_constraint('ck_clause_chunk_source', 'clause_chunk', type_='check')
    op.drop_constraint('uq_clause_chunk_source_chunk', 'clause_chunk', type_='unique')
    op.drop_column('clause_chunk', 'source')
    op.drop_column('clause_chunk', 'source_id')

    # 2. O braço que faltava do arco. `coverage_id` já existe (era a cópia
    #    denormalizada da cobertura dona) e passa a ser o outro braço.
    op.add_column('clause_chunk', sa.Column('exclusion_id', sa.Integer(), nullable=True))
    op.create_foreign_key(
        'clause_chunk_exclusion_id_fkey', 'clause_chunk', 'exclusion',
        ['exclusion_id'], ['id'],
    )

    # 3. XOR: exatamente um dos dois braços preenchido. `<>` entre booleanos é XOR.
    op.create_check_constraint(
        'ck_clause_chunk_exactly_one_source',
        'clause_chunk',
        '(exclusion_id IS NOT NULL) <> (coverage_id IS NOT NULL)',
    )
    op.create_unique_constraint(
        'uq_clause_chunk_exclusion_chunk', 'clause_chunk', ['exclusion_id', 'chunk_index']
    )
    op.create_unique_constraint(
        'uq_clause_chunk_coverage_chunk', 'clause_chunk', ['coverage_id', 'chunk_index']
    )

    # 4. O Postgres não indexa coluna filha de FK sozinho. Sem este índice, todo
    #    DELETE em policy_document (o fluxo do --force) faz seq scan na que é, por
    #    desenho, a maior tabela do schema. exclusion_id/coverage_id já saem
    #    indexadas pelos dois UNIQUE acima.
    op.create_index('ix_clause_chunk_document_id', 'clause_chunk', ['document_id'])

    # 5. O REVOKE é CIRÚRGICO, e de propósito: tira o SELECT desta tabela da role
    #    read-only do worker SQL, sem desfazer o ALTER DEFAULT PRIVILEGES da
    #    4b285ffad59b — que continua valendo para as tabelas futuras. Sem isto, o
    #    "clause_chunk não é exposta ao worker SQL" era só omissão no get_schema():
    #    o `run_query` não tem allowlist de tabela, então um SELECT * escrito pelo
    #    LLM despejaria vetores de 1024 floats no contexto da próxima chamada.
    op.execute("REVOKE SELECT ON clause_chunk FROM insurance_ro")


def downgrade() -> None:
    """Desfaz na ordem inversa, incluindo o GRANT de volta."""
    op.execute("GRANT SELECT ON clause_chunk TO insurance_ro")

    op.drop_index('ix_clause_chunk_document_id', table_name='clause_chunk')
    op.drop_constraint('uq_clause_chunk_coverage_chunk', 'clause_chunk', type_='unique')
    op.drop_constraint('uq_clause_chunk_exclusion_chunk', 'clause_chunk', type_='unique')
    op.drop_constraint('ck_clause_chunk_exactly_one_source', 'clause_chunk', type_='check')

    op.drop_constraint('clause_chunk_exclusion_id_fkey', 'clause_chunk', type_='foreignkey')
    op.drop_column('clause_chunk', 'exclusion_id')

    # Volta o desenho antigo. `source` é NOT NULL lá atrás; a tabela está vazia, então
    # não é preciso server_default temporário para readicionar a coluna.
    op.add_column('clause_chunk', sa.Column('source_id', sa.Integer(), nullable=True))
    op.add_column('clause_chunk', sa.Column('source', sa.String(), nullable=False))
    op.create_unique_constraint(
        'uq_clause_chunk_source_chunk', 'clause_chunk', ['source', 'source_id', 'chunk_index']
    )
    op.create_check_constraint(
        'ck_clause_chunk_source', 'clause_chunk', "source IN ('exclusion', 'deductible_rule')"
    )
