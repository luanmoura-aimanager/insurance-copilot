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


# O REVOKE/GRANT depende de uma role, que é objeto de CLUSTER, não de banco — e a
# cadeia de migrations só garante objetos do banco. Um `pg_dump` restaurado num
# cluster novo (pg_dump não emite roles) ou um segundo banco num cluster onde a
# 4b285ffad59b abortou no `CREATE ROLE ... already exists` chega aqui sem
# `insurance_ro`, e o `alembic upgrade head` do Procfile mata o boot do web. O
# Postgres não tem `REVOKE ... IF EXISTS` pra role, então o guard é explícito.
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
    #    denormalizada da cobertura dona) e passa a ser o outro braço. A FK simples
    #    dele criada pela d3f8a1c60b47 sai: as duas FKs do arco são compostas (passo 4).
    op.add_column('clause_chunk', sa.Column('exclusion_id', sa.Integer(), nullable=True))
    op.drop_constraint('clause_chunk_coverage_id_fkey', 'clause_chunk', type_='foreignkey')

    # 3. XOR: exatamente um dos dois braços preenchido. `<>` entre booleanos é XOR.
    op.create_check_constraint(
        'ck_clause_chunk_exactly_one_source',
        'clause_chunk',
        '(exclusion_id IS NOT NULL) <> (coverage_id IS NOT NULL)',
    )

    # 4. As FKs do arco são COMPOSTAS com document_id. Uma FK simples
    #    (exclusion_id -> exclusion.id) deixaria passar um chunk com document_id=B e
    #    origem no documento A — e o delete_document_by_hash(A), que apaga chunks por
    #    document_id, não veria esse chunk e estouraria ForeignKeyViolation ao apagar
    #    a `exclusion`, deixando o documento pela metade. Exatamente o bug que esta
    #    migration existe pra fechar, reaberto por uma porta lateral.
    #
    #    MATCH SIMPLE (o padrão) é o que faz a FK composta conviver com o arco: se
    #    qualquer coluna da FK é NULL, a constraint não é checada — então o braço não
    #    usado fica inerte e só o braço preenchido é validado.
    #
    #    O Postgres exige unique/PK no lado referenciado, daí os dois uniques
    #    (document_id, id) — redundantes como unicidade (id já é PK), presentes só
    #    como alvo da FK.
    op.create_unique_constraint(
        'uq_exclusion_document_id_id', 'exclusion', ['document_id', 'id']
    )
    op.create_unique_constraint(
        'uq_coverage_document_id_id', 'coverage', ['document_id', 'id']
    )
    op.create_foreign_key(
        'fk_clause_chunk_exclusion', 'clause_chunk', 'exclusion',
        ['document_id', 'exclusion_id'], ['document_id', 'id'],
    )
    op.create_foreign_key(
        'fk_clause_chunk_coverage', 'clause_chunk', 'coverage',
        ['document_id', 'coverage_id'], ['document_id', 'id'],
    )

    # 5. Idempotência da re-indexação. São índices únicos PARCIAIS, não constraints:
    #    metade das linhas tem exclusion_id NULL e a outra metade coverage_id NULL,
    #    por desenho — um índice total carregaria uma entrada morta por linha do outro
    #    braço, o dobro do tamanho na tabela que também carrega o HNSW. O predicado
    #    não afrouxa nada: as linhas que ele exclui são as que o NULL já tornava
    #    não-conflitantes.
    op.create_index(
        'uq_clause_chunk_exclusion_chunk', 'clause_chunk', ['exclusion_id', 'chunk_index'],
        unique=True, postgresql_where=sa.text('exclusion_id IS NOT NULL'),
    )
    op.create_index(
        'uq_clause_chunk_coverage_chunk', 'clause_chunk', ['coverage_id', 'chunk_index'],
        unique=True, postgresql_where=sa.text('coverage_id IS NOT NULL'),
    )

    # 6. O Postgres não indexa coluna filha de FK sozinho. Sem este índice, todo
    #    DELETE em policy_document (o fluxo do --force) faz seq scan na que é, por
    #    desenho, a maior tabela do schema. exclusion_id/coverage_id já saem
    #    indexadas pelos dois índices parciais acima.
    op.create_index('ix_clause_chunk_document_id', 'clause_chunk', ['document_id'])

    # 7. O REVOKE é CIRÚRGICO, e de propósito: tira o SELECT desta tabela da role
    #    read-only do worker SQL, sem desfazer o ALTER DEFAULT PRIVILEGES da
    #    4b285ffad59b — que continua valendo para as tabelas futuras. Note que ele é
    #    só uma das duas camadas: a outra é a allowlist de tabela do run_query
    #    (mcp_servers/postgres_mcp_server.py), que vale mesmo quando o worker conecta
    #    com o DATABASE_URL admin porque DATABASE_URL_RO não foi configurada.
    op.execute(_if_role_exists("REVOKE SELECT ON clause_chunk FROM insurance_ro;"))


def downgrade() -> None:
    """Desfaz na ordem inversa, incluindo o GRANT de volta."""
    op.execute(_if_role_exists("GRANT SELECT ON clause_chunk TO insurance_ro;"))

    # `source` volta como NOT NULL e `exclusion_id` some — não há como reconstruir o
    # ponteiro antigo a partir do arco sem inventar dado. Os chunks são derivados
    # (texto + embedding saem de re-indexar as cláusulas, que continuam em
    # exclusion.clause_text / coverage.deductible_rule_text), então descartá-los é
    # recuperável; deixar o ADD COLUMN estourar com "column contains null values"
    # depois de já ter dropado exclusion_id não é. Enquanto a R2 não existir a tabela
    # está vazia e isto é no-op.
    op.execute("DELETE FROM clause_chunk")

    op.drop_index('ix_clause_chunk_document_id', table_name='clause_chunk')
    op.drop_index('uq_clause_chunk_coverage_chunk', table_name='clause_chunk')
    op.drop_index('uq_clause_chunk_exclusion_chunk', table_name='clause_chunk')

    op.drop_constraint('fk_clause_chunk_coverage', 'clause_chunk', type_='foreignkey')
    op.drop_constraint('fk_clause_chunk_exclusion', 'clause_chunk', type_='foreignkey')
    op.drop_constraint('uq_coverage_document_id_id', 'coverage', type_='unique')
    op.drop_constraint('uq_exclusion_document_id_id', 'exclusion', type_='unique')

    op.drop_constraint('ck_clause_chunk_exactly_one_source', 'clause_chunk', type_='check')
    op.drop_column('clause_chunk', 'exclusion_id')

    # Devolve a FK simples de coverage_id que a d3f8a1c60b47 tinha criado.
    op.create_foreign_key(
        'clause_chunk_coverage_id_fkey', 'clause_chunk', 'coverage', ['coverage_id'], ['id'],
    )

    # Volta o desenho antigo. `source` é NOT NULL lá atrás; a tabela acabou de ser
    # esvaziada, então não é preciso server_default temporário.
    op.add_column('clause_chunk', sa.Column('source_id', sa.Integer(), nullable=True))
    op.add_column('clause_chunk', sa.Column('source', sa.String(), nullable=False))
    op.create_unique_constraint(
        'uq_clause_chunk_source_chunk', 'clause_chunk', ['source', 'source_id', 'chunk_index']
    )
    op.create_check_constraint(
        'ck_clause_chunk_source', 'clause_chunk', "source IN ('exclusion', 'deductible_rule')"
    )
