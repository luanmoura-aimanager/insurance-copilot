"""add clause_chunk (pgvector)

Revision ID: d3f8a1c60b47
Revises: 91956643f279
Create Date: 2026-08-06 10:12:44.019823

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'd3f8a1c60b47'
down_revision: Union[str, Sequence[str], None] = '91956643f279'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Infraestrutura de vetores do worker RAG: extensão + tabela + índice.

    Escrita à mão, **não** autogenerate: o autogenerate enxerga a tabela e a coluna
    `vector(1024)`, mas não emite nem o CREATE EXTENSION (sem ele o CREATE TABLE
    falha com "type vector does not exist") nem o índice HNSW (alembic não modela
    métodos de acesso com operator class). A ordem abaixo é obrigatória: extensão →
    tabela → índice.
    """
    # A imagem do compose e a dos testes são pgvector/pgvector:pg16, que já traz a
    # extensão compilada; aqui só a habilitamos neste banco. IF NOT EXISTS porque em
    # provedores gerenciados (Railway) ela pode já vir criada.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table('clause_chunk',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('document_id', sa.Integer(), nullable=False),
    sa.Column('coverage_id', sa.Integer(), nullable=True),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('source_id', sa.Integer(), nullable=True),
    sa.Column('chunk_index', sa.Integer(), nullable=False),
    sa.Column('text', sa.String(), nullable=False),
    sa.Column('embedding', Vector(1024), nullable=True),
    sa.Column('embedding_model', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), server_default=sa.text('now()'), nullable=False),
    sa.CheckConstraint("source IN ('exclusion', 'deductible_rule')", name='ck_clause_chunk_source'),
    sa.ForeignKeyConstraint(['coverage_id'], ['coverage.id'], ),
    sa.ForeignKeyConstraint(['document_id'], ['policy_document.id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source', 'source_id', 'chunk_index', name='uq_clause_chunk_source_chunk')
    )

    # HNSW com vector_cosine_ops: a busca do RAG é por cosseno, e no pgvector o
    # índice só é usado se a operator class casar com o operador da query (<=>).
    op.execute(
        "CREATE INDEX ix_clause_chunk_embedding ON clause_chunk "
        "USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    """Desfaz na ordem inversa — índice, tabela.

    A extensão **não** é dropada de propósito: ela é do banco, não desta tabela.
    Outro objeto (ou outro branch de migration) pode depender dela, e um
    DROP EXTENSION levaria junto qualquer coluna `vector` que exista.
    """
    op.execute("DROP INDEX IF EXISTS ix_clause_chunk_embedding")
    op.drop_table('clause_chunk')
