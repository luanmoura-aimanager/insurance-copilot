from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, ForeignKey, Index, Numeric, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Dimensão do vetor de embedding. Fixa na coluna (`vector(1024)`), porque o
# Postgres precisa dela para indexar — trocar de modelo de embedding depois exige
# uma migration, não é config. 1024 é a saída do voyage-4-lite, o modelo escolhido
# para o RAG: multilíngue (o corpus é pt-BR), barato, e a dimensão menor mantém o
# índice HNSW leve.
EMBEDDING_DIM = 1024


class Base(DeclarativeBase):
    """Root of all models; Alembic reads Base.metadata to autogenerate migrations."""
    pass


class PolicyDocument(Base):
    """One SUSEP general-terms document (a product, not a customer policy)."""

    __tablename__ = "policy_document"
    __table_args__ = (UniqueConstraint("susep_process", "version"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    insurer: Mapped[str]
    product: Mapped[str]
    susep_process: Mapped[str]
    version: Mapped[str | None]
    property_type: Mapped[str | None]
    pdf_url: Mapped[str]
    pdf_hash: Mapped[str]
    extracted_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Coverage(Base):
    """One coverage within a document. Grain: (insurer x coverage)."""

    __tablename__ = "coverage"
    __table_args__ = (
        CheckConstraint("kind IN ('basic', 'additional')", name="ck_coverage_kind"),
        CheckConstraint(
            "deductible_type IN ('none', 'percentage', 'fixed_amount', 'defined_in_policy')",
            name="ck_coverage_deductible_type",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("policy_document.id"))
    coverage_name: Mapped[str]                  # commercial name (not canonical — see Peril)
    plan: Mapped[str | None]                    # commercial tier (Essencial/Fácil); free text, insurer-specific
    kind: Mapped[str]                           # basic | additional
    deductible_type: Mapped[str | None]         # none | percentage | fixed_amount | defined_in_policy
    deductible_rule_text: Mapped[str | None]    # verbatim POS/deductible rule (feeds RAG)


class Peril(Base):
    """Canonical peril (fire, windstorm, hail...). The real entity behind coverage names."""

    __tablename__ = "peril"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(unique=True)


class CoveragePeril(Base):
    """Join table: insurers bundle perils differently, so coverage <-> peril is many-to-many."""

    __tablename__ = "coverage_peril"

    coverage_id: Mapped[int] = mapped_column(ForeignKey("coverage.id"), primary_key=True)
    peril_id: Mapped[int] = mapped_column(ForeignKey("peril.id"), primary_key=True)


class CostEvent(Base):
    """One LLM call = one row. The grain is deliberate: attributing cost per call (not per
    document or per run) is what lets us answer 'what did extraction cost per doc?' and,
    later, 'which agent burns the budget?'.

    Not part of the domain schema — this is governance/observability.
    """

    __tablename__ = "cost_event"

    id: Mapped[int] = mapped_column(primary_key=True)
    agent_name: Mapped[str]                     # 'extraction' now; supervisor/sql/rag later
    model: Mapped[str]                          # exact model string billed
    input_tokens: Mapped[int]
    output_tokens: Mapped[int]
    # Numeric (not float): billing must be exact — float would drift over thousands of calls.
    cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6))
    # Batch API bills at 50%. Recording the flag makes a half-priced row self-explanatory
    # instead of looking like a pricing bug during an audit.
    batch: Mapped[bool] = mapped_column(default=False)
    label: Mapped[str | None]                   # what this call was about (e.g. susep_<id>)
    # Preenchidas só pelas chamadas feitas dentro de um request (o grafo do /ask); as
    # linhas da extração offline ficam NULL — lá não existe request nem cliente.
    request_id: Mapped[str | None]              # correlaciona TODAS as chamadas de um /ask
    client: Mapped[str | None]                  # identidade do token (app/auth.py) — custo por cliente
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class ClauseChunk(Base):
    """Um pedaço de texto de cláusula pronto para busca semântica (worker RAG).

    Tabela separada de propósito. O texto bruto já existe em `exclusion.clause_text`
    e `coverage.deductible_rule_text`, mas lá o grão é a *linha extraída*; aqui o
    grão é o **chunk**: uma cláusula longa vira vários pedaços, e cada pedaço tem o
    seu próprio vetor. Guardar o embedding na tabela de origem forçaria um chunk por
    linha (péssimo para recuperação) e amarraria o schema de domínio ao modelo de
    embedding da vez — re-embeddar com outro modelo viraria migration em vez de
    reprocessamento.

    `source` + `source_id` são o vínculo de volta com a linha original, que é o que
    permite citar a origem ("essa exclusão está no documento X"); o unique
    (source, source_id, chunk_index) torna a re-indexação idempotente.

    Fica **fora** do TABLES do MCP server de propósito: o worker SQL responde por
    agregação sobre colunas categóricas, e um vetor não é coisa que se responda com
    SELECT — o acesso a esta tabela é do worker RAG, por similaridade.
    """

    __tablename__ = "clause_chunk"
    __table_args__ = (
        CheckConstraint(
            "source IN ('exclusion', 'deductible_rule')", name="ck_clause_chunk_source"
        ),
        UniqueConstraint(
            "source", "source_id", "chunk_index", name="uq_clause_chunk_source_chunk"
        ),
        # O índice HNSW é criado na migration com SQL cru (o Alembic não emite
        # operator class), mas precisa estar declarado aqui: sem isso o próximo
        # `alembic revision --autogenerate` enxerga um índice que o modelo não
        # conhece e emite um drop_index — perder o índice de vetor de brinde numa
        # migration sobre outro assunto. `alembic check` é quem trava isso.
        Index(
            "ix_clause_chunk_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("policy_document.id"))
    coverage_id: Mapped[int | None] = mapped_column(ForeignKey("coverage.id"))
    source: Mapped[str]                         # exclusion | deductible_rule
    source_id: Mapped[int | None]               # id da linha de origem
    chunk_index: Mapped[int] = mapped_column(default=0)   # posição dentro do texto de origem
    text: Mapped[str]
    # Nullable porque o chunking e o embedding são passos separados: a linha nasce
    # com o texto e recebe o vetor depois (R2). NULL = "ainda não indexado".
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM))
    embedding_model: Mapped[str | None]         # ex.: voyage-4-lite — qual modelo gerou o vetor
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())


class Exclusion(Base):
    """One exclusion. Scope is either general (document-wide) or tied to a coverage."""

    __tablename__ = "exclusion"
    __table_args__ = (
        CheckConstraint("scope IN ('general', 'coverage')", name="ck_exclusion_scope"),
        CheckConstraint(
            "(scope = 'general' AND coverage_id IS NULL) OR (scope = 'coverage' AND coverage_id IS NOT NULL)",
            name="ck_exclusion_scope_coverage_id",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("policy_document.id"))
    coverage_id: Mapped[int | None] = mapped_column(ForeignKey("coverage.id"))
    scope: Mapped[str]                          # general | coverage
    clause_text: Mapped[str]                    # verbatim clause (feeds RAG)
