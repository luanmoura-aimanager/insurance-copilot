from datetime import datetime
from decimal import Decimal

from sqlalchemy import CheckConstraint, ForeignKey, Numeric, UniqueConstraint, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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
