from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Numeric,
    UniqueConstraint,
    func,
    text,
)
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
        # Redundante como unicidade (`id` já é PK), mas é o alvo exigido pela FK
        # composta de `clause_chunk`: o Postgres só aceita referenciar um conjunto de
        # colunas que tenha unique/PK. Ver o comentário do arco em ClauseChunk.
        UniqueConstraint("document_id", "id", name="uq_coverage_document_id_id"),
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

    A origem do chunk é um **arco exclusivo**: ou `exclusion_id` (o chunk é um pedaço
    de `exclusion.clause_text`) ou `coverage_id` (é um pedaço da regra de franquia,
    `coverage.deductible_rule_text`) — exatamente um dos dois, nunca os dois, nunca
    nenhum. O check `ck_clause_chunk_exactly_one_source` é quem impõe isso.

    Substituiu o ponteiro polimórfico (`source` + `source_id`) da R1, que tinha dois
    furos. (1) `source_id` era um id sem FK: apagar a exclusão de origem deixava o
    chunk apontando pro vazio, e como a re-extração reusa ids, o chunk podia passar a
    citar uma cláusula *diferente* — justamente a garantia de "citar a origem" que a
    coluna existia pra dar. (2) `UniqueConstraint(source, source_id, chunk_index)` não
    tornava a re-indexação idempotente coisa nenhuma: `source_id` era nullable e o
    Postgres trata NULLs como distintos num índice único, então dois chunks idênticos
    com `source_id IS NULL` entravam os dois. Com FKs de verdade o banco impede o
    ponteiro órfão, e os uniques por (origem, chunk_index) valem porque as colunas
    participantes são NOT NULL sempre que a linha usa aquele braço do arco.

    As FKs do arco são **compostas com `document_id`**, o que impede o terceiro caso
    ruim: um chunk cujo `document_id` aponta pra um documento e cuja origem pertence a
    outro. Sem isso o `delete_document_by_hash` (que apaga chunks por `document_id`)
    não enxergaria esse chunk e estouraria na FK ao apagar a exclusão — o mesmo
    documento pela metade de sempre. Detalhe em `__table_args__`.

    `coverage_id` mudou de significado: era cópia denormalizada da cobertura dona da
    exclusão (podia divergir de `exclusion.coverage_id`), agora é um dos dois braços
    do arco — preenchido *só* quando o chunk é a regra de franquia daquela cobertura.

    Fica **fora** do alcance do worker SQL de propósito: o worker responde por
    agregação sobre colunas categóricas, e um vetor não é coisa que se responda com
    SELECT. Isso é imposto por privilégio (`REVOKE SELECT ... FROM insurance_ro`, na
    migration a4c91e5d7f28), não só pela omissão no `get_schema()` do MCP server — o
    acesso a esta tabela é do worker RAG, por similaridade.
    """

    __tablename__ = "clause_chunk"
    __table_args__ = (
        # Arco exclusivo: `<>` entre dois booleanos é XOR. Exatamente um preenchido.
        CheckConstraint(
            "(exclusion_id IS NOT NULL) <> (coverage_id IS NOT NULL)",
            name="ck_clause_chunk_exactly_one_source",
        ),
        # As FKs do arco são COMPOSTAS com `document_id` de propósito. Com FKs
        # simples (`exclusion_id -> exclusion.id`) nada obrigava `document_id` a ser
        # o documento da origem: um chunk com document_id=B e exclusion_id de uma
        # exclusão do documento A entrava sem reclamar, e aí
        # `delete_document_by_hash(A)` — que apaga os chunks por document_id — não
        # via esse chunk e estourava ForeignKeyViolation em `exclusion`, deixando o
        # documento pela metade. Exatamente o bug que esta fatia existe pra matar.
        # MATCH SIMPLE (o padrão) é o que faz isto compor com o arco: quando o braço
        # está NULL a FK inteira não é checada, então só o braço em uso é validado.
        ForeignKeyConstraint(
            ["document_id", "exclusion_id"],
            ["exclusion.document_id", "exclusion.id"],
            name="fk_clause_chunk_exclusion",
        ),
        ForeignKeyConstraint(
            ["document_id", "coverage_id"],
            ["coverage.document_id", "coverage.id"],
            name="fk_clause_chunk_coverage",
        ),
        # Idempotência da re-indexação, agora de verdade: nas linhas em que a coluna
        # de origem está preenchida ela é NOT NULL, então o unique morde (o problema
        # do NULL-distinto do desenho antigo não existe mais).
        #
        # São índices únicos PARCIAIS, não UNIQUE constraints: por desenho metade das
        # linhas tem `exclusion_id` NULL e a outra metade `coverage_id` NULL, então um
        # índice total carregaria uma entrada morta por linha do outro braço — o dobro
        # do tamanho, na tabela que também carrega o índice HNSW. O predicado não
        # enfraquece a garantia (as linhas excluídas são justamente as que o NULL já
        # tornava não-conflitantes) e continua servindo a validação da FK, porque
        # `exclusion_id = $1` implica `exclusion_id IS NOT NULL`.
        Index(
            "uq_clause_chunk_exclusion_chunk",
            "exclusion_id",
            "chunk_index",
            unique=True,
            postgresql_where=text("exclusion_id IS NOT NULL"),
        ),
        Index(
            "uq_clause_chunk_coverage_chunk",
            "coverage_id",
            "chunk_index",
            unique=True,
            postgresql_where=text("coverage_id IS NOT NULL"),
        ),
        # O Postgres não indexa coluna filha de FK sozinho, e esta é, por desenho, a
        # maior tabela do schema: sem isto todo DELETE em policy_document (o fluxo do
        # --force) vira seq scan pra validar a FK. `exclusion_id` e `coverage_id` já
        # saem indexadas pelos dois índices parciais acima — só `document_id` precisa.
        Index("ix_clause_chunk_document_id", "document_id"),
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
    # Arco exclusivo — exatamente um destes dois é NOT NULL (ver o check acima). As
    # FKs não ficam aqui na coluna: são compostas com document_id, declaradas em
    # __table_args__.
    exclusion_id: Mapped[int | None]
    coverage_id: Mapped[int | None]
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
        # Idem `coverage`: alvo da FK composta de `clause_chunk`, não uma regra de
        # unicidade nova.
        UniqueConstraint("document_id", "id", name="uq_exclusion_document_id_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("policy_document.id"))
    coverage_id: Mapped[int | None] = mapped_column(ForeignKey("coverage.id"))
    scope: Mapped[str]                          # general | coverage
    clause_text: Mapped[str]                    # verbatim clause (feeds RAG)
