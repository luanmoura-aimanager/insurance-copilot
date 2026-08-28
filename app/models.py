from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Numeric,
    Text,
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


class WhatsAppMessage(Base):
    """Fila de entrada do WhatsApp: uma mensagem recebida, ainda **não** respondida.

    NÃO é domínio. As cinco tabelas de domínio descrevem o produto de seguro; esta é a
    fronteira operacional de um canal — mesma natureza de `cost_event`, e é por isso que
    fica fora do alcance do worker SQL (ver o REVOKE, abaixo).

    O grão é a **mensagem entregue**, não a conversa: a Meta entrega em lote e cada item
    de `value.messages` vira uma linha. Conversa (histórico, turno, contexto) é desenho
    da W2b, quando existir com o que responder.

    **A tabela existe porque o 200 mudou de significado.** Na W1 o webhook lia o evento,
    logava e jogava fora — o 200 dizia "li o que deu". Com o inbox ele passa a dizer
    "aceito com durabilidade", e é essa promessa que autoriza a W2b a processar fora do
    request: a Meta considera a entrega concluída e nunca mais a reenvia.

    **`wamid` é UNIQUE, e é isso que torna o 500 do webhook seguro.** A Meta REENTREGA o
    mesmo `wamid` quando não recebe 200 a tempo — e a rota agora provoca isso de
    propósito quando não consegue gravar (ver `app/main.py`). Sem a unicidade, cada
    retentativa viraria uma linha nova e a W2b responderia a mesma pergunta N vezes,
    pagando N rodadas de LLM. É a idempotência que autoriza o retry, não o contrário.

    **`phone_number_id` fecha o handoff que a W1 anunciou.** Um app secret da Meta cobre
    TODOS os números da conta e o envio da Cloud API é `POST /{phone_number_id}/messages`
    — o comentário em `app/whatsapp.py` carrega esse campo pela borda justamente pra que
    a W2b não fixe um número no código (errado em silêncio no dia do segundo número).
    Como ela varre os pendentes FORA do request, o payload já não existe lá: se o número
    não viajar na linha, aquela justificativa morre exatamente no handoff. Nullable
    porque a borda não descarta mensagem por falta dele.

    **Duas fases, e a coluna `answer` é o que as separa (W2b).** O ciclo é
    `pending` -> `computed` -> `answered` | `failed`, e a resposta é GRAVADA antes de ser
    ENVIADA. O motivo é dinheiro: se o envio falha depois do grafo, um retry ingênuo
    re-pergunta e paga outra rodada de LLM. Com a resposta em `answer`, a retentativa só
    reenvia. `answer` é nullable porque em `pending` ela ainda não existe — `NOT NULL`
    seria mentira sobre a metade da tabela.

    **`attempts` é UMA contagem por mensagem, compartilhada pelas duas fases.** O orçamento
    que ela guarda é "quantas vezes tentamos responder esta pessoa", não "por fase" — e ela
    sobe também quando o GRAFO falha, não só o envio. Sem isso a varredura passa fome: ela
    reivindica `ORDER BY received_at LIMIT 1`, então uma linha cujo `ainvoke` sempre levanta
    (o supervisor não tem `try` — ver CLAUDE.md) seria a primeira escolhida em TODA varredura,
    para sempre, e nenhuma mensagem nova seria respondida. Não há coluna de ERRO de propósito:
    o texto de uma exceção do SQLAlchemy carrega `[parameters: (...)]`, ou seja o telefone e a
    mensagem inteira (achado da W2a). O erro vai pro log; a linha guarda só a contagem.

    **PENDÊNCIA — retenção.** `from_phone` e `text` são PII guardada com finalidade
    (responder), o que é legítimo e é também o que cria a obrigação: sem política de
    expurgo um inbox acumula conversa de usuário para sempre, e o "por finalidade" deixa
    de valer no instante em que a finalidade se cumpre. Esta fatia NÃO implementa
    expurgo. O que falta decidir é a janela (dias após `processed_at`) e quem executa —
    e `processed_at` existe, entre outras coisas, para ser o marco dessa contagem.

    Duas coisas ficam pendentes JUNTO com ela, e as duas são de propósito. (1) Os dois
    timestamps são `TIMESTAMP` ingênuo, como os outros quatro do schema
    (`extracted_at`, os dois `created_at`): uniformidade vale mais do que corrigir um
    isolado, e o servidor roda em UTC, onde não há ambiguidade. Mas `processed_at` será o
    único campo do projeto cujo valor é um PRAZO e não um diagnóstico — se a política
    exigir precisão de fuso, é aí que ele vira `timestamptz`, junto com os outros. (2)
    Não há índice em `processed_at`: um `DELETE ... WHERE processed_at < now() - N` faria
    seq scan, mas o índice certo depende do predicado que a política escolher, e indexar
    para uma query que ninguém escreveu ainda é chutar. Os dois são migration barata numa
    tabela que, por construção, é a que menos cresce.
    """

    __tablename__ = "whatsapp_message"
    __table_args__ = (
        CheckConstraint(
            "status IN ('pending', 'computed', 'answered', 'failed')",
            name="ck_whatsapp_message_status",
        ),
        # A chave de idempotência. Ver o parágrafo do `wamid` no docstring: é o que
        # transforma a reentrega da Meta (e o retry que o nosso 500 provoca) em no-op.
        UniqueConstraint("wamid", name="uq_whatsapp_message_wamid"),
        # Pra varredura de pendentes. A W2b existe e a query agora é conhecida:
        # `WHERE status IN ('pending','computed') ORDER BY received_at LIMIT 1`. O índice
        # que a serviria inteira é PARCIAL e por `received_at` — e continua não sendo este,
        # de propósito: apertá-lo é migration própria, e em regime permanente (`answered`
        # dominando a tabela) é ela que paga. Fica registrado que a promessa da W2a segue
        # aberta, agora sem o chute: o predicado está escrito acima.
        Index("ix_whatsapp_message_status", "status"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    wamid: Mapped[str]                          # id da Meta — a chave de idempotência
    from_phone: Mapped[str]                     # pra QUEM responder
    text: Mapped[str]                           # a pergunta; só mensagem de texto vira linha
    phone_number_id: Mapped[str | None]         # de QUAL número responder (value.metadata)
    # server_default, e não só default de ORM: o INSERT do webhook é Core e a W2b vai
    # mexer nesta coluna por SQL — a autoridade do valor inicial é o banco.
    status: Mapped[str] = mapped_column(server_default="pending")
    # A resposta, gravada ANTES de sair — ver o parágrafo das duas fases no docstring.
    answer: Mapped[str | None] = mapped_column(Text)
    # `server_default`, e não default de ORM, pelo mesmo motivo do `status`: o INSERT do
    # webhook é Core e não lista esta coluna, então a autoridade do valor inicial é o banco.
    attempts: Mapped[int] = mapped_column(server_default="0")
    received_at: Mapped[datetime] = mapped_column(server_default=func.now())
    # Marca estado TERMINAL, e não "foi respondida": preenchido em `answered` E em `failed`,
    # nunca em `computed` (que não é terminal). A W2a dizia "NULL até a W2b"; a W2b afina o
    # significado, e a razão é a retenção — ela conta dias A PARTIR DAQUI, então uma linha
    # `failed` com `processed_at` NULL seria PII que a política nunca conseguiria expirar.
    # O caminho de falha viraria o vazamento permanente.
    processed_at: Mapped[datetime | None]
