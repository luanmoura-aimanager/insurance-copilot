"""Contrato de saída da extração (Pydantic).

Este é o shape que o LLM deve devolver — uma árvore aninhada, não linhas de tabela.
O LLM não conhece ids do banco, então FKs inteiras não existem aqui: a hierarquia é
expressa por aninhamento (coverage dentro de document, peril dentro de coverage).
A camada de persistência (Fatia 2) achata esta árvore nas 5 tabelas de app/models.py.

Os enums abaixo viram o `enum` do JSON Schema passado ao LLM como tool — é assim que
o vocabulário de perigos fica FIXO (entity resolution no momento da extração, não depois).
"""

from enum import Enum

from pydantic import BaseModel, Field


class Peril(str, Enum):
    """Perigos canônicos — a entidade real por trás do nome comercial da cobertura.

    Lista inicial das 2 CGs do piloto (data/pilot_findings.md). Perigos são DADOS
    (linhas da tabela peril), por isso ficam em pt-BR snake_case, não em inglês.
    Seguradoras empacotam perigos diferente sob nomes comerciais distintos; canonizar
    aqui é o que faz "quem cobre vendaval?" funcionar entre seguradoras.
    """

    incendio_explosao = "incendio_explosao"
    danos_eletricos = "danos_eletricos"
    vendaval = "vendaval"
    granizo = "granizo"
    fumaca = "fumaca"
    impacto_veiculos = "impacto_veiculos"
    roubo_furto_qualificado = "roubo_furto_qualificado"
    rc_familiar = "rc_familiar"
    perda_aluguel = "perda_aluguel"
    quebra_vidros = "quebra_vidros"
    desmoronamento = "desmoronamento"
    alagamento = "alagamento"


class Kind(str, Enum):
    """coverage.kind — explícito na CG."""

    basic = "basic"
    additional = "additional"


class DeductibleType(str, Enum):
    """coverage.deductible_type — POS (Participação Obrigatória do Segurado) ≡ franquia.

    O padrão dominante em CG residencial é "valor ou percentual definido na apólice":
    a CG fixa a ESTRUTURA, o número vive na apólice do cliente → defined_in_policy.
    """

    none = "none"                       # sem POS (ex.: RC, perda de aluguel)
    percentage = "percentage"           # % concreto fixado na própria CG
    fixed_amount = "fixed_amount"        # valor concreto fixado na própria CG
    defined_in_policy = "defined_in_policy"  # valor OU % definido na apólice (dominante)


class ExtractedCoverage(BaseModel):
    """Uma cobertura da CG. Grão = (seguradora × cobertura)."""

    coverage_name: str = Field(
        description="Nome comercial da cobertura, verbatim da CG (ex.: 'Vendaval, furacão...'). "
        "Ruidoso e não canônico — a canonicalização vive em `perils`. NÃO inclua o plano no nome."
    )
    plan: str | None = Field(
        default=None,
        description="Plano/tier comercial ao qual esta cobertura pertence (ex.: 'Essencial', "
        "'Fácil'). Texto livre — não canônico, varia por seguradora. Se a cobertura é IDÊNTICA "
        "em mais de um plano, combine num rótulo só (ex.: 'Essencial/Fácil') em vez de duplicar "
        "a linha. None se a CG não separa coberturas em planos.",
    )
    kind: Kind = Field(description="Cobertura básica ou adicional, conforme a CG.")
    deductible_type: DeductibleType | None = Field(
        default=None,
        description="Estrutura da POS/franquia desta cobertura. None se a CG não menciona.",
    )
    deductible_rule_text: str | None = Field(
        default=None,
        description="Texto CRU da regra de POS/franquia, verbatim. Alimenta o RAG. "
        "None se não houver regra.",
    )
    perils: list[Peril] = Field(
        description="Perigos canônicos que ESTA cobertura protege. Mapeie o nome comercial "
        "para um ou mais perigos da lista fixa. Vazio se nenhum se aplica."
    )
    exclusions: list[str] = Field(
        default_factory=list,
        description="Cláusulas de exclusão específicas DESTA cobertura, verbatim. "
        "Não inclua aqui exclusões gerais do documento.",
    )


class ExtractedDocument(BaseModel):
    """Uma CG inteira → coberturas + exclusões gerais.

    Campos de proveniência (pdf_url, pdf_hash) NÃO entram aqui: vêm do manifesto na
    persistência, não da leitura do LLM. insurer/product/susep_process/version são
    extraídos da CG e depois reconciliados com o manifesto (sinal barato de eval).
    """

    insurer: str = Field(description="Nome da seguradora, conforme a CG.")
    product: str = Field(description="Nome do produto/plano, conforme a CG.")
    susep_process: str = Field(
        description="Número do processo SUSEP (formato 15414.NNNNNN/AAAA-DD)."
    )
    version: str | None = Field(
        default=None, description="Versão/vigência da CG (ex.: data de vigência)."
    )
    property_type: str | None = Field(
        default=None,
        description="Tipo(s) de imóvel coberto (ex.: 'habitual', 'veraneio'). None se não claro.",
    )
    coverages: list[ExtractedCoverage] = Field(
        description="Todas as coberturas da CG, básicas e adicionais."
    )
    general_exclusions: list[str] = Field(
        default_factory=list,
        description="Exclusões GERAIS (valem pro documento todo, não amarradas a uma "
        "cobertura), verbatim.",
    )
