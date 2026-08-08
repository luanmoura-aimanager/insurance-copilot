"""Monta o texto de cada chunk. Funções PURAS: sem banco, sem LLM, sem I/O.

**Não há splitter aqui, e isso é uma decisão medida, não uma pendência.** No corpus
persistido (30 documentos) as 4.176 exclusões têm comprimento mediano de 101
caracteres, p99 de 456 e máximo de 907; as 210 regras de franquia têm média de 180.
Nada chega perto de 1.400 caracteres — bem abaixo do limite de qualquer modelo de
embedding. Então **uma linha de origem = um chunk**, e `chunk_index` é sempre 0.
Cortar por tamanho e sobrepor pedaços seria complexidade sem nenhum texto para
exercitá-la; quando (e se) o corpus trouxer cláusulas longas, o `chunk_index` já está
no schema esperando.

**O que entra no texto do chunk: significado, não identidade.** O cabeçalho existe
porque `"Danos causados por ato doloso"` isolado não diz *de que* isso é exclusão —
uma exclusão da cobertura de Incêndio e uma exclusão geral da apólice são fatos
diferentes, e o vetor precisa saber disso. Já a seguradora, o produto e o processo
SUSEP **não** entram: isso é identidade, ela já está em `policy_document` e é
respondida por filtro SQL (`WHERE document_id ...`) na busca, com precisão exata e de
graça. Enfiar "Porto Seguro" em todo chunk só aproximaria todos os chunks da mesma
seguradora entre si — ruído que empurra a similaridade para longe do que a pergunta
realmente quer.

O cabeçalho é curto pelo mesmo motivo: com a cláusula mediana em 101 caracteres, um
prefixo longo dominaria o vetor e a busca passaria a ordenar por cabeçalho.
"""

SCOPE_GENERAL = "general"
SCOPE_COVERAGE = "coverage"


def build_exclusion_text(clause_text: str, scope: str, coverage_name: str | None) -> str:
    """Texto do chunk de uma exclusão (`exclusion.clause_text`).

    O escopo muda o significado da mesma frase: uma exclusão de cobertura vale só
    dentro daquela cobertura, uma geral vale para a apólice inteira.

    `coverage_name` é obrigatório quando `scope='coverage'`. O check
    `ck_exclusion_scope_coverage_id` já garante isso no banco, mas aqui a falha é
    barulhenta de propósito: sem ele o texto sairia como "Exclusão da cobertura None",
    um chunk que *mente* sobre a própria origem — e mentira em texto indexado não
    quebra nada, só devolve resposta errada meses depois.
    """
    if scope == SCOPE_COVERAGE:
        if not coverage_name:
            raise ValueError(
                "exclusão de escopo 'coverage' exige coverage_name — sem ele o chunk "
                "diria 'Exclusão da cobertura None'"
            )
        return f"Exclusão da cobertura {coverage_name}: {clause_text}"
    if scope == SCOPE_GENERAL:
        return f"Exclusão geral da apólice: {clause_text}"
    raise ValueError(f"escopo desconhecido: {scope!r}")


def build_deductible_text(rule_text: str, coverage_name: str) -> str:
    """Texto do chunk de uma regra de franquia (`coverage.deductible_rule_text`).

    A regra de POS é sempre *de uma cobertura* — o nome não é enfeite, é o que separa
    "10% dos prejuízos" da cobertura de Vendaval da mesma frase em Danos Elétricos.
    """
    if not coverage_name:
        raise ValueError("regra de franquia exige coverage_name")
    return f"Regra de franquia da cobertura {coverage_name}: {rule_text}"
