"""Formato do texto do chunk. Puro: não toca no banco, não chama modelo.

O que estes testes travam é o *contrato de conteúdo* do vetor — qual informação o
cabeçalho carrega (escopo, cobertura: significado) e qual ele não carrega
(seguradora, produto, processo: identidade, que é filtro SQL).
"""
import pytest

from app.rag.chunking import build_deductible_text, build_exclusion_text


def test_exclusao_de_cobertura_leva_o_nome_da_cobertura():
    assert (
        build_exclusion_text("Danos por ato doloso.", "coverage", "Incêndio")
        == "Exclusão da cobertura Incêndio: Danos por ato doloso."
    )


def test_exclusao_geral_diz_que_vale_pra_apolice_inteira():
    """O mesmo texto de cláusula significa outra coisa em escopo geral — o cabeçalho
    é o que o vetor tem pra distinguir os dois casos."""
    assert (
        build_exclusion_text("Danos por ato doloso.", "general", None)
        == "Exclusão geral da apólice: Danos por ato doloso."
    )


def test_regra_de_franquia_leva_a_cobertura():
    assert (
        build_deductible_text("POS de 10% dos prejuízos.", "Vendaval")
        == "Regra de franquia da cobertura Vendaval: POS de 10% dos prejuízos."
    )


def test_exclusao_de_cobertura_sem_nome_falha_alto():
    """O check do banco garante o arco, mas não o texto: sem este guard o chunk sairia
    como "Exclusão da cobertura None" — um texto indexado que mente sobre a própria
    origem, e mentira em índice não quebra nada, só responde errado depois."""
    with pytest.raises(ValueError, match="coverage_name"):
        build_exclusion_text("Danos por ato doloso.", "coverage", None)


def test_escopo_desconhecido_falha_alto():
    with pytest.raises(ValueError, match="escopo desconhecido"):
        build_exclusion_text("Danos por ato doloso.", "outro", None)


def test_identidade_nao_entra_no_texto():
    """Seguradora/produto/processo ficam FORA do vetor de propósito: são filtro SQL
    (`WHERE document_id ...`), exato e de graça. Dentro do texto só empurrariam todos
    os chunks da mesma seguradora pra perto uns dos outros."""
    frase = build_exclusion_text("Danos por ato doloso.", "coverage", "Incêndio")
    assert "Porto Seguro" not in frase
    assert "15414" not in frase
    # o cabeçalho é curto: com a cláusula mediana em 101 chars, prefixo longo dominaria
    # o vetor e a busca passaria a ordenar por cabeçalho.
    prefixo = frase.split(":")[0]
    assert len(prefixo) < 60
