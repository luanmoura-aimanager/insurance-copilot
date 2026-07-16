"""Testa a lógica da eval — sem chamar LLM.

A eval é quem decide onde gastar o orçamento de API; se ela mente, a decisão é errada.
Por isso ela mesma é testada: um golden contra si é 100%, uma saída degradada tem que
acusar exatamente o que degradou, e ruído de nome NÃO pode virar divergência.
"""

import json
from pathlib import Path

from app.extraction.evaluate import evaluate
from app.extraction.schema import (
    DeductibleType,
    ExtractedCoverage,
    ExtractedDocument,
    Kind,
    Peril,
)

GOLDEN_PATH = Path(__file__).resolve().parent.parent / "data" / "golden" / "482868.json"


def doc() -> ExtractedDocument:
    return ExtractedDocument(
        insurer="X",
        product="Y",
        susep_process="15414.1/2004-31",
        coverages=[
            ExtractedCoverage(
                coverage_name="Vendaval, furacão e granizo",
                kind=Kind.additional,
                deductible_type=DeductibleType.defined_in_policy,
                perils=[Peril.vendaval, Peril.granizo],
                exclusions=["a"],
            ),
            ExtractedCoverage(
                coverage_name="Incêndio",
                kind=Kind.basic,
                deductible_type=DeductibleType.defined_in_policy,
                perils=[Peril.incendio_explosao],
                exclusions=[],
            ),
        ],
        general_exclusions=["g1", "g2"],
    )


def test_identical_scores_perfect():
    rep = evaluate(doc(), doc(), "m")
    assert rep.recall == 1.0
    assert rep.structure_score == 1.0
    assert rep.missing == [] and rep.extra == []


def test_name_noise_is_not_a_divergence():
    """Acento/caixa/ordem diferentes no nome comercial não podem contar como erro."""
    other = doc()
    other.coverages[0].coverage_name = "VENDAVAL, FURACAO E GRANIZO"
    rep = evaluate(doc(), other, "m")
    assert rep.recall == 1.0
    assert rep.structure_score == 1.0


def test_missing_coverage_is_caught():
    worse = doc()
    worse.coverages = worse.coverages[:1]
    rep = evaluate(doc(), worse, "m")
    assert rep.recall == 0.5
    assert rep.missing == ["Incêndio"]


def test_wrong_fields_are_caught():
    worse = doc()
    worse.coverages[0].kind = Kind.basic                       # errou kind
    worse.coverages[1].deductible_type = DeductibleType.none   # errou POS
    worse.coverages[0].perils = [Peril.vendaval]               # perdeu granizo
    rep = evaluate(doc(), worse, "m")
    assert rep.recall == 1.0            # achou as duas coberturas
    assert rep.kind_agree == 1
    assert rep.deductible_agree == 1
    assert rep.perils_agree == 1
    assert rep.structure_score == 0.5   # 3 acertos de 6 checagens
    assert "granizo" in rep.perils_golden - rep.perils_candidate


def test_extra_coverage_is_reported():
    more = doc()
    more.coverages.append(
        ExtractedCoverage(
            coverage_name="Roubo de Celular",
            kind=Kind.additional,
            perils=[Peril.roubo_furto_qualificado],
        )
    )
    rep = evaluate(doc(), more, "m")
    assert rep.extra == ["Roubo de Celular"]


def test_golden_file_is_valid():
    """Guarda-corpo: o golden precisa continuar parseável no schema atual."""
    golden = ExtractedDocument.model_validate_json(GOLDEN_PATH.read_text())
    assert len(golden.coverages) == 8
    rep = evaluate(golden, golden, "self")
    assert rep.structure_score == 1.0
