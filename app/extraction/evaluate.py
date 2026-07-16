"""Eval leve de extração: compara a saída de um modelo contra um golden.

O que é o "golden": a extração do Sonnet que revisamos à mão contra a CG real
(data/golden/482868.json). NÃO é verdade absoluta — é uma referência conferida.
A pergunta que a eval responde é a que importa pra decisão: *um modelo mais barato
entrega a mesma estrutura que a saída que validamos?*

Comparamos o que alimenta o SQL worker (perigos, kind, POS), não o nome comercial —
duas extrações podem escrever o nome diferente e ainda serem equivalentes.
"""

import unicodedata
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .schema import ExtractedCoverage, ExtractedDocument

_MATCH_THRESHOLD = 0.6  # similaridade mínima de nome pra considerar a mesma cobertura


def _norm(s: str) -> str:
    """Tira acento, caixa e espaço extra — ruído de nome não deve virar divergência."""
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.split())


def _similar(a: str, b: str) -> float:
    return SequenceMatcher(None, _norm(a), _norm(b)).ratio()


@dataclass
class EvalReport:
    model: str
    n_golden: int
    n_candidate: int
    matched: list[tuple[str, str]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)   # está no golden, faltou no candidato
    extra: list[str] = field(default_factory=list)     # candidato inventou/achou a mais
    kind_agree: int = 0
    deductible_agree: int = 0
    perils_agree: int = 0
    perils_golden: set[str] = field(default_factory=set)
    perils_candidate: set[str] = field(default_factory=set)
    n_general_excl: tuple[int, int] = (0, 0)
    n_coverage_excl: tuple[int, int] = (0, 0)

    @property
    def recall(self) -> float:
        """Fração das coberturas do golden que o candidato achou."""
        return len(self.matched) / self.n_golden if self.n_golden else 0.0

    @property
    def structure_score(self) -> float:
        """Das coberturas casadas, quantas bateram em kind + POS + perigos."""
        n = len(self.matched)
        if not n:
            return 0.0
        return (self.kind_agree + self.deductible_agree + self.perils_agree) / (3 * n)


def _match(golden: list[ExtractedCoverage], cand: list[ExtractedCoverage]):
    """Casa coberturas por similaridade de nome (guloso: melhor par primeiro)."""
    pairs = sorted(
        (
            (_similar(g.coverage_name, c.coverage_name), gi, ci)
            for gi, g in enumerate(golden)
            for ci, c in enumerate(cand)
        ),
        reverse=True,
    )
    used_g: set[int] = set()
    used_c: set[int] = set()
    out = []
    for score, gi, ci in pairs:
        if score < _MATCH_THRESHOLD or gi in used_g or ci in used_c:
            continue
        used_g.add(gi)
        used_c.add(ci)
        out.append((gi, ci))
    return out, used_g, used_c


def evaluate(golden: ExtractedDocument, candidate: ExtractedDocument, model: str) -> EvalReport:
    rep = EvalReport(
        model=model,
        n_golden=len(golden.coverages),
        n_candidate=len(candidate.coverages),
    )
    pairs, used_g, used_c = _match(golden.coverages, candidate.coverages)

    for gi, ci in pairs:
        g, c = golden.coverages[gi], candidate.coverages[ci]
        rep.matched.append((g.coverage_name, c.coverage_name))
        if g.kind == c.kind:
            rep.kind_agree += 1
        if g.deductible_type == c.deductible_type:
            rep.deductible_agree += 1
        if {p.value for p in g.perils} == {p.value for p in c.perils}:
            rep.perils_agree += 1

    rep.missing = [g.coverage_name for i, g in enumerate(golden.coverages) if i not in used_g]
    rep.extra = [c.coverage_name for i, c in enumerate(candidate.coverages) if i not in used_c]

    rep.perils_golden = {p.value for cov in golden.coverages for p in cov.perils}
    rep.perils_candidate = {p.value for cov in candidate.coverages for p in cov.perils}

    rep.n_general_excl = (len(golden.general_exclusions), len(candidate.general_exclusions))
    rep.n_coverage_excl = (
        sum(len(c.exclusions) for c in golden.coverages),
        sum(len(c.exclusions) for c in candidate.coverages),
    )
    return rep
