"""Seleção da amostra do corpus pro batch.

Com orçamento apertado, QUAIS 30 docs importa tanto quanto quantos. Duas armadilhas
que esta seleção evita:

1. **Volume sem variedade.** Pegar os 30 primeiros do manifesto traria várias versões
   do mesmo produto da mesma seguradora, e o SQL não teria o que comparar. Regra: um
   doc por processo, e round-robin entre seguradoras.
2. **Viés alfabético.** Iterar as seguradoras em ordem alfabética e cortar em 30 dá
   "as 30 primeiras do alfabeto", não uma amostra — deixava Porto, Tokio, Zurich, HDI,
   Itaú e Mapfre de fora. Por isso a ordem é embaralhada com semente fixa: aleatória
   o bastante pra ser representativa, determinística o bastante pra ser reproduzível.
"""

from pathlib import Path
from random import Random

SEED = 42  # fixo: a mesma amostra hoje e daqui a 6 meses


def eligible_docs(docs: list[dict]) -> list[dict]:
    """Filtra o que dá pra extrair: residencial, com texto, versão vigente."""
    return [
        d
        for d in docs
        if d.get("parece_residencial")
        and d.get("tem_texto")
        and d.get("vigente")
        and d.get("arquivo")
    ]


def pick_sample(
    docs: list[dict],
    limit: int,
    corpus_dir: Path,
    seed: int = SEED,
    pin_ids: tuple[int, ...] = (),
) -> list[dict]:
    """Escolhe até `limit` docs maximizando variedade de seguradoras.

    `pin_ids` entra sempre (usamos pro doc do golden, pra que a eval de qualidade
    rode de graça sobre a saída do batch em vez de exigir uma chamada só pra ela).
    """
    pinned: list[dict] = []
    by_insurer: dict[str, list[dict]] = {}
    seen_processes: set[str] = set()

    for d in eligible_docs(docs):
        if not (corpus_dir / f"susep_{d['id']}.pdf").exists():
            continue
        if d["id"] in pin_ids:
            pinned.append(d)
            seen_processes.add(d.get("processo"))

    pinned_insurers = {d.get("seguradora") for d in pinned}

    for d in eligible_docs(docs):
        if d["id"] in pin_ids:
            continue
        processo = d.get("processo")
        if processo in seen_processes:  # 1 doc por produto, não várias versões
            continue
        if not (corpus_dir / f"susep_{d['id']}.pdf").exists():
            continue
        insurer = d.get("seguradora", "?")
        if insurer in pinned_insurers:  # essa seguradora já entrou via pin
            continue
        seen_processes.add(processo)
        by_insurer.setdefault(insurer, []).append(d)

    # sorted() primeiro pra ordem estável, DEPOIS embaralha com semente: reproduzível
    # sem ser alfabético.
    insurers = sorted(by_insurer)
    Random(seed).shuffle(insurers)

    out: list[dict] = list(pinned)[:limit]
    rounds = max((len(v) for v in by_insurer.values()), default=0)
    for i in range(rounds):
        for insurer in insurers:
            if len(out) >= limit:
                return out
            bucket = by_insurer[insurer]
            if i < len(bucket):
                out.append(bucket[i])
    return out[:limit]
