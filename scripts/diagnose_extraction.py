"""Diagnóstico de extração incompleta — custo ZERO (lê banco + PDFs, sem LLM).

HISTÓRICO DESTE ARQUIVO (vale mais que o código):

A v1 media "menções a 'Condições Especiais'" e marcava como suspeito quem tinha muitas
menções e poucas coberturas. Marcou CHUBB e SIMPLE2U. Estava ERRADA: "Condições Especiais"
é um termo definido no GLOSSÁRIO de praticamente toda CG — a métrica contava boilerplate.
A própria tabela desmentia (ZURICH: 0 menções, 17 coberturas; COBUCCIO: 92 menções, 17
coberturas — correlação zero), mas o flag parecia rigoroso e ninguém olhou.

A v2 mede o sinal certo: quanto o texto-fonte FALA de cobertura (termos de perigo) versus
quantas coberturas saíram. Nos dados reais isso correlaciona forte:

    CHUBB 2.6 termos/10k -> 1 cob      ALIANÇA BAHIA 10.3 -> 35 cob
    SIMPLE2U 3.6         -> 2 cob      HDI 11.7           -> 33 cob

Conclusão que a v2 permitiu e a v1 impedia: CHUBB/SIMPLE2U não são extração quebrada, são
documentos esparsos. A extração é fiel ao input.

Uma falha REAL de extração seria o descolamento: texto cheio de termos de cobertura e
quase nada extraído. É só isso que este script marca.
"""

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import func, select  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.extraction.manifest import load_manifest_docs  # noqa: E402
from app.extraction.pdf import pdf_to_text  # noqa: E402
from app.models import Coverage, PolicyDocument  # noqa: E402

CORPUS = ROOT / "data" / "corpus"

# Vocabulário de superfície: como as CGs FALAM de cobertura. Não é o enum canônico —
# aqui queremos medir presença no texto cru, com as variações que aparecem de verdade.
TERMS = (
    "vendaval", "granizo", "roubo", "furto", "danos el", "quebra de vidro",
    "alagamento", "desmoronamento", "impacto de ve", "aluguel", "incend", "incênd",
    "responsabilidade civil", "fumaça", "explos",
)

DENSITY_ALTA = 8.0   # a partir daqui o doc claramente descreve muitas coberturas
COBERTURAS_POUCAS = 5


async def main() -> None:
    by_hash = {d["sha256"]: d for d in load_manifest_docs()}

    async with SessionLocal() as session:
        rows = (
            await session.execute(
                select(
                    PolicyDocument.insurer,
                    PolicyDocument.pdf_hash,
                    func.count(Coverage.id).label("n_cov"),
                )
                .outerjoin(Coverage, Coverage.document_id == PolicyDocument.id)
                .group_by(PolicyDocument.id)
            )
        ).all()

    report = []
    for insurer, pdf_hash, n_cov in rows:
        manifest_row = by_hash.get(pdf_hash)
        if not manifest_row:
            continue
        path = CORPUS / f"susep_{manifest_row['id']}.pdf"
        if not path.exists():
            continue
        low = pdf_to_text(path).lower()
        hits = sum(low.count(t) for t in TERMS)
        density = hits / max(len(low) / 10_000, 0.01)
        report.append((insurer, n_cov, hits, density))

    report.sort(key=lambda r: r[3])
    print(f"{'seguradora':<44} {'cob':>4} {'termos':>7} {'dens/10k':>9}")
    print("-" * 68)
    suspects = []
    for insurer, n_cov, hits, density in report:
        suspeito = density >= DENSITY_ALTA and n_cov <= COBERTURAS_POUCAS
        if suspeito:
            suspects.append(insurer)
        print(f"{insurer[:44]:<44} {n_cov:>4} {hits:>7} {density:>9.1f}"
              + ("  <-- SUSPEITO" if suspeito else ""))

    print(f"\n== {len(suspects)} de {len(report)} docs suspeitos")
    print("== critério: o texto fala MUITO de cobertura (densidade >= "
          f"{DENSITY_ALTA}/10k) mas saíram <= {COBERTURAS_POUCAS} coberturas.")
    print("== densidade baixa + poucas coberturas NÃO é suspeito: é documento esparso,")
    print("==   e a extração estar fiel a ele é o comportamento correto.")


if __name__ == "__main__":
    asyncio.run(main())
