"""Harness de eval F4: dump do texto-fonte + coberturas persistidas por doc, custo ZERO (sem LLM).

Para cada doc no banco: escreve texto-fonte (cacheado) e coberturas persistidas
num diretório de scratchpad, pro juiz (Claude) ler um doc de cada vez.
Uso: python scripts/eval/dump_judge_bundles.py <out_dir>
"""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402
from sqlalchemy.orm import selectinload  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.extraction.manifest import load_manifest_docs  # noqa: E402
from app.extraction.pdf import pdf_to_text  # noqa: E402
from app.models import Coverage, CoveragePeril, Peril, PolicyDocument  # noqa: E402

CORPUS = ROOT / "data" / "corpus"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)


async def main() -> None:
    by_hash = {d["sha256"]: d for d in load_manifest_docs()}
    index = []

    async with SessionLocal() as session:
        docs = (await session.execute(select(PolicyDocument).order_by(PolicyDocument.id))).scalars().all()
        for d in docs:
            covs = (
                await session.execute(
                    select(Coverage).where(Coverage.document_id == d.id).order_by(Coverage.id)
                )
            ).scalars().all()
            # perils por cobertura
            cov_list = []
            for c in covs:
                peril_names = (
                    await session.execute(
                        select(Peril.name)
                        .join(CoveragePeril, CoveragePeril.peril_id == Peril.id)
                        .where(CoveragePeril.coverage_id == c.id)
                        .order_by(Peril.name)
                    )
                ).scalars().all()
                cov_list.append({
                    "coverage_name": c.coverage_name,
                    "plan": c.plan,
                    "kind": c.kind,
                    "deductible_type": c.deductible_type,
                    "perils": peril_names,
                })

            mrow = by_hash.get(d.pdf_hash)
            mid = mrow["id"] if mrow else None
            text = ""
            if mid is not None:
                pdf = CORPUS / f"susep_{mid}.pdf"
                if pdf.exists():
                    text = pdf_to_text(pdf)

            slug = f"{d.id:02d}_{mid}"
            (OUT / f"{slug}.txt").write_text(text)
            (OUT / f"{slug}.covs.json").write_text(
                json.dumps({
                    "doc_id": d.id,
                    "manifest_id": mid,
                    "insurer": d.insurer,
                    "product": d.product,
                    "susep_process": d.susep_process,
                    "n_coverages": len(cov_list),
                    "coverages": cov_list,
                }, ensure_ascii=False, indent=2)
            )
            index.append({
                "slug": slug, "doc_id": d.id, "manifest_id": mid,
                "insurer": d.insurer, "n_cov": len(cov_list),
                "text_chars": len(text),
            })

    (OUT / "_index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
    for r in index:
        print(f"{r['slug']:>12}  {r['insurer'][:36]:<36} covs={r['n_cov']:>3}  chars={r['text_chars']:>7}")


if __name__ == "__main__":
    asyncio.run(main())
