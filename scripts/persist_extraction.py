"""Fatia 2 — persiste um JSON extraído nas 5 tabelas (sem re-chamar o LLM).

Fluxo: carrega o JSON salvo pelo run_extraction --out → valida no Pydantic → acha a
proveniência no manifesto (via o PDF de origem) → INSERT nas 5 tabelas → confere as linhas.

Uso:
    # 1) extrai uma vez e salva o JSON
    python scripts/run_extraction.py data/corpus/susep_482868.pdf --out data/extractions/482868.json
    # 2) persiste (quantas vezes precisar, sem gastar chamada LLM)
    python scripts/persist_extraction.py data/extractions/482868.json --pdf data/corpus/susep_482868.pdf
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import func, select  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.extraction.manifest import manifest_row_for_pdf  # noqa: E402
from app.extraction.persist import document_exists, persist_document  # noqa: E402
from app.extraction.schema import ExtractedDocument  # noqa: E402
from app.models import Coverage, CoveragePeril, Exclusion, Peril, PolicyDocument  # noqa: E402


async def run(json_path: Path, pdf_path: Path) -> None:
    doc = ExtractedDocument.model_validate_json(json_path.read_text())
    row = manifest_row_for_pdf(pdf_path)
    if row is None:
        sys.exit(f"[erro] sem linha no manifesto pra {pdf_path.name}")

    async with SessionLocal() as session:
        if await document_exists(session, row["processo"], doc.version):
            print(f"[skip] {row['processo']} v{doc.version} já persistido — idempotência.")
            return

        pd_id = await persist_document(session, doc, row)
        print(f"[ok] policy_document.id = {pd_id}")

        # Verificação: conta o que entrou pra este documento.
        n_cov = await session.scalar(
            select(func.count()).select_from(Coverage).where(Coverage.document_id == pd_id)
        )
        n_exc = await session.scalar(
            select(func.count()).select_from(Exclusion).where(Exclusion.document_id == pd_id)
        )
        n_link = await session.scalar(
            select(func.count())
            .select_from(CoveragePeril)
            .join(Coverage, Coverage.id == CoveragePeril.coverage_id)
            .where(Coverage.document_id == pd_id)
        )
        n_peril = await session.scalar(select(func.count()).select_from(Peril))
        print(f"[check] coverages={n_cov}  coverage_peril={n_link}  "
              f"exclusions={n_exc}  perigos_distintos_no_banco={n_peril} (de 12 no vocabulário)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", help="JSON extraído (saída de run_extraction --out)")
    ap.add_argument("--pdf", required=True, help="PDF de origem (pra achar a proveniência no manifesto)")
    args = ap.parse_args()
    asyncio.run(run(Path(args.json), Path(args.pdf)))


if __name__ == "__main__":
    main()
