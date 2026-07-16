"""Reprocessa os resultados de um batch JÁ PAGO — sem gastar nada.

Propriedade da Batch API que vale conhecer: os resultados ficam disponíveis por ~29 dias.
Ou seja, uma falha de PARSE não obriga a re-extrair — a resposta ainda está lá pra ser
buscada de novo. Isso separa "a chamada falhou" (custa dinheiro refazer) de "o nosso
código falhou em ler a resposta" (custa zero refazer).

NÃO grava cost_event: o custo dessas chamadas já foi registrado quando o batch rodou.
Gravar de novo seria contar o mesmo dinheiro duas vezes.

Uso:
    python scripts/rescue_batch.py msgbatch_01G9Vz2Bwd1sRMvGQryyYXUY
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import SessionLocal  # noqa: E402
from app.extraction import batch as batch_api  # noqa: E402
from app.extraction.extract import MODEL, ExtractionFailed, parse_response  # noqa: E402
from app.extraction.manifest import load_manifest_docs  # noqa: E402
from app.extraction.persist import document_exists_by_hash, persist_document  # noqa: E402

DUMPS = ROOT / "data" / "extractions"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("batch_id")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    by_custom_id = {f"susep_{d['id']}": d for d in load_manifest_docs()}
    DUMPS.mkdir(parents=True, exist_ok=True)

    async with SessionLocal() as session:
        n_ok = n_skip = n_fail = 0
        for outcome in batch_api.results(args.batch_id):
            row = by_custom_id.get(outcome.custom_id)
            if row is None:
                print(f"  [{outcome.custom_id}] sem linha no manifesto — pulando")
                continue

            if outcome.status != "succeeded" or outcome.message is None:
                print(f"  [{outcome.custom_id}] a CHAMADA falhou ({outcome.status}) — "
                      "esta o batch teria que refazer, custa dinheiro")
                n_fail += 1
                continue

            if await document_exists_by_hash(session, row["sha256"]):
                n_skip += 1
                continue

            msg = outcome.message
            try:
                doc = parse_response(
                    msg.content, args.model, msg.usage.input_tokens, msg.usage.output_tokens
                )
            except ExtractionFailed as exc:
                if exc.raw is not None:
                    dump = DUMPS / f"{outcome.custom_id}.RAW.json"
                    dump.write_text(json.dumps(exc.raw, ensure_ascii=False, indent=2))
                    print(f"  [{outcome.custom_id}] parse falhou — cru em {dump}")
                    print(f"      chaves no topo: {list(exc.raw)[:6]}")
                print(f"  [{outcome.custom_id}] {str(exc)[:160]}")
                n_fail += 1
                continue

            pd_id = await persist_document(session, doc, row)
            await session.commit()
            n_ok += 1
            print(f"  [{outcome.custom_id}] resgatado  doc_id={pd_id}  "
                  f"coberturas={len(doc.coverages)}")

        print(f"\n== resgatados: {n_ok}  |  já estavam: {n_skip}  |  ainda falhando: {n_fail}")
        print("== custo desta operação: US$ 0 (resultados já pagos)")


if __name__ == "__main__":
    asyncio.run(main())
