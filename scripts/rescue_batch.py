"""Reprocessa os resultados de um batch JÁ PAGO — sem gastar nada.

Propriedade da Batch API que vale conhecer: os resultados ficam disponíveis por ~29 dias.
Ou seja, uma falha de PARSE não obriga a re-extrair — a resposta ainda está lá pra ser
buscada de novo. Isso separa "a chamada falhou" (custa dinheiro refazer) de "o nosso
código falhou em ler a resposta" (custa zero refazer).

Custo: reconcilia por label. O normal é o custo já ter sido gravado quando o batch
rodou, então o resgate não grava nada. MAS se o run original morreu antes de registrar
(ex.: TimeoutError no polling), o doc teria sido resgatado sem NENHUMA linha de custo —
uma chamada paga e invisível. Então o resgate grava a linha SÓ quando ela falta, sem
nunca contar o mesmo dinheiro duas vezes (ver reconcile_cost_event).

Uso:
    python scripts/rescue_batch.py msgbatch_01G9Vz2Bwd1sRMvGQryyYXUY
"""

import argparse
import asyncio
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cost import reconcile_cost_event  # noqa: E402
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
        n_ok = n_skip = n_fail = n_reconciled = 0
        reconciled_usd = Decimal("0")
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

            msg = outcome.message

            # Reconcilia o custo ANTES de qualquer skip: o caso perigoso é justamente o doc
            # que já está no banco (o run persistiu e morreu antes de gravar o custo). Grava
            # a linha só se faltar — nunca conta o mesmo dinheiro duas vezes.
            event = await reconcile_cost_event(
                session,
                agent_name="extraction",
                model=args.model,
                input_tokens=msg.usage.input_tokens,
                output_tokens=msg.usage.output_tokens,
                label=outcome.custom_id,
                batch=True,
            )
            if event is not None:
                n_reconciled += 1
                reconciled_usd += event.cost_usd
                await session.commit()
                print(f"  [{outcome.custom_id}] custo faltante gravado: US$ {event.cost_usd}")

            if await document_exists_by_hash(session, row["sha256"]):
                n_skip += 1
                continue

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
        if n_reconciled:
            print(f"== custo faltante reconciliado: {n_reconciled} linha(s), US$ {reconciled_usd} "
                  "(chamadas pagas que o run original não chegou a registrar)")
        else:
            print("== custo desta operação: US$ 0 (todos os custos já estavam registrados)")


if __name__ == "__main__":
    asyncio.run(main())
