"""F3b — extrai uma amostra do corpus via Batch API e popula as tabelas.

Dry-run por padrão: mostra a amostra escolhida e o custo projetado, e NÃO gasta nada.
Só com --yes ele submete. Com orçamento apertado, gastar tem que ser um ato deliberado.

Uso:
    python scripts/run_batch.py --limit 30              # dry-run: mostra e projeta
    python scripts/run_batch.py --limit 30 --yes        # roda de verdade
"""

import argparse
import asyncio
import json
import sys
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cost import cost_usd, record_cost_event  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.extraction import batch as batch_api  # noqa: E402
from app.extraction.extract import MODEL, ExtractionFailed, parse_response  # noqa: E402
from app.extraction.manifest import load_manifest_docs  # noqa: E402
from app.extraction.pdf import pdf_to_text  # noqa: E402
from app.extraction.persist import (  # noqa: E402
    delete_document_by_hash,
    document_exists_by_hash,
    persist_document,
)
from app.extraction.sample import SEED, eligible_docs, pick_sample  # noqa: E402

CORPUS = ROOT / "data" / "corpus"
# O doc do golden (Porto, lido à mão e revisado). Fixado na amostra pra que a eval de
# qualidade rode sobre a saída do batch sem custar uma chamada extra.
GOLDEN_ID = 482868


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--yes", action="store_true", help="submete de verdade (gasta dinheiro)")
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--seed", type=int, default=SEED, help="semente da amostra (reproduzível)")
    ap.add_argument("--only", help="ids específicos, separados por vírgula (ignora a amostra)")
    ap.add_argument("--force", action="store_true",
                    help="re-extrai mesmo já estando no banco (apaga o antigo antes)")
    args = ap.parse_args()

    docs = load_manifest_docs()
    n_eligible = len(eligible_docs(docs))

    if args.only:
        wanted = {int(x) for x in args.only.split(",")}
        sample = [d for d in docs if d["id"] in wanted]
        missing = wanted - {d["id"] for d in sample}
        if missing:
            sys.exit(f"[erro] ids não encontrados no manifesto: {sorted(missing)}")
    else:
        # GOLDEN_ID entra fixo: com ele no batch, a eval de qualidade roda de graça depois.
        sample = pick_sample(docs, args.limit, CORPUS, seed=args.seed, pin_ids=(GOLDEN_ID,))

    async with SessionLocal() as session:
        if args.force:
            # NÃO apaga nada aqui: o dry-run tem que ser livre de efeito colateral.
            # A remoção acontece na hora de persistir, junto do INSERT do substituto —
            # assim o dado antigo só morre quando existe um novo pra pôr no lugar.
            fresh = sample
        else:
            # Guard PRÉ-chamada: pula o que já está no banco (por sha256) ANTES de pagar.
            fresh = [d for d in sample if not await document_exists_by_hash(session, d["sha256"])]
        n_skipped = len(sample) - len(fresh)

        print(f"\nCorpus elegível (residencial, com texto, vigente): {n_eligible} docs")
        print(f"Amostra escolhida: {len(sample)}  |  já no banco (pulados): {n_skipped}  "
              f"|  a extrair: {len(fresh)}")
        print(f"Seguradoras distintas na amostra: {len({d['seguradora'] for d in fresh})}\n")

        if not fresh:
            print("Nada a fazer.")
            return

        # Texto é grátis (pdfplumber) — extraímos antes pra projetar o custo com dado real.
        items: list[tuple[str, str]] = []
        est_in = 0
        for d in fresh:
            path = CORPUS / f"susep_{d['id']}.pdf"
            text = pdf_to_text(path)
            items.append((path.stem, text))
            est_in += len(text) // 4
            print(f"  {path.stem:16s} {d['seguradora'][:38]:40s} {len(text)//4:>7,} tok")

        est_out = 13_000 * len(items)  # medido no piloto
        projected = cost_usd(args.model, est_in, est_out, batch=True)
        per_doc = projected / len(items)
        print(f"\nProjeção ({args.model}, batch -50%):")
        print(f"  entrada ~{est_in:,} tok  |  saída ~{est_out:,} tok")
        print(f"  custo projetado: US$ {projected:.2f}  (~US$ {per_doc:.4f}/doc)")
        print(f"  extrapolando pro corpus elegível ({n_eligible} docs): "
              f"US$ {per_doc * n_eligible:.2f}")

        if not args.yes:
            print("\n[dry-run] Nada foi gasto. Rode de novo com --yes pra submeter.")
            return

        print(f"\n[batch] submetendo {len(items)} requisições ...", file=sys.stderr)
        batch_id = batch_api.submit(items, model=args.model)
        print(f"[batch] id={batch_id}", file=sys.stderr)
        batch_api.wait(batch_id, poll_seconds=args.poll)

        by_id = {f"susep_{d['id']}": d for d in fresh}
        total = Decimal("0")
        n_ok = n_fail = 0

        for outcome in batch_api.results(batch_id):
            row = by_id.get(outcome.custom_id)
            if outcome.status != "succeeded" or outcome.message is None:
                print(f"  [{outcome.custom_id}] {outcome.status}: {outcome.error}")
                n_fail += 1
                continue

            msg = outcome.message
            in_tok, out_tok = msg.usage.input_tokens, msg.usage.output_tokens

            # A chamada foi cobrada — grava o custo ANTES de tentar parsear.
            event = await record_cost_event(
                session,
                agent_name="extraction",
                model=args.model,
                input_tokens=in_tok,
                output_tokens=out_tok,
                label=outcome.custom_id,
                batch=True,
            )
            total += event.cost_usd
            await session.commit()

            try:
                doc = parse_response(msg.content, args.model, in_tok, out_tok)
            except ExtractionFailed as exc:
                # Salva o cru: a chamada já foi paga, debugar não pode custar de novo.
                if exc.raw is not None:
                    dump = Path("data/extractions") / f"{outcome.custom_id}.RAW.json"
                    dump.parent.mkdir(parents=True, exist_ok=True)
                    dump.write_text(json.dumps(exc.raw, ensure_ascii=False, indent=2))
                    print(f"  [{outcome.custom_id}] parse falhou — cru salvo em {dump}")
                print(f"  [{outcome.custom_id}] parse falhou: {str(exc)[:120]}")
                n_fail += 1
                continue

            # Substituição atômica: apaga o antigo e grava o novo na MESMA transação.
            # Se o INSERT falhar, o rollback devolve o dado antigo — nunca fica o buraco.
            #
            # E um erro inesperado no persist (constraint, etc.) NÃO pode derrubar o loop:
            # o custo deste doc já foi commitado acima, e os docs seguintes — também já
            # cobrados pelo batch — ainda precisam registrar o custo deles. Abortar aqui
            # deixaria essas chamadas pagas sem linha em cost_event.
            try:
                if args.force:
                    old_id = await delete_document_by_hash(session, row["sha256"])
                    if old_id:
                        print(f"  [force] substituindo doc_id={old_id}")
                pd_id = await persist_document(session, doc, row)
                await session.commit()
            except Exception as exc:
                await session.rollback()  # descarta delete+persist parciais; o cost_event já está gravado
                print(f"  [{outcome.custom_id}] persist falhou (custo já registrado): {str(exc)[:120]}")
                n_fail += 1
                continue
            n_ok += 1
            print(f"  [{outcome.custom_id}] ok  doc_id={pd_id}  "
                  f"coberturas={len(doc.coverages)}  US$ {event.cost_usd}")

        print(f"\n== persistidos: {n_ok}  |  falhas: {n_fail}")
        print(f"== custo REAL do batch: US$ {total}")
        if n_ok:
            real_per_doc = total / n_ok
            print(f"== medido: US$ {real_per_doc:.4f}/doc  ->  corpus elegível "
                  f"({n_eligible} docs) custaria US$ {real_per_doc * n_eligible:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
