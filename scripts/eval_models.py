"""F3b — roda N modelos sobre a MESMA CG e compara com o golden.

Decide com dado (não com chute) qual modelo roda o batch: mostra, lado a lado, o que
cada modelo acha e o que cada um custa. Cada chamada vira uma linha em cost_event —
a fundação da F3a servindo pro primeiro uso real.

Uso:
    python scripts/eval_models.py data/corpus/susep_482868.pdf \
        --golden data/golden/482868.json \
        --models claude-sonnet-5,claude-haiku-4-5-20251001
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.cost import cost_usd, record_cost_event  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.extraction.evaluate import EvalReport, evaluate  # noqa: E402
from app.extraction.extract import ExtractionFailed, extract_document  # noqa: E402
from app.extraction.pdf import pdf_to_text  # noqa: E402
from app.extraction.schema import ExtractedDocument  # noqa: E402


def render(rep: EvalReport, in_tok: int, out_tok: int) -> None:
    normal = cost_usd(rep.model, in_tok, out_tok)
    batched = cost_usd(rep.model, in_tok, out_tok, batch=True)
    print(f"\n=== {rep.model} ===")
    print(f"  tokens      in={in_tok:,}  out={out_tok:,}")
    print(f"  custo       US$ {normal} normal  |  US$ {batched} em batch")
    print(f"  coberturas  golden={rep.n_golden}  modelo={rep.n_candidate}  "
          f"casadas={len(rep.matched)}  recall={rep.recall:.0%}")
    if rep.missing:
        print(f"    faltando: {', '.join(m[:40] for m in rep.missing)}")
    if rep.extra:
        print(f"    a mais:   {', '.join(e[:40] for e in rep.extra)}")
    n = len(rep.matched) or 1
    print(f"  concordância (nas casadas):  kind {rep.kind_agree}/{len(rep.matched)}  ·  "
          f"POS {rep.deductible_agree}/{len(rep.matched)}  ·  perigos {rep.perils_agree}/{len(rep.matched)}")
    print(f"  score de estrutura: {rep.structure_score:.0%}")
    only_g = rep.perils_golden - rep.perils_candidate
    only_c = rep.perils_candidate - rep.perils_golden
    print(f"  perigos (doc)  golden={len(rep.perils_golden)}  modelo={len(rep.perils_candidate)}"
          + (f"  | não achou: {sorted(only_g)}" if only_g else "")
          + (f"  | a mais: {sorted(only_c)}" if only_c else ""))
    print(f"  exclusões   gerais {rep.n_general_excl[0]}→{rep.n_general_excl[1]}  ·  "
          f"por cobertura {rep.n_coverage_excl[0]}→{rep.n_coverage_excl[1]}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--golden", required=True)
    ap.add_argument("--models", required=True, help="lista separada por vírgula")
    ap.add_argument("--out-dir", default="data/extractions")
    args = ap.parse_args()

    golden = ExtractedDocument.model_validate_json(Path(args.golden).read_text())
    text = pdf_to_text(args.pdf)
    label = Path(args.pdf).stem
    print(f"[eval] {label}: {len(text):,} chars | golden com {len(golden.coverages)} coberturas",
          file=sys.stderr)

    async with SessionLocal() as session:
        for model in [m.strip() for m in args.models.split(",") if m.strip()]:
            print(f"\n[eval] rodando {model} ...", file=sys.stderr)
            try:
                result = extract_document(text, model=model)
            except ExtractionFailed as exc:
                # A chamada foi COBRADA mesmo falhando — grava o custo antes de tudo.
                await record_cost_event(
                    session,
                    agent_name="extraction-eval",
                    model=exc.model,
                    input_tokens=exc.input_tokens,
                    output_tokens=exc.output_tokens,
                    label=f"{label} (FALHOU)",
                )
                await session.commit()
                if exc.raw is not None:  # salva o cru: debugar sem re-pagar a chamada
                    dump = Path(args.out_dir) / f"{label}.{model}.RAW.json"
                    dump.write_text(json.dumps(exc.raw, ensure_ascii=False, indent=2))
                    print(f"  [falhou] payload cru salvo em {dump}", file=sys.stderr)
                print(f"  [falhou] {exc}"[:300], file=sys.stderr)
                continue  # segue pro próximo modelo em vez de derrubar a eval inteira

            await record_cost_event(
                session,
                agent_name="extraction-eval",
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                label=label,
            )
            await session.commit()

            out = Path(args.out_dir) / f"{label}.{model}.json"
            out.write_text(result.document.model_dump_json(indent=2))

            rep = evaluate(golden, result.document, model)
            render(rep, result.input_tokens, result.output_tokens)

    print("\n[eval] JSONs salvos em", args.out_dir, file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
