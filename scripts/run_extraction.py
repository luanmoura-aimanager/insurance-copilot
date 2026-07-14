"""Fatia 1 — roda a extração sobre UM PDF e imprime o objeto estruturado.

NÃO grava no banco (isso é a Fatia 2). Serve pra revisar a saída do LLM na mão contra
a CG real. Faz também um cross-check barato: compara o susep_process extraído pelo LLM
com o que o manifesto registra pra aquele arquivo (sinal de eval de graça).

Uso:
    export ANTHROPIC_API_KEY=...
    python scripts/run_extraction.py data/corpus/susep_482875.pdf
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.extraction.extract import extract_document  # noqa: E402
from app.extraction.manifest import manifest_row_for_pdf  # noqa: E402
from app.extraction.pdf import pdf_to_text  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", help="Caminho do PDF (ex.: data/corpus/susep_482875.pdf)")
    ap.add_argument("--out", help="Salva o JSON extraído aqui (pra persistir depois sem re-chamar o LLM)")
    args = ap.parse_args()

    pdf_path = Path(args.pdf)
    print(f"[1/3] Extraindo texto de {pdf_path.name} ...", file=sys.stderr)
    text = pdf_to_text(pdf_path)
    print(f"      {len(text):,} chars (~{len(text)//4:,} tokens)", file=sys.stderr)

    print(f"[2/3] Chamando o LLM ...", file=sys.stderr)
    doc = extract_document(text)

    print(f"[3/3] {len(doc.coverages)} coberturas, "
          f"{len(doc.general_exclusions)} exclusões gerais\n", file=sys.stderr)

    # Cross-check com o manifesto (ground truth de proveniência).
    row = manifest_row_for_pdf(pdf_path)
    if row:
        ok = row.get("processo") == doc.susep_process
        mark = "OK" if ok else "DIVERGE"
        print(f"[check] susep_process  LLM={doc.susep_process!r}  "
              f"manifesto={row.get('processo')!r}  -> {mark}", file=sys.stderr)
        print(f"[check] insurer        LLM={doc.insurer!r}  "
              f"manifesto={row.get('seguradora')!r}\n", file=sys.stderr)

    payload = json.dumps(doc.model_dump(), ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload)
        print(f"[out] JSON salvo em {args.out}", file=sys.stderr)
    else:
        print(payload)


if __name__ == "__main__":
    main()
