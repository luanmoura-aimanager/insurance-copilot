#!/usr/bin/env python3
"""
Sonda de viabilidade do corpus SUSEP — seguro residencial.

OBJETIVO (gate): converter "deve dar volume" em número real. Amostra IDs do
endpoint público de download da SUSEP, baixa o PDF, extrai texto e mede:
  - taxa de acerto (IDs válidos vs. inexistentes)
  - quantos são RESIDENCIAL (classificado pelo conteúdo)
  - % texto extraível vs. provável scan (precisaria de OCR)
  - tamanho médio

NÃO é o harvester final. O harvester completo (índice oficial -> IDs ->
download em massa filtrando residencial) precisa ser construído iterando
contra o site vivo — ver CLAUDE_CODE_BRIEF.md. Esta sonda usa amostragem
"cega" de IDs, que basta pra responder o gate de volumetria.

Endpoint confirmado (08/06/2026):
  https://www2.susep.gov.br/safe/menumercado/REP2/Produto.aspx/DownloadConsultaPublica/{id}
  -> ex.: id 417717 = CG Youse Auto (Content-Type: application/pdf)

Requisitos:
  pip install requests pypdf
  (no Mac, dentro de um venv; ver gotcha do keyring se pip travar)

Uso:
  python susep_probe.py --start 400000 --count 300 --out ../data/corpus
  python susep_probe.py --ids 417717,417718,417719   # IDs específicos
"""

import argparse
import io
import json
import random
import sys
import time
from pathlib import Path

import requests
from pypdf import PdfReader

BASE = "https://www2.susep.gov.br/safe/menumercado/REP2/Produto.aspx/DownloadConsultaPublica/{id}"

# Palavras que indicam ramo residencial no texto da CG.
RESIDENCIAL_HINTS = (
    "seguro residencial",
    "residência habitual",
    "residencia habitual",
    "imóvel residencial",
    "imovel residencial",
    "compreensivo residencial",
)
# Marcadores fortes de OUTRO ramo (pra descartar falso-positivo).
OTHER_RAMO = ("seguro auto", "automóvel", "automovel", "seguro de vida",
              "seguro viagem", "transportes", "responsabilidade civil profissional")

# Abaixo disso de texto extraído => provável PDF escaneado (precisa OCR).
SCAN_TEXT_THRESHOLD = 300  # caracteres

HEADERS = {"User-Agent": "insurance-copilot-research/0.1 (corpus feasibility probe)"}
RATE_LIMIT_SECONDS = 1.0  # educado: 1 req/s


def fetch_pdf(session: requests.Session, doc_id: int, timeout: int = 30):
    """Retorna bytes do PDF ou None se o ID não existir / não for PDF."""
    url = BASE.format(id=doc_id)
    try:
        r = session.get(url, headers=HEADERS, timeout=timeout)
    except requests.RequestException as e:
        return None, f"erro de rede: {e}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    ctype = r.headers.get("Content-Type", "")
    if "pdf" not in ctype.lower():
        return None, f"não-PDF ({ctype})"
    return r.content, None


def extract_text(pdf_bytes: bytes, max_pages: int = 5) -> str:
    """Extrai texto das primeiras páginas. Vazio => provável scan."""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return ""
    parts = []
    for page in reader.pages[:max_pages]:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(parts)


def classify(text: str) -> str:
    """'residencial' | 'outro' | 'indef' (sem texto -> indef/scan)."""
    if len(text) < SCAN_TEXT_THRESHOLD:
        return "indef"
    low = text.lower()
    is_resid = any(h in low for h in RESIDENCIAL_HINTS)
    is_other = any(o in low for o in OTHER_RAMO)
    if is_resid and not is_other:
        return "residencial"
    if is_resid and is_other:
        return "residencial?"  # menciona os dois — revisar manualmente
    return "outro"


def main():
    ap = argparse.ArgumentParser(description="Sonda de viabilidade SUSEP residencial")
    ap.add_argument("--start", type=int, default=400000, help="início da faixa de IDs")
    ap.add_argument("--count", type=int, default=200, help="quantos IDs amostrar")
    ap.add_argument("--span", type=int, default=100000,
                    help="amostra aleatória dentro de [start, start+span]")
    ap.add_argument("--ids", type=str, default=None,
                    help="lista de IDs específicos separados por vírgula (ignora start/count)")
    ap.add_argument("--out", type=str,
                    default=str(Path(__file__).resolve().parent.parent / "data" / "corpus"),
                    help="pasta pra salvar os PDFs residenciais")
    ap.add_argument("--save-residencial", action="store_true",
                    help="salvar os PDFs classificados como residencial")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    else:
        rng = random.Random(42)  # determinístico pra reprodutibilidade
        ids = rng.sample(range(args.start, args.start + args.span), k=args.count)

    session = requests.Session()
    stats = {"total": len(ids), "validos": 0, "invalidos": 0,
             "residencial": 0, "residencial?": 0, "outro": 0, "scan_indef": 0,
             "bytes_total": 0}
    achados = []

    print(f"Sondando {len(ids)} IDs...", file=sys.stderr)
    for i, doc_id in enumerate(ids, 1):
        pdf, err = fetch_pdf(session, doc_id)
        if pdf is None:
            stats["invalidos"] += 1
        else:
            stats["validos"] += 1
            stats["bytes_total"] += len(pdf)
            text = extract_text(pdf)
            cls = classify(text)
            if cls == "indef":
                stats["scan_indef"] += 1
            else:
                stats[cls] = stats.get(cls, 0) + 1
            if cls.startswith("residencial"):
                achados.append({"id": doc_id, "classe": cls,
                                "kb": round(len(pdf) / 1024, 1),
                                "tem_texto": len(text) >= SCAN_TEXT_THRESHOLD})
                if args.save_residencial:
                    (out / f"susep_{doc_id}.pdf").write_bytes(pdf)
        if i % 25 == 0:
            print(f"  {i}/{len(ids)} (válidos={stats['validos']}, "
                  f"resid={stats['residencial']})", file=sys.stderr)
        time.sleep(RATE_LIMIT_SECONDS)

    # Relatório
    v = stats["validos"]
    print("\n===== RESULTADO DA SONDA =====")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if v:
        scan_pct = 100 * stats["scan_indef"] / v
        resid_pct = 100 * (stats["residencial"] + stats["residencial?"]) / v
        avg_kb = stats["bytes_total"] / v / 1024
        print(f"\nTaxa de acerto de ID : {100*v/stats['total']:.1f}%")
        print(f"% residencial        : {resid_pct:.1f}% dos válidos")
        print(f"% provável scan/OCR  : {scan_pct:.1f}% dos válidos")
        print(f"Tamanho médio        : {avg_kb:.0f} KB")
        print("\nLeitura do gate:")
        print("  - taxa de acerto alta + % residencial razoável => filtrar por índice vale a pena")
        print("  - taxa de acerto baixa => NÃO enumerar cego; usar só o índice (ver brief)")
        print("  - % scan alto => orçar OCR no pipeline de extraction")
    (out / "probe_report.json").write_text(
        json.dumps({"stats": stats, "achados": achados}, indent=2, ensure_ascii=False))
    print(f"\nDetalhe salvo em {out/'probe_report.json'}")


if __name__ == "__main__":
    main()
