"""Lookup no corpus_manifest.json — proveniência autoritativa por arquivo.

O arquivo em disco é `susep_<id>.pdf`; o manifesto keyeia por `id` numérico. Uma linha
traz seguradora, processo, url, sha256, etc. — o ground truth de proveniência que a
persistência usa em vez da leitura (encurtada) do LLM.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST = ROOT / "data" / "corpus" / "corpus_manifest.json"


def manifest_row_for_pdf(pdf_path: str | Path) -> dict | None:
    """Acha a linha do manifesto pro arquivo em disco (susep_<id>.pdf)."""
    if not MANIFEST.exists():
        return None
    doc_id = Path(pdf_path).stem.split("_")[-1]
    data = json.loads(MANIFEST.read_text())
    for d in data.get("documentos", []):
        if str(d.get("id")) == doc_id:
            return d
    return None
