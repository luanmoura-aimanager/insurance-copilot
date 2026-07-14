"""PDF → texto. Etapa determinística antes da chamada LLM.

Usa pdfplumber (extração layout-aware). Docs escaneados (~2% do corpus, tem_texto=false
no manifesto) devolveriam texto vazio → precisariam de OCR; ficam fora da Fatia 1.
"""

from pathlib import Path

import pdfplumber


def pdf_to_text(path: str | Path) -> str:
    """Extrai o texto de todas as páginas, concatenado com quebras de página."""
    pages: list[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n\n".join(pages)
