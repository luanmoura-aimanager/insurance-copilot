"""Fatia R2a — materializa clause_chunk a partir do texto já persistido.

Não gasta nada: sem LLM, sem API de embedding. Lê `exclusion.clause_text` e
`coverage.deductible_rule_text` e grava uma linha de chunk por origem, com
`embedding` NULL — o vetor vem depois (R2b). É idempotente por (origem,
chunk_index), então rodar de novo é seguro; quando o texto de origem muda, o chunk
é reescrito e o vetor antigo é descartado.

Uso:
    python scripts/index_chunks.py                  # todos os documentos
    python scripts/index_chunks.py --document-id 7  # só um
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sqlalchemy import select  # noqa: E402

from app.db import SessionLocal  # noqa: E402
from app.models import PolicyDocument  # noqa: E402
from app.rag.index import index_document  # noqa: E402


async def run(document_id: int | None) -> None:
    async with SessionLocal() as session:
        if document_id is None:
            docs = (
                await session.execute(
                    select(PolicyDocument.id, PolicyDocument.insurer, PolicyDocument.product)
                    .order_by(PolicyDocument.id)
                )
            ).all()
        else:
            doc = await session.get(PolicyDocument, document_id)
            if doc is None:
                sys.exit(f"[erro] policy_document.id={document_id} não existe")
            docs = [(doc.id, doc.insurer, doc.product)]

        if not docs:
            print("[nada] nenhum documento persistido — rode a extração antes.")
            return

        total = 0
        for doc_id, insurer, product in docs:
            n = await index_document(session, doc_id)
            total += n
            # `[..]` e não `[ok]`: nada está gravado ainda. O commit é único, no fim
            # (o script é dono da transação — mesmo contrato de persist_document), e
            # um erro no documento 29 desfaz os 28 anteriores. Tudo ou nada é a escolha
            # certa aqui (diferente do run_batch, onde cada doc já custou dinheiro):
            # esta passada não gasta nada, então repetir é de graça e um documento
            # meio-indexado seria pior. Mas dizer "ok" antes do commit seria mentira.
            print(f"[..] doc {doc_id:>4}  chunks={n:>5}  {insurer} — {product}")

        await session.commit()
        print(f"[ok] commit — {len(docs)} documento(s), {total} chunks")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--document-id", type=int, help="indexa só este policy_document.id")
    args = ap.parse_args()
    asyncio.run(run(args.document_id))


if __name__ == "__main__":
    main()
