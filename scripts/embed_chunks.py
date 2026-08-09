"""Fatia R2b — preenche os vetores de clause_chunk chamando a Voyage.

**Este script GASTA DINHEIRO** (ao contrário do `index_chunks.py`, que só monta o
texto). Ele pega as linhas com `embedding IS NULL`, embedda em lotes com
`input_type="document"` e grava vetor + modelo, uma linha de `cost_event` por lote.

É retomável: cada lote é commitado na hora, então uma falha no meio não joga fora o que
já foi pago — rodar de novo continua de onde parou. Rodar com tudo já indexado é no-op
e **não chama a API**.

Precisa de `VOYAGE_API_KEY` (e `DATABASE_URL`).

Uso:
    python scripts/embed_chunks.py --dry-run   # quanto falta e quanto custaria
    python scripts/embed_chunks.py             # embedda tudo que está pendente
    python scripts/embed_chunks.py --limit 50  # só os 50 primeiros pendentes
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import SessionLocal  # noqa: E402
from app.rag.embed import embed_pending, pending_stats  # noqa: E402
from app.rag.embedding import EMBED_MODEL, MAX_BATCH, MAX_BATCH_API  # noqa: E402


async def run(limit: int | None, batch_size: int, dry_run: bool, remodel: bool) -> None:
    async with SessionLocal() as session:
        prev = await pending_stats(session, limit, remodel)
        if prev["desatualizados"] and not remodel:
            # Sem isto, a mensagem seria "nada pendente" com o índice já inconsistente —
            # ver a recusa em embed_pending. Aqui só antecipamos o diagnóstico.
            sys.exit(
                f"[erro] {prev['desatualizados']} chunk(s) têm vetor de um modelo "
                f"diferente de {EMBED_MODEL}. Rode com --remodel pra re-embeddar tudo "
                "(dois modelos no mesmo índice HNSW = busca silenciosamente errada)."
            )
        if prev["chunks"] == 0:
            print("[nada] nenhum chunk pendente — tudo já tem vetor.")
            return

        lotes_previstos = -(-prev["chunks"] // batch_size)   # divisão pra cima
        print(
            f"[plano] {prev['chunks']} chunk(s) pendente(s), {prev['chars']} caracteres "
            f"=> ~{prev['tokens_estimados']} tokens em {lotes_previstos} lote(s) de até "
            f"{batch_size} — custo estimado ~US$ {prev['custo_estimado']} ({EMBED_MODEL})"
        )
        if dry_run:
            print("[dry-run] nada foi enviado à API e nada foi gravado.")
            return

        # `embed_pending` commita lote a lote (ver o docstring dela): o script NÃO é dono
        # da transação aqui, de propósito — uma falha no lote 15 tem que preservar os 14
        # que já foram pagos. Por isso o resumo abaixo sai depois dos commits, e é o
        # número que a API cobrou de verdade, não a estimativa.
        r = await embed_pending(
            session, limit=limit, batch_size=batch_size, remodel=remodel
        )

    print(
        f"[ok] {r['batches']} lote(s), {r['chunks']} chunk(s), {r['tokens']} tokens "
        f"cobrados — US$ {r['cost_usd']}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, help="embedda no máximo N chunks pendentes")
    ap.add_argument(
        "--batch-size", type=int, default=MAX_BATCH, help=f"chunks por chamada (padrão {MAX_BATCH})"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="só conta os pendentes e estima tokens/custo; não chama a API",
    )
    ap.add_argument(
        "--remodel",
        action="store_true",
        help=f"re-embedda também os chunks com vetor de outro modelo (≠ {EMBED_MODEL})",
    )
    args = ap.parse_args()

    # Validação aqui, e não só em embed_pending: `--batch-size 0` estouraria antes, num
    # ZeroDivisionError da conta de lotes previstos, e `--limit 0` é pior ainda — vira
    # `LIMIT 0`, faz `pending_stats` devolver zero e o script imprime "nenhum chunk
    # pendente — tudo já tem vetor" com o corpus inteiro por indexar. Mentira plausível
    # é o pior tipo de saída num script que gasta dinheiro.
    if args.limit is not None and args.limit < 1:
        ap.error(f"--limit tem que ser >= 1 (veio {args.limit})")
    if not 1 <= args.batch_size <= MAX_BATCH_API:
        ap.error(
            f"--batch-size tem que estar entre 1 e {MAX_BATCH_API}, o teto de itens por "
            f"requisição da Voyage (veio {args.batch_size})"
        )

    asyncio.run(run(args.limit, args.batch_size, args.dry_run, args.remodel))


if __name__ == "__main__":
    main()
