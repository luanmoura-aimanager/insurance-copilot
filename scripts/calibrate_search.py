"""Calibra o limiar de relevância da busca semântica (R3a).

Roda `search_clauses` **sem limiar** (`max_distance=None`) sobre perguntas rotuladas e
mostra as distâncias que voltaram. Sem corte, a busca sempre devolve `k` vizinhos — até
para uma pergunta sobre bicicleta —, então o número que separa "achei" de "não sei" só
pode sair de olhar as duas distribuições lado a lado:

    maior distância do 1º hit entre os '+'   <   menor distância do 1º hit entre os '-'

Se essa desigualdade valer, qualquer valor no meio serve de limiar e é ele que vai para
`app/rag/search.py::MAX_DISTANCE_PADRAO`. Se as faixas se cruzarem, não existe corte por
distância que separe os dois grupos — e isso também é resultado: significa que o filtro
tem que vir de outro lugar (mais contexto no chunk, ou o supervisor decidindo escopo).

**GASTA DINHEIRO**, pouco: uma chamada de embedding por pergunta (o arquivo padrão tem
15, todas de uma linha). Precisa de `VOYAGE_API_KEY`, `DATABASE_URL` e do corpus já
embeddado (`scripts/embed_chunks.py`).

Uso:
    python scripts/calibrate_search.py                          # data/eval/search_questions.txt
    python scripts/calibrate_search.py --questions outro.txt
    python scripts/calibrate_search.py -k 10
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.db import SessionLocal  # noqa: E402
from app.rag.search import search_clauses  # noqa: E402

PADRAO = ROOT / "data" / "eval" / "search_questions.txt"

# Quanto do melhor hit cabe na tabela. Só pra dar pra reconhecer a cláusula de relance —
# quem quiser o texto inteiro busca pelo chunk_id, que sai impresso ao lado.
TRECHO = 60


def carregar(caminho: Path) -> list[tuple[str, str]]:
    """Lê `[('+'|'-', pergunta), ...]`. Linhas vazias e comentários (`#`) são ignorados.

    Um prefixo desconhecido é erro, não linha ignorada: uma pergunta que sai calada da
    calibração enviesa as médias sem aparecer em lugar nenhum.
    """
    perguntas: list[tuple[str, str]] = []
    for n, linha in enumerate(caminho.read_text().splitlines(), start=1):
        linha = linha.strip()
        if not linha or linha.startswith("#"):
            continue
        rotulo, resto = linha[0], linha[1:].strip()
        if rotulo not in "+-" or not resto:
            sys.exit(
                f"[erro] {caminho}:{n}: cada pergunta precisa começar com '+' "
                f"(em escopo) ou '-' (fora de escopo) — veio {linha!r}"
            )
        perguntas.append((rotulo, resto))
    if not perguntas:
        sys.exit(f"[erro] nenhuma pergunta em {caminho}")
    return perguntas


def _dist(hits: list, i: int) -> str:
    """Distância do i-ésimo hit (0-based), ou '—' se a busca voltou com menos que isso."""
    return f"{hits[i].distance:.3f}" if len(hits) > i else "—"


async def run(caminho: Path, k: int) -> None:
    perguntas = carregar(caminho)
    print(f"[plano] {len(perguntas)} pergunta(s), k={k}, SEM limiar — {len(perguntas)} "
          "chamada(s) de embedding à Voyage\n")

    linhas = []
    primeiros: dict[str, list[float]] = {"+": [], "-": []}

    async with SessionLocal() as session:
        for i, (rotulo, pergunta) in enumerate(perguntas, start=1):
            hits = await search_clauses(session, pergunta, k=k)
            # `search_clauses` abre transação (o `set_config` local) e não fecha — o dono
            # da transação é o chamador. Sem este rollback, a conexão ficaria
            # `idle in transaction` durante TODAS as chamadas HTTP seguintes, que é
            # exatamente onde um Postgres com `idle_in_transaction_session_timeout`
            # derruba a sessão no meio da passada, com os embeddings já pagos.
            await session.rollback()
            # Progresso linha a linha: a tabela só é impressa no fim (precisa da largura
            # máxima), e uma passada que morre no meio não pode sair muda depois de ter
            # gasto — o `\r` some quando a tabela é impressa por cima.
            print(f"  [{i}/{len(perguntas)}] {pergunta[:50]}", end="\r", flush=True)
            if hits:
                primeiros[rotulo].append(hits[0].distance)
            melhor = (
                f"#{hits[0].chunk_id} {hits[0].text[:TRECHO]}".replace("\n", " ")
                if hits
                else "(nada indexado)"
            )
            linhas.append(
                (rotulo, pergunta, _dist(hits, 0), _dist(hits, 2), _dist(hits, 4), melhor)
            )

    largura = max(len(p) for _, p, *_ in linhas)
    print(" " * 60)   # apaga a última linha de progresso antes da tabela
    cab = f"{'':2} {'pergunta':<{largura}}  {'d1':>6} {'d3':>6} {'d5':>6}  melhor hit"
    print(cab)
    print("-" * len(cab))
    for rotulo, pergunta, d1, d3, d5, melhor in linhas:
        print(f"{rotulo:2} {pergunta:<{largura}}  {d1:>6} {d3:>6} {d5:>6}  {melhor}")

    print()
    for rotulo, nome in (("+", "em escopo"), ("-", "fora de escopo")):
        ds = primeiros[rotulo]
        if ds:
            print(f"[{nome:>14}] {len(ds)} pergunta(s), distância média do 1º hit: "
                  f"{sum(ds) / len(ds):.3f}")

    # É AQUI que o limiar mora: entre o pior caso do que deve passar e o melhor caso do
    # que deve ser barrado. A ordem dos dois números é o resultado da calibração.
    if primeiros["+"] and primeiros["-"]:
        pior_bom, melhor_ruim = max(primeiros["+"]), min(primeiros["-"])
        print(f"\n[fronteira] maior '+' = {pior_bom:.3f} | menor '-' = {melhor_ruim:.3f}")
        if pior_bom < melhor_ruim:
            meio = (pior_bom + melhor_ruim) / 2
            print(f"[fronteira] os grupos SEPARAM — qualquer limiar em "
                  f"({pior_bom:.3f}, {melhor_ruim:.3f}) serve; meio = {meio:.3f}. "
                  "Esse é o candidato a MAX_DISTANCE_PADRAO.")
        else:
            print("[fronteira] os grupos SE CRUZAM — nenhum limiar por distância separa "
                  "os dois. Escolher um número aqui só troca falso positivo por falso "
                  "negativo; o corte precisa vir de outro lugar.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", type=Path, default=PADRAO, help=f"padrão: {PADRAO}")
    ap.add_argument("-k", type=int, default=5, help="hits por pergunta (padrão 5)")
    args = ap.parse_args()

    if args.k < 1:
        ap.error(f"-k tem que ser >= 1 (veio {args.k})")
    if not args.questions.exists():
        ap.error(f"arquivo de perguntas não encontrado: {args.questions}")

    asyncio.run(run(args.questions, args.k))


if __name__ == "__main__":
    main()
