"""Fatia W2b — responde as mensagens pendentes do WhatsApp.

**Este script GASTA DINHEIRO e ENVIA MENSAGENS REAIS PARA PESSOAS REAIS.** Cada linha
pendente vira uma rodada do grafo (3 chamadas de LLM na rota de worker) e uma mensagem
entregue pela Cloud API. O `--dry-run` é o único ensaio seguro.

Ele é o BACKSTOP, não o caminho normal: em regime, quem dispara a varredura é o próprio
webhook, por BackgroundTask, logo depois do 200. Este script existe para o que aquele
gatilho não cobre — um deploy no meio de uma varredura, um processo morto, uma linha que
ficou `computed` sem sair. A forma madura é um processo `worker:` no Procfile rodando isto
em laço; ver o docstring de `app/inbox.py`.

É retomável e seguro de repetir: as duas fases (`pending` -> `computed` -> `answered`) fazem
com que uma resposta já paga NÃO seja repaga, e o `FOR UPDATE SKIP LOCKED` faz com que duas
execuções simultâneas (esta e a do webhook) não peguem a mesma linha.

Precisa de `WHATSAPP_ACCESS_TOKEN` (e `DATABASE_URL`).

Uso:
    python scripts/process_pending.py --dry-run     # o que faria, sem gastar e sem enviar
    python scripts/process_pending.py               # responde até 10
    python scripts/process_pending.py --limite 3    # prova em três antes de soltar
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ANTES de importar app.inbox, e não por hábito: (a) importar a cadeia do grafo dispara o
# `basicConfig` do FastMCP, que é o acidente que a fatia de observabilidade matou; e (b) aqui
# as linhas por mensagem SÃO o registro operacional de mensagens enviadas a pessoas — deixá-las
# no `lastResort` (stderr, sem formato, sem timestamp) é pior do que na passada de embedding.
from app.logging_config import configure_logging  # noqa: E402

configure_logging()

from app.db import SessionLocal  # noqa: E402
from app.inbox import LIMITE_PADRAO, contar_a_fazer, processar_pendentes  # noqa: E402
from app.whatsapp_api import ConfiguracaoAusente  # noqa: E402


async def run(limite: int, dry_run: bool) -> None:
    async with SessionLocal() as session:
        antes = await contar_a_fazer(session)
        a_fazer = antes["pendentes"] + antes["computadas"]
        if a_fazer == 0:
            print("[nada] nenhuma mensagem a responder — tudo já saiu (ou virou failed).")
            return

        print(
            f"[plano] {antes['pendentes']} pendente(s) + {antes['computadas']} já "
            f"computada(s) aguardando envio; esta passada pega até {limite}. Cada PENDENTE "
            "custa uma rodada do grafo (3 chamadas de LLM na rota de worker) e uma mensagem "
            "entregue; cada COMPUTADA custa só o envio, porque a resposta já foi paga."
        )
        if dry_run:
            print("[dry-run] nada foi perguntado ao grafo e nenhuma mensagem foi enviada.")
            return

        # `processar_pendentes` commita por FASE (ver o docstring dela): o script NÃO é dono
        # da transação aqui, de propósito — uma falha na quinta mensagem tem que preservar as
        # quatro já respondidas, e sobretudo as respostas já pagas.
        r = await processar_pendentes(session, limite=limite)

    print(
        f"[ok] {r['reivindicadas']} reivindicada(s), {r['computadas']} computada(s) "
        f"({r['recusadas']} recusada(s) por tamanho), {r['enviadas']} enviada(s), "
        f"{r['falhas']} falha(s), {r['desistidas']} desistida(s), {r['cedidas']} cedida(s)."
    )
    if r["desistidas"]:
        print(
            f"[erro] {r['desistidas']} mensagem(ns) bateram no teto de tentativas e NÃO "
            "foram respondidas. Os ids curtos estão no log, em ERROR."
        )
    if r["interrompidas"]:
        # Sem esta linha, uma passada que perdeu uma resposta JÁ PAGA num erro de banco
        # imprimia só `[ok]` com zeros — a mesma mentira plausível que o guard do `--limite 0`
        # existe pra impedir, entrando pelo caminho de erro. O `[ok]` acima continua saindo
        # porque o que foi enviado foi mesmo enviado; o que muda é que ele deixa de ser tudo.
        print(
            "[erro] a passada PAROU num erro de banco antes de terminar — pode haver "
            "trabalho pago perdido. O diagnóstico está no log, em ERROR."
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--limite",
        type=int,
        default=LIMITE_PADRAO,
        help=f"responde no máximo N mensagens (padrão {LIMITE_PADRAO})",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="só conta o que há a fazer; não chama o grafo e não envia nada",
    )
    args = ap.parse_args()

    # Validação aqui E em `processar_pendentes`, e a duplicação é de propósito: `--limite 0`
    # sairia do laço na primeira volta e imprimiria um relatório de zeros indistinguível de
    # "não havia nada a fazer" — mentira plausível é a pior saída possível num script que
    # gasta dinheiro e manda mensagem. E quem chama de fora do argparse não passa por aqui.
    if args.limite < 1:
        ap.error(f"--limite tem que ser >= 1 (veio {args.limite})")

    try:
        asyncio.run(run(args.limite, args.dry_run))
    except ConfiguracaoAusente as exc:
        # SÓ o fail-closed do `WHATSAPP_ACCESS_TOKEN`, que `processar_pendentes` cobra ANTES
        # da primeira reivindicação. Sai limpo: a mensagem já diz o que fazer, e um traceback
        # aqui só esconderia isso. O tipo é dedicado de propósito — `except RuntimeError`
        # engoliria também um "Event loop is closed" ou um "Session is already flushing" e os
        # imprimiria como se fossem o token faltando, escondendo um crash de verdade num
        # script que acabou de gastar dinheiro e pode ter deixado linhas no meio do caminho.
        sys.exit(f"[erro] {exc}")


if __name__ == "__main__":
    main()
