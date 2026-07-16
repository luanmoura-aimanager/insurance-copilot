"""Batch API — extração em massa por 50% do preço.

Por que batch e não um loop de chamadas síncronas: nosso caso é offline e ninguém está
esperando a resposta. A Batch API existe exatamente pra isso e cobra metade. Num corpus
de 745 docs isso é a diferença entre US$ 204 e US$ 102.

O preço é: não é imediato. Você submete, o serviço processa quando puder (até 24h, na
prática bem menos) e você busca os resultados. Daí o polling.
"""

import sys
import time
from dataclasses import dataclass

import anthropic

from .extract import build_params


@dataclass(frozen=True)
class BatchOutcome:
    """Resultado de UMA requisição do batch. `message` é None quando a request falhou."""

    custom_id: str
    status: str            # succeeded | errored | canceled | expired
    message: object | None
    error: str | None = None


def submit(items: list[tuple[str, str]], model: str | None = None) -> str:
    """Submete o batch. `items` = [(custom_id, texto_da_cg), ...]. Devolve o batch_id.

    O custom_id é como reencontramos quem é quem no retorno — usamos o stem do PDF
    (susep_<id>), que amarra o resultado de volta ao manifesto.
    """
    client = anthropic.Anthropic()
    requests = [
        {"custom_id": custom_id, "params": build_params(text, model)}
        for custom_id, text in items
    ]
    batch = client.messages.batches.create(requests=requests)
    return batch.id


def wait(batch_id: str, poll_seconds: int = 30, timeout_seconds: int = 24 * 3600) -> str:
    """Espera o batch terminar. Devolve o processing_status final."""
    client = anthropic.Anthropic()
    started = time.monotonic()
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  [batch] {batch.processing_status}  "
            f"ok={counts.succeeded} erro={counts.errored} "
            f"processando={counts.processing}",
            file=sys.stderr,
        )
        if batch.processing_status == "ended":
            return batch.processing_status
        if time.monotonic() - started > timeout_seconds:
            raise TimeoutError(f"batch {batch_id} não terminou em {timeout_seconds}s")
        time.sleep(poll_seconds)


def results(batch_id: str) -> list[BatchOutcome]:
    """Busca os resultados já concluídos do batch."""
    client = anthropic.Anthropic()
    out: list[BatchOutcome] = []
    for entry in client.messages.batches.results(batch_id):
        result = entry.result
        if result.type == "succeeded":
            out.append(BatchOutcome(entry.custom_id, "succeeded", result.message))
        else:
            detail = getattr(result, "error", None)
            out.append(
                BatchOutcome(entry.custom_id, result.type, None, error=str(detail))
            )
    return out
