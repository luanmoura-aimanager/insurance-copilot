"""Atribuição de custo — 1 chamada LLM = 1 linha em cost_event.

Duas decisões que sustentam isso:

- **Preço é config, não código.** Vive em `pricing.json`. Não é firula: o Sonnet 5 está em
  preço promocional até 31/08/2026 e sobe depois — hardcodar o número no código faria o
  custo histórico virar mentira silenciosa.
- **Decimal, não float.** Custo é dinheiro. float acumula erro de arredondamento ao longo
  de milhares de chamadas; Decimal é exato.

Se o modelo não estiver no pricing, FALHA ALTO. Melhor quebrar do que gravar custo errado.
"""

import json
from decimal import Decimal
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import CostEvent

PRICING_PATH = Path(__file__).resolve().parent / "pricing.json"
_MILLION = Decimal(1_000_000)
_CENTS = Decimal("0.000001")  # 6 casas — bate com Numeric(12, 6) da coluna

# A Batch API cobra 50% do preço normal. Não é detalhe: se o cálculo ignorar isso, o
# cost_event grava o DOBRO do que foi cobrado de verdade — número errado com cara de certo.
# https://platform.claude.com/docs/en/about-claude/pricing
BATCH_MULTIPLIER = Decimal("0.5")


def load_pricing(path: Path | None = None) -> dict:
    """Carrega a tabela de preços; chaves com `_` são notas, não modelos."""
    data = json.loads((path or PRICING_PATH).read_text())
    return {k: v for k, v in data.items() if not k.startswith("_")}


def cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    pricing: dict | None = None,
    batch: bool = False,
) -> Decimal:
    """Custo exato de UMA chamada, em USD. `batch=True` aplica o desconto da Batch API."""
    pricing = load_pricing() if pricing is None else pricing
    if model not in pricing:
        raise KeyError(
            f"modelo {model!r} não está no pricing.json — adicione o preço antes de rodar, "
            "senão o custo gravado seria errado."
        )
    p = pricing[model]
    per_in = Decimal(str(p["input_per_mtok"]))
    per_out = Decimal(str(p["output_per_mtok"]))
    total = (Decimal(input_tokens) * per_in + Decimal(output_tokens) * per_out) / _MILLION
    if batch:
        total *= BATCH_MULTIPLIER
    return total.quantize(_CENTS)


async def record_cost_event(
    session: AsyncSession,
    *,
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    label: str | None = None,
    pricing: dict | None = None,
    batch: bool = False,
) -> CostEvent:
    """Grava 1 linha por chamada. NÃO commita — o chamador é dono da transação."""
    event = CostEvent(
        agent_name=agent_name,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=cost_usd(model, input_tokens, output_tokens, pricing, batch),
        batch=batch,
        label=label,
    )
    session.add(event)
    await session.flush()
    return event


async def cost_event_exists_by_label(
    session: AsyncSession, label: str, *, batch: bool | None = None
) -> bool:
    """Já existe uma linha de custo pra essa chamada (identificada por `label`)?

    `batch` restringe a busca ao tipo de chamada: o MESMO doc pode ter uma linha da eval
    (batch=False, label=stem) e uma do batch (batch=True, mesmo label). O doc do golden é
    fixado em todo batch E é o que a eval roda, então essa colisão de label é real — sem o
    filtro, uma linha da eval mascararia o custo faltante de uma chamada de batch.
    """
    q = select(CostEvent.id).where(CostEvent.label == label)
    if batch is not None:
        q = q.where(CostEvent.batch == batch)
    hit = await session.scalar(q)
    return hit is not None


async def reconcile_cost_event(
    session: AsyncSession,
    *,
    agent_name: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    label: str,
    pricing: dict | None = None,
    batch: bool = False,
) -> CostEvent | None:
    """Grava a linha de custo SÓ se ainda não houver uma pra essa chamada (por `label`).

    É o que fecha a lacuna do resgate: se o run original morreu antes de registrar o
    custo (ex.: TimeoutError no batch.wait), o resgate preenche; se o custo já foi
    registrado, não conta o mesmo dinheiro duas vezes. Devolve o evento novo ou None.

    Reconcilia por (`label`, `batch`): a existência é checada só contra linhas do MESMO
    tipo de chamada, senão uma linha da eval (batch=False) mascararia o custo faltante de
    uma chamada de batch (batch=True) do mesmo doc — colisão real no doc do golden. Uma
    re-extração via --force gera uma chamada nova com o MESMO (label, batch): se um run
    anterior já deixou linha, esta reconciliação não grava a nova. É o trade-off consciente
    do resgate: preferimos nunca contar em dobro a garantir cada centavo de re-extrações.
    """
    if await cost_event_exists_by_label(session, label, batch=batch):
        return None
    return await record_cost_event(
        session,
        agent_name=agent_name,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        label=label,
        pricing=pricing,
        batch=batch,
    )
