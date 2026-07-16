"""Testa a atribuição de custo: matemática exata em Decimal + 1 linha por chamada.

A tabela de preços do teste é FIXA aqui — não lê pricing.json de propósito. Se o preço
real mudar (o do Sonnet muda em 31/08/2026), o teste não pode quebrar junto: ele testa
a matemática, não o preço vigente.
"""

from decimal import Decimal

import pytest
from sqlalchemy import func, select

from app.cost import (
    cost_event_exists_by_label,
    cost_usd,
    load_pricing,
    reconcile_cost_event,
    record_cost_event,
)
from app.models import CostEvent

PRICING = {
    "modelo-teste": {"input_per_mtok": "2", "output_per_mtok": "10"},
}


def test_cost_math_is_exact():
    # 70.578 in a $2/M  +  13.230 out a $10/M
    #   = 0.141156 + 0.132300 = 0.273456
    got = cost_usd("modelo-teste", 70_578, 13_230, PRICING)
    assert got == Decimal("0.273456")
    assert isinstance(got, Decimal)  # nunca float — dinheiro não arredonda sozinho


def test_zero_tokens_costs_zero():
    assert cost_usd("modelo-teste", 0, 0, PRICING) == Decimal("0")


def test_batch_api_costs_half():
    """A Batch API cobra 50%. Ignorar isso gravaria o dobro do custo real."""
    normal = cost_usd("modelo-teste", 70_578, 13_230, PRICING)
    batched = cost_usd("modelo-teste", 70_578, 13_230, PRICING, batch=True)
    assert batched == Decimal("0.136728")
    assert batched == (normal / 2).quantize(Decimal("0.000001"))


def test_unknown_model_fails_loudly():
    """Melhor quebrar do que gravar custo errado silenciosamente."""
    with pytest.raises(KeyError, match="pricing.json"):
        cost_usd("modelo-que-nao-existe", 1000, 1000, PRICING)


def test_real_pricing_file_has_the_model_we_use():
    """Guarda-corpo: o modelo default da extração precisa ter preço cadastrado."""
    from app.extraction.extract import MODEL

    assert MODEL in load_pricing(), f"{MODEL} não está no pricing.json"


async def test_batch_flag_is_persisted(db_session):
    """O flag `batch` fica gravado junto do custo — senão uma linha pela metade do preço
    pareceria bug de pricing numa auditoria."""
    await record_cost_event(
        db_session,
        agent_name="extraction",
        model="modelo-teste",
        input_tokens=1_000_000,
        output_tokens=0,
        label="susep_000",
        pricing=PRICING,
        batch=True,
    )
    ev = (await db_session.execute(select(CostEvent))).scalar_one()
    assert ev.batch is True
    assert ev.cost_usd == Decimal("1.000000")  # 1M x $2 x 50%


async def test_record_cost_event_writes_one_row_per_call(db_session):
    await record_cost_event(
        db_session,
        agent_name="extraction",
        model="modelo-teste",
        input_tokens=70_578,
        output_tokens=13_230,
        label="susep_482868",
        pricing=PRICING,
    )
    n = await db_session.scalar(select(func.count()).select_from(CostEvent))
    assert n == 1

    ev = (await db_session.execute(select(CostEvent))).scalar_one()
    assert ev.agent_name == "extraction"
    assert ev.label == "susep_482868"
    assert ev.cost_usd == Decimal("0.273456")  # sobreviveu ao round-trip do Numeric


async def test_two_calls_two_rows(db_session):
    """O grão é a CHAMADA: duas chamadas = duas linhas, mesmo pro mesmo doc."""
    for _ in range(2):
        await record_cost_event(
            db_session,
            agent_name="extraction",
            model="modelo-teste",
            input_tokens=1_000_000,
            output_tokens=0,
            label="susep_482868",
            pricing=PRICING,
        )
    n = await db_session.scalar(select(func.count()).select_from(CostEvent))
    total = await db_session.scalar(select(func.sum(CostEvent.cost_usd)))
    assert n == 2
    assert total == Decimal("4.000000")  # 2 chamadas x 1M tokens x $2


# --- reconciliação por label (resgate de run parcial) ---
# Fecha a lacuna: um run que morreu antes de gravar o custo deixaria uma chamada paga
# sem linha. O resgate grava a linha que falta, mas NUNCA conta o mesmo dinheiro 2x.


async def test_exists_by_label_reflects_the_row(db_session):
    assert await cost_event_exists_by_label(db_session, "susep_777") is False
    await record_cost_event(
        db_session,
        agent_name="extraction",
        model="modelo-teste",
        input_tokens=1000,
        output_tokens=0,
        label="susep_777",
        pricing=PRICING,
    )
    assert await cost_event_exists_by_label(db_session, "susep_777") is True


async def test_reconcile_records_when_missing(db_session):
    """Custo ausente (run morreu antes de gravar) → o resgate preenche a linha."""
    event = await reconcile_cost_event(
        db_session,
        agent_name="extraction",
        model="modelo-teste",
        input_tokens=1_000_000,
        output_tokens=0,
        label="susep_482868",
        pricing=PRICING,
        batch=True,
    )
    assert event is not None
    assert event.cost_usd == Decimal("1.000000")  # 1M x $2 x 50% (batch)
    n = await db_session.scalar(select(func.count()).select_from(CostEvent))
    assert n == 1


async def test_reconcile_is_noop_when_already_recorded(db_session):
    """Custo já registrado pelo run → o resgate NÃO grava de novo (sem contar 2x)."""
    await record_cost_event(
        db_session,
        agent_name="extraction",
        model="modelo-teste",
        input_tokens=1_000_000,
        output_tokens=0,
        label="susep_482868",
        pricing=PRICING,
        batch=True,
    )
    event = await reconcile_cost_event(
        db_session,
        agent_name="extraction",
        model="modelo-teste",
        input_tokens=1_000_000,
        output_tokens=0,
        label="susep_482868",
        pricing=PRICING,
        batch=True,
    )
    assert event is None  # nada gravado
    n = await db_session.scalar(select(func.count()).select_from(CostEvent))
    assert n == 1  # continua uma só
