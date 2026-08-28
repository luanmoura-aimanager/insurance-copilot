"""A varredura que responde (fatia W2b): duas fases, uma linha por transação.

**Nenhum teste gasta dinheiro e nenhum manda mensagem**: `processar_pendentes` recebe
`graph=` e `sender=` falsos, e a rota nunca é usada aqui (quem testa a rota é
`tests/test_whatsapp_inbox.py`, e lá a varredura é o no-op autouse do conftest).

**Dois regimes de fixture no mesmo módulo, e a escolha é por teste.**

- `db_session` (sobrescrito abaixo) para quase tudo: o do `conftest.py` liga a Session a uma
  Connection já em transação, e aí o SQLAlchemy resolve `join_transaction_mode` como
  `rollback_only` — `commit()` vira NO-OP. Sob esse regime, "grava a resposta, commita,
  depois envia" e "faz tudo num commit só" são literalmente indistinguíveis, que é
  exatamente a decisão desta fatia. Com um `begin_nested()` antes, o modo vira
  `create_savepoint` e cada commit libera um SAVEPOINT durável. Mesma lição (e mesmo código)
  de `tests/test_embed.py`.
- `sessions` (fábrica ligada ao `engine`, com limpeza manual) para os testes que precisam de
  DUAS CONEXÕES: o de concorrência e o que espia o banco de dentro do `sender`. Savepoint
  vive numa conexão só, e travas entre conexões é justamente o que eles medem.
"""
import asyncio
import json
import logging
import os

import pytest
import pytest_asyncio
from sqlalchemy import delete, insert, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker
from langchain_core.messages import AIMessage

from app.inbox import (
    COMPUTADA,
    FALHOU,
    MENSAGEM_LONGA,
    PENDENTE,
    RESPONDIDA,
    contar_a_fazer,
    processar_pendentes,
)
# Capturada no IMPORT do módulo, que acontece na coleta — ou seja, ANTES de o autouse
# `sem_varredura_automatica` do conftest trocar o atributo por um no-op. É a única forma de
# um teste alcançar a função de verdade sem desfazer a trava de dinheiro pra todo mundo.
from app.inbox import varrer_em_background as varredura_real
from app.models import CostEvent, WhatsAppMessage
from app.whatsapp import id_curto
from tests.test_whatsapp import TELEFONE, TEXTO, WAMID

PHONE_NUMBER_ID = "999"
RESPOSTA = "Vendaval é coberto em 12 das 30 apólices."
TOKEN_FALSO = "token-de-envio-do-teste"


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class GrafoFalso:
    """Grafo que devolve sempre a mesma frase e CONTA as chamadas.

    Contar é o ponto: a segunda passada não pode chamar o grafo de novo, e olhar só o banco
    não provaria nada — uma versão de fase única que re-perguntasse e regravasse a MESMA
    resposta deixaria o banco idêntico e a fatura maior. Mesma razão do `self.calls` do
    `FakeVoyage`.
    """

    def __init__(self, resposta: str = RESPOSTA, erro: Exception | None = None):
        self.resposta = resposta
        self.erro = erro
        self.calls: list[str] = []

    async def ainvoke(self, state):
        self.calls.append(state["messages"][-1].content)
        if self.erro is not None:
            raise self.erro
        return {
            "iterations": 1,
            "next": "",
            "messages": [AIMessage(content=self.resposta, name="final")],
        }


class SenderFalso:
    """Envio falso. `erro` faz levantar; `falhas_iniciais` faz falhar só nas N primeiras."""

    def __init__(self, erro: Exception | None = None, falhas_iniciais: int = 0):
        self.erro = erro
        self.falhas_iniciais = falhas_iniciais
        self.chamadas: list[tuple[str, str, str]] = []

    async def __call__(self, phone_number_id, to, body):
        self.chamadas.append((phone_number_id, to, body))
        if self.falhas_iniciais > 0:
            self.falhas_iniciais -= 1
            raise self.erro or RuntimeError("envio falhou")
        if self.erro is not None:
            raise self.erro
        return "wamid.RESPOSTA"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def estado_global_limpo():
    """`_varrendo`/`_pedida_de_novo` são globais de MÓDULO, e vazam entre testes.

    Uma varredura que morre no meio (um assert de teste, um `wait_for` estourado) deixa
    `_varrendo = True`, e daí em diante todo teste que chame `varrer_em_background` sai pela
    saída antecipada e passa **vacuamente** — verde sem exercitar nada. Zerar antes e depois
    é o mesmo cuidado do `reset_limiter` do conftest com o limiter global do slowapi.
    """
    import app.inbox as inbox_mod

    inbox_mod._varrendo = False
    inbox_mod._pedida_de_novo = False
    yield
    inbox_mod._varrendo = False
    inbox_mod._pedida_de_novo = False


@pytest.fixture(autouse=True)
def env_do_canal(monkeypatch):
    """Token presente (senão a varredura se recusa a rodar) e limites previsíveis."""
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", TOKEN_FALSO)
    monkeypatch.delenv("WHATSAPP_MAX_CHARS", raising=False)
    monkeypatch.delenv("WHATSAPP_MAX_ATTEMPTS", raising=False)


@pytest_asyncio.fixture
async def db_session(engine):
    """Override do `db_session` do conftest: aqui `commit()` COMMITA de verdade.

    Ver o docstring do módulo — sem o `begin_nested()`, commit por fase e commit único no
    fim seriam indistinguíveis, e o teste da fatia passaria verde contra as duas versões.
    """
    conn = await engine.connect()
    txn = await conn.begin()
    await conn.begin_nested()
    Session = async_sessionmaker(bind=conn, expire_on_commit=False)
    session = Session()
    try:
        yield session
    finally:
        await session.close()
        await txn.rollback()
        await conn.close()


@pytest_asyncio.fixture
async def sessions(engine):
    """Fábrica de sessions REAIS (duas conexões) + limpeza manual da tabela.

    Mesmo padrão de `tests/test_whatsapp_inbox.py`: estas linhas são COMMITADAS de verdade,
    fora do alcance de qualquer rollback de fixture.
    """
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async def _limpa():
        async with Session() as s:
            await s.execute(delete(WhatsAppMessage))
            await s.commit()

    await _limpa()
    yield Session
    await _limpa()


@pytest_asyncio.fixture(autouse=True)
async def limpa_cost_event(engine):
    """`record_call_cost` commita em transação própria e escaparia do rollback do fixture,
    vazando pro `tests/test_cost.py`, que conta a tabela inteira. Mesmo motivo do
    `cost_rows` de `tests/test_cost_graph.py` e do fixture autouse de `tests/test_search.py`.
    """
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async def _limpa():
        async with Session() as s:
            await s.execute(delete(CostEvent))
            await s.commit()

    await _limpa()
    yield
    await _limpa()


@pytest.fixture
def linhas_json():
    """Desvia a saída do handler CONFIGURADO e devolve um parser das linhas JSON.

    Não dá pra usar o `caplog` pra isto: o `ContextFilter` que injeta `request_id`/`client`
    mora no HANDLER (`app/logging_config.py`), de propósito — assim nenhum call site precisa
    lembrar de passar o id, e nenhum pode esquecer. O handler de captura do pytest não é o
    nosso, então os registros dele chegam sem o campo. Mesma fixture (e mesmo motivo) de
    `tests/test_logging.py`.
    """
    import app.main  # noqa: F401 — o import é o que dispara configure_logging()

    from app.logging_config import JsonFormatter

    nossos = [h for h in logging.getLogger().handlers if isinstance(h.formatter, JsonFormatter)]
    assert len(nossos) == 1, f"esperava exatamente um handler nosso na raiz, achei {nossos}"
    handler = nossos[0]

    import io
    import json

    original, handler.stream = handler.stream, io.StringIO()
    try:
        yield lambda: [
            json.loads(l) for l in handler.stream.getvalue().splitlines() if l.strip()
        ]
    finally:
        handler.stream = original


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def planta(session, **campos) -> int:
    """Uma linha de inbox, com defaults sensatos. Devolve o id."""
    valores = {
        "wamid": WAMID,
        "from_phone": TELEFONE,
        "text": TEXTO,
        "phone_number_id": PHONE_NUMBER_ID,
        "status": PENDENTE,
    }
    valores.update(campos)
    r = await session.execute(
        insert(WhatsAppMessage).values(**valores).returning(WhatsAppMessage.id)
    )
    return r.scalar_one()


async def le(session, id_) -> dict:
    r = await session.execute(
        select(WhatsAppMessage).where(WhatsAppMessage.id == id_)
    )
    linha = r.scalar_one()
    return {
        "status": linha.status,
        "answer": linha.answer,
        "attempts": linha.attempts,
        "processed_at": linha.processed_at,
    }


# ---------------------------------------------------------------------------
# O caminho feliz e as duas fases
# ---------------------------------------------------------------------------


async def test_pendente_vira_respondida(db_session):
    id_ = await planta(db_session)
    await db_session.commit()

    grafo, sender = GrafoFalso(), SenderFalso()
    r = await processar_pendentes(db_session, limite=5, graph=grafo, sender=sender)

    assert grafo.calls == [TEXTO]
    # Os três valores do envio vêm da LINHA, não de config: é o handoff que a W1 anunciou
    # (um app secret cobre vários números; fixar um no código erraria em silêncio).
    assert sender.chamadas == [(PHONE_NUMBER_ID, TELEFONE, RESPOSTA)]

    linha = await le(db_session, id_)
    assert linha["status"] == RESPONDIDA
    assert linha["answer"] == RESPOSTA
    assert linha["processed_at"] is not None
    assert r == {"reivindicadas": 1, "computadas": 1, "recusadas": 0, "enviadas": 1,
                 "falhas": 0, "desistidas": 0, "cedidas": 0, "interrompidas": 0}


async def test_falha_no_envio_deixa_computed_com_a_resposta(db_session):
    """Fase 1 do teste da fatia: a resposta paga SOBREVIVE ao envio que falhou."""
    id_ = await planta(db_session)
    await db_session.commit()

    grafo = GrafoFalso()
    r = await processar_pendentes(
        db_session, limite=5, graph=grafo, sender=SenderFalso(erro=RuntimeError("502"))
    )

    linha = await le(db_session, id_)
    assert linha["status"] == COMPUTADA
    assert linha["answer"] == RESPOSTA        # gravada ANTES de sair — é a fatia
    assert linha["attempts"] == 1
    assert linha["processed_at"] is None      # `computed` não é terminal
    assert len(grafo.calls) == 1
    assert r["computadas"] == 1 and r["enviadas"] == 0 and r["falhas"] == 1


async def test_segunda_passada_nao_chama_o_grafo(db_session):
    """**O teste da fatia.** Envio falhou depois do grafo; a retentativa só reenvia.

    Contra uma versão de FASE ÚNICA (computar e enviar na mesma transação, sem gravar a
    resposta antes) este teste fica vermelho em `assert 2 == 1`: a segunda passada
    re-pergunta ao grafo e paga a mesma pergunta duas vezes.
    """
    id_ = await planta(db_session)
    await db_session.commit()

    grafo = GrafoFalso()
    await processar_pendentes(
        db_session, limite=5, graph=grafo, sender=SenderFalso(erro=RuntimeError("502"))
    )
    assert len(grafo.calls) == 1

    sender = SenderFalso()
    r = await processar_pendentes(db_session, limite=5, graph=grafo, sender=sender)

    assert len(grafo.calls) == 1              # <- a asserção da fatia: NÃO repagou
    assert sender.chamadas == [(PHONE_NUMBER_ID, TELEFONE, RESPOSTA)]
    linha = await le(db_session, id_)
    assert linha["status"] == RESPONDIDA
    assert linha["processed_at"] is not None
    assert r["computadas"] == 0 and r["enviadas"] == 1


async def test_orfa_computed_e_retomada_sem_o_grafo(db_session):
    """Processo morto entre os dois commits: a resposta já paga sai sem repassar pelo grafo."""
    id_ = await planta(db_session, status=COMPUTADA, answer=RESPOSTA)
    await db_session.commit()

    grafo, sender = GrafoFalso(), SenderFalso()
    r = await processar_pendentes(db_session, limite=5, graph=grafo, sender=sender)

    assert grafo.calls == []
    assert sender.chamadas == [(PHONE_NUMBER_ID, TELEFONE, RESPOSTA)]
    assert (await le(db_session, id_))["status"] == RESPONDIDA
    assert r["computadas"] == 0 and r["enviadas"] == 1


# ---------------------------------------------------------------------------
# Tentativas, inanição e aposentadoria
# ---------------------------------------------------------------------------


async def test_teto_de_tentativas_vira_failed_e_a_linha_para_de_ser_varrida(db_session, monkeypatch):
    monkeypatch.setenv("WHATSAPP_MAX_ATTEMPTS", "2")
    id_ = await planta(db_session)
    await db_session.commit()

    grafo = GrafoFalso()
    quebrado = SenderFalso(erro=RuntimeError("502"))

    await processar_pendentes(db_session, limite=5, graph=grafo, sender=quebrado)
    assert (await le(db_session, id_))["status"] == COMPUTADA

    r2 = await processar_pendentes(db_session, limite=5, graph=grafo, sender=quebrado)
    linha = await le(db_session, id_)
    assert linha["status"] == FALHOU
    assert linha["attempts"] == 2
    # `processed_at` marca estado TERMINAL, não "foi respondida": sem ele a política de
    # retenção nunca conseguiria expirar esta linha, e o caminho de FALHA viraria o
    # vazamento permanente de PII.
    assert linha["processed_at"] is not None
    assert r2["desistidas"] == 1

    # E a terceira passada não a enxerga mais: a fila anda.
    r3 = await processar_pendentes(db_session, limite=5, graph=grafo, sender=SenderFalso())
    assert r3["reivindicadas"] == 0


async def test_falha_no_grafo_tambem_conta_tentativa_e_nao_mata_a_fila(db_session, monkeypatch):
    """A guarda da INANIÇÃO. Sem incrementar `attempts` no compute, a linha envenenada é
    reivindicada primeiro em TODA varredura (ORDER BY received_at LIMIT 1) e a mensagem nova
    nunca é respondida — vermelho em `assert 'pending' == 'answered'`.
    """
    monkeypatch.setenv("WHATSAPP_MAX_ATTEMPTS", "1")
    velha = await planta(db_session, wamid="wamid.VELHA",
                         received_at=text("now() - interval '1 hour'"))
    nova = await planta(db_session, wamid="wamid.NOVA")
    await db_session.commit()

    # O grafo levanta para QUALQUER pergunta: é a indisponibilidade da Anthropic, que o
    # supervisor (sem `try`, ver CLAUDE.md) deixa subir.
    quebrado = GrafoFalso(erro=RuntimeError("anthropic fora do ar"))
    r1 = await processar_pendentes(db_session, limite=1, graph=quebrado, sender=SenderFalso())
    assert r1["desistidas"] == 1
    assert (await le(db_session, velha))["status"] == FALHOU

    grafo, sender = GrafoFalso(), SenderFalso()
    await processar_pendentes(db_session, limite=5, graph=grafo, sender=sender)
    assert (await le(db_session, nova))["status"] == RESPONDIDA
    assert len(sender.chamadas) == 1


async def test_linha_sem_phone_number_id_aposenta_sem_pagar(db_session):
    """Não há de qual número responder; pagar o grafo seria queimar dinheiro."""
    id_ = await planta(db_session, phone_number_id=None)
    await db_session.commit()

    grafo, sender = GrafoFalso(), SenderFalso()
    r = await processar_pendentes(db_session, limite=5, graph=grafo, sender=sender)

    assert grafo.calls == [] and sender.chamadas == []
    linha = await le(db_session, id_)
    assert linha["status"] == FALHOU and linha["processed_at"] is not None
    assert r["desistidas"] == 1


async def test_reivindica_a_mais_antiga_primeiro(db_session):
    """`received_at` FORA da ordem de id: na ordem "certa" um ORDER BY ignorado passaria
    verde. Mesma lição dos vetores plantados fora de ordem em `tests/test_search.py`.
    """
    await planta(db_session, wamid="wamid.C", text="terceira",
                 received_at=text("now() - interval '1 minute'"))
    await planta(db_session, wamid="wamid.A", text="primeira",
                 received_at=text("now() - interval '3 minute'"))
    await planta(db_session, wamid="wamid.B", text="segunda",
                 received_at=text("now() - interval '2 minute'"))
    await db_session.commit()

    grafo = GrafoFalso()
    await processar_pendentes(db_session, limite=3, graph=grafo, sender=SenderFalso())
    assert grafo.calls == ["primeira", "segunda", "terceira"]


# ---------------------------------------------------------------------------
# Recusa da mensagem longa
# ---------------------------------------------------------------------------


async def test_mensagem_longa_e_recusada_sem_chamar_o_grafo(db_session, monkeypatch):
    """RECUSAR, nunca truncar: a resposta enviada não pode conter pedaço da pergunta.

    Truncar responderia, com confiança, a uma pergunta que o usuário não fez — e a asserção
    palavra por palavra é o que impede uma versão que "só corte um pouquinho" de passar.
    """
    monkeypatch.setenv("WHATSAPP_MAX_CHARS", "20")
    # Palavras DISTINTIVAS de propósito: a asserção abaixo é palavra por palavra, e com uma
    # frase em português comum ela colidiria com o vocabulário da própria recusa ("de",
    # "por"), provando nada. Estas não aparecem em frase nenhuma do sistema.
    pergunta = "granizo vendaval alagamento desmoronamento incendio raio explosao vazamento"
    assert len(pergunta) > 20
    id_ = await planta(db_session, text=pergunta)
    await db_session.commit()

    grafo, sender = GrafoFalso(), SenderFalso()
    r = await processar_pendentes(db_session, limite=5, graph=grafo, sender=sender)

    assert grafo.calls == []                      # custo ZERO: a recusa substitui o grafo
    corpo = sender.chamadas[0][2]
    assert corpo == MENSAGEM_LONGA.format(limite=20)
    # O limite aparece na frase, e vem do valor que DE FATO cortou — nunca de um literal.
    assert "20" in corpo
    for palavra in pergunta.split():
        assert palavra not in corpo

    assert (await le(db_session, id_))["status"] == RESPONDIDA
    assert r["recusadas"] == 1 and r["computadas"] == 1 and r["enviadas"] == 1


async def test_mensagem_no_limite_exato_passa(db_session, monkeypatch):
    """O corte é `>`, não `>=`: a mensagem de tamanho exato é respondida."""
    monkeypatch.setenv("WHATSAPP_MAX_CHARS", "10")
    id_ = await planta(db_session, text="0123456789")
    await db_session.commit()

    grafo = GrafoFalso()
    await processar_pendentes(db_session, limite=5, graph=grafo, sender=SenderFalso())
    assert grafo.calls == ["0123456789"]
    assert (await le(db_session, id_))["answer"] == RESPOSTA


# ---------------------------------------------------------------------------
# Limite
# ---------------------------------------------------------------------------


async def test_limite_limita_e_zero_e_erro(db_session):
    for n in range(3):
        await planta(db_session, wamid=f"wamid.{n}", text=f"pergunta {n}")
    await db_session.commit()

    grafo = GrafoFalso()
    r = await processar_pendentes(db_session, limite=2, graph=grafo, sender=SenderFalso())
    assert len(grafo.calls) == 2
    assert r["reivindicadas"] == 2 and r["enviadas"] == 2

    # `limite=0` sairia do laço na primeira volta e devolveria um relatório de ZEROS
    # indistinguível de "não havia nada a fazer" — mentira plausível num caminho que gasta.
    with pytest.raises(ValueError, match="limite"):
        await processar_pendentes(db_session, limite=0, graph=grafo, sender=SenderFalso())


async def test_nada_a_fazer_devolve_zeros_sem_chamar_nada(db_session):
    grafo, sender = GrafoFalso(), SenderFalso()
    r = await processar_pendentes(db_session, limite=5, graph=grafo, sender=sender)
    assert r["reivindicadas"] == 0
    assert grafo.calls == [] and sender.chamadas == []


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("valor", ["", "CHANGE_ME_TOKEN"])
async def test_sem_token_a_varredura_nao_chama_o_grafo(db_session, monkeypatch, valor):
    """Fail-closed cobrado ANTES da primeira reivindicação.

    Sem isso, um deploy sem token pagaria o grafo por todas as pendentes e depois queimaria
    `attempts` em todas até `failed` — perder perguntas por erro de CONFIGURAÇÃO. Daí a
    segunda metade: nenhuma linha pode virar `failed`. `CHANGE_ME*` conta como ausente.
    """
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", valor)
    id_ = await planta(db_session)
    await db_session.commit()

    grafo = GrafoFalso()
    with pytest.raises(RuntimeError, match="WHATSAPP_ACCESS_TOKEN"):
        await processar_pendentes(db_session, limite=5, graph=grafo)

    assert grafo.calls == []
    assert (await le(db_session, id_))["status"] == PENDENTE


@pytest.mark.parametrize("bruto", ["abc", "0", "-1"])
def test_env_invalido_cai_no_default_com_aviso(monkeypatch, caplog, bruto):
    """Fail-OPEN validado, ao contrário do token: um typo no Railway não pode virar
    `MAX_CHARS=0` recusando todo mundo em silêncio. Mesma forma do `_limit_from_env`.
    """
    from app.inbox import MAX_ATTEMPTS_PADRAO, MAX_CHARS_PADRAO, _max_attempts, _max_chars

    monkeypatch.setenv("WHATSAPP_MAX_CHARS", bruto)
    monkeypatch.setenv("WHATSAPP_MAX_ATTEMPTS", bruto)
    with caplog.at_level(logging.WARNING, logger="app.inbox"):
        assert _max_chars() == MAX_CHARS_PADRAO
        assert _max_attempts() == MAX_ATTEMPTS_PADRAO
    # O aviso é o que impede o default de ser invisível.
    assert len([r for r in caplog.records if r.levelno == logging.WARNING]) == 2


async def test_contar_a_fazer_nao_trava_linhas(db_session):
    await planta(db_session, wamid="wamid.P")
    await planta(db_session, wamid="wamid.C", status=COMPUTADA, answer=RESPOSTA)
    await planta(db_session, wamid="wamid.F", status=FALHOU)
    await db_session.commit()

    assert await contar_a_fazer(db_session) == {
        "pendentes": 1, "computadas": 1, "respondidas": 0, "falhas": 1
    }


# ---------------------------------------------------------------------------
# Concorrência — DUAS CONEXÕES (savepoint vive numa só)
# ---------------------------------------------------------------------------


async def test_duas_varreduras_nao_pegam_a_mesma_linha(sessions):
    """`FOR UPDATE SKIP LOCKED`: a varredura B não toca na linha que A está processando.

    A trava é segurada DURANTE a chamada paga, então B chega enquanto A está dentro do
    grafo. O `wait_for` existe porque a mutação interessante (tirar `skip_locked=True`)
    produziria DEADLOCK e não falha — e suíte pendurada é o pior tipo de vermelho.
    """
    async with sessions() as s:
        await planta(s, wamid="wamid.UNICA")
        await s.commit()

    entrou, libera = asyncio.Event(), asyncio.Event()

    class GrafoQueTrava(GrafoFalso):
        async def ainvoke(self, state):
            entrou.set()
            await libera.wait()
            return await super().ainvoke(state)

    grafo_a, sender_a = GrafoQueTrava(), SenderFalso()
    grafo_b, sender_b = GrafoFalso(), SenderFalso()

    async def varredura_a():
        async with sessions() as sa:
            return await processar_pendentes(sa, limite=5, graph=grafo_a, sender=sender_a)

    async def varredura_b():
        await entrou.wait()            # A já segura a trava, dentro da chamada paga
        try:
            async with sessions() as sb:
                return await asyncio.wait_for(
                    processar_pendentes(sb, limite=5, graph=grafo_b, sender=sender_b), 10
                )
        finally:
            libera.set()

    ra, rb = await asyncio.gather(varredura_a(), varredura_b())

    assert rb["reivindicadas"] == 0, "B pegou a linha que A estava processando"
    assert grafo_b.calls == [] and sender_b.chamadas == []
    assert ra["enviadas"] == 1
    assert len(grafo_a.calls) == 1
    assert len(sender_a.chamadas) == 1     # entregue exatamente UMA vez


async def test_a_resposta_ja_esta_no_banco_QUANDO_o_envio_e_chamado(sessions):
    """A prova DIRETA da ordem das fases: de dentro do `sender`, outra conexão já vê a
    resposta commitada como `computed`.

    Os testes 2 e 3 só INFEREM isso do comportamento entre passadas; este observa o instante.
    Contra uma versão de fase única, `visto` fica `('pending', None)`.
    """
    async with sessions() as s:
        await planta(s, wamid="wamid.ESPIA")
        await s.commit()

    visto = {}

    async def sender_que_espia(phone_number_id, to, body):
        async with sessions() as outra:
            r = await outra.execute(select(WhatsAppMessage.status, WhatsAppMessage.answer))
            visto["linha"] = r.one()

    async with sessions() as sa:
        await processar_pendentes(sa, limite=1, graph=GrafoFalso(), sender=sender_que_espia)

    assert visto["linha"] == (COMPUTADA, RESPOSTA)


# ---------------------------------------------------------------------------
# PII e correlação
# ---------------------------------------------------------------------------


# Só tokens INVENTADOS, e nenhum é substring de palavra do sistema. A asserção de ausência
# é palavra por palavra, então uma frase em português comum daria falso vermelho contra o
# próprio log ("em" dentro de "mensagem") — e, pior, esconderia um vazamento de verdade atrás
# de um teste que ninguém confia.
PERGUNTA_MARCADA = "xilofonezinho queimadaxis varandalha zumbibundo pratelheiro"
RESPOSTA_MARCADA = "morcegoblau trepadeirinha zanzibarcus florestalva bicicletax"


async def test_o_log_nao_carrega_a_pergunta_nem_a_resposta(db_session, caplog):
    """Nem a pergunta nem a resposta saem no log — nem inteiras, nem em pedaços.

    **A pré-condição vem primeiro, e ela é a lição do wamid**: um teste de ausência só prova
    algo se a mesma busca ENCONTRARIA a string caso ela estivesse lá. O `id_curto` no fim é o
    controle — ele é logado de propósito e é achado pela mesma busca, então o `not in` acima
    está medindo o log e não um caplog vazio.

    Renderizado com um `logging.Formatter()` PELADO de propósito: a garantia tem que ser do
    call site, não do `JsonFormatter` — quem mandar estes registros pra outro lugar (um
    trace, um handler de terceiro) perderia a proteção sem aviso.
    """
    await planta(db_session, wamid="wamid.PII", text=PERGUNTA_MARCADA)
    await db_session.commit()

    with caplog.at_level(logging.DEBUG, logger="app.inbox"):
        await processar_pendentes(
            db_session, limite=5,
            graph=GrafoFalso(resposta=RESPOSTA_MARCADA),
            sender=SenderFalso(),
        )

    formatter = logging.Formatter()
    tudo = "\n".join(formatter.format(r) for r in caplog.records)

    # Controle: a busca funciona, e o operador continua servido.
    assert id_curto("wamid.PII") in tudo
    assert "8888" in tudo                       # os 4 últimos, que é o que se mascara PARA
    assert str(len(RESPOSTA_MARCADA)) in tudo   # o TAMANHO da resposta, nunca ela

    # E a ausência, palavra por palavra: truncar continuaria vazando o começo.
    for texto in (PERGUNTA_MARCADA, RESPOSTA_MARCADA):
        assert texto not in tudo
        for palavra in texto.split():
            assert palavra not in tudo, f"{palavra!r} vazou no log"

    assert TELEFONE not in tudo
    assert "wamid.PII" not in tudo              # o wamid cru embute o telefone em base64


async def test_cost_event_da_varredura_tem_request_id_e_nao_tem_a_pergunta(
    db_session, engine, linhas_json, monkeypatch
):
    """O `cost_event` da varredura é correlacionável, e o mesmo id aparece no log dela.

    A varredura roda FORA de um request, então sem o `set_request_context` por linha o custo
    e o log ficariam órfãos — e com várias mensagens no mesmo lote seria impossível dizer
    qual chamada foi de qual pergunta. `client` fica NULL de propósito: ninguém se autenticou,
    e a coluna continua significando "identidade do token".
    """
    from app.agents.context import get_client_name, get_request_id
    from app.cost import record_call_cost

    Session = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.SessionLocal", lambda **kw: Session(**kw))

    ids_vistos = []

    class GrafoQueGasta(GrafoFalso):
        async def ainvoke(self, state):
            ids_vistos.append((get_request_id(), get_client_name()))
            await record_call_cost(
                agent_name="supervisor", model="claude-haiku-4-5-20251001",
                input_tokens=100, output_tokens=10,
            )
            return await super().ainvoke(state)

    await planta(db_session, wamid="wamid.CUSTO", text=PERGUNTA_MARCADA)
    await db_session.commit()

    await processar_pendentes(
        db_session, limite=5,
        graph=GrafoQueGasta(resposta=RESPOSTA_MARCADA), sender=SenderFalso(),
    )

    (request_id, client) = ids_vistos[0]
    assert request_id is not None
    assert client is None

    async with Session() as s:
        eventos = list((await s.execute(select(CostEvent))).scalars())
    assert len(eventos) == 1
    assert eventos[0].request_id == request_id
    assert eventos[0].client is None
    # A pergunta não entra no livro-caixa, nem como `label`.
    for coluna in (eventos[0].label, eventos[0].agent_name):
        assert coluna is None or PERGUNTA_MARCADA not in coluna

    # O MESMO id no log daquela varredura — é isso que "correlacionado" significa.
    correlacionadas = [l for l in linhas_json() if "request_id" in l]
    assert correlacionadas, "nenhuma linha da varredura saiu correlacionada"
    assert {l["request_id"] for l in correlacionadas} == {request_id}


# ---------------------------------------------------------------------------
# A borda de saída (`app/whatsapp_api.py`) — sem rede, com MockTransport
# ---------------------------------------------------------------------------


async def test_enviar_texto_monta_o_pedido_certo():
    import httpx

    from app.whatsapp_api import GRAPH_API_VERSION, enviar_texto

    visto = {}

    def responder(request: httpx.Request) -> httpx.Response:
        visto["url"] = str(request.url)
        visto["auth"] = request.headers.get("authorization")
        visto["body"] = json.loads(request.content)
        return httpx.Response(200, json={"messages": [{"id": "wamid.SAIU"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(responder)) as c:
        wamid = await enviar_texto(PHONE_NUMBER_ID, TELEFONE, RESPOSTA, client=c)

    assert wamid == "wamid.SAIU"
    assert visto["url"] == (
        f"https://graph.facebook.com/{GRAPH_API_VERSION}/{PHONE_NUMBER_ID}/messages"
    )
    assert visto["auth"] == f"Bearer {TOKEN_FALSO}"
    assert visto["body"] == {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": TELEFONE,
        "type": "text",
        # `preview_url=False` é decisão, não default: o corpo é escrito por um MODELO, e
        # ligar o preview faria a plataforma buscar uma URL que o modelo inventou.
        "text": {"preview_url": False, "body": RESPOSTA},
    }


async def test_erro_de_envio_nao_carrega_o_corpo():
    """A exceção identifica a falha e NÃO publica o telefone, a resposta nem o token.

    `raise_for_status()` levantaria uma `HTTPStatusError` que segura `.request` — cujo
    `.content` é exatamente o corpo que acabamos de mandar. O `from None` fecha a última
    fresta: `traceback_da_cadeia` respeita `__suppress_context__`, então o objeto do httpx
    não volta ao log pela cadeia.
    """
    import httpx

    from app.whatsapp_api import EnvioRecusado, enviar_texto

    def recusar(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401,
            json={"error": {"code": 190, "error_subcode": 463,
                            "message": f"token expirado para {TELEFONE}"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(recusar)) as c:
        with pytest.raises(EnvioRecusado) as exc_info:
            await enviar_texto(PHONE_NUMBER_ID, TELEFONE, RESPOSTA_MARCADA, client=c)

    exc = exc_info.value
    assert exc.status == 401
    # O operador continua servido: classe, status e os códigos ESTRUTURADOS do catálogo.
    assert "http=401" in str(exc) and "code=190" in str(exc) and "error_subcode=463" in str(exc)

    # E nada do que saiu, nem do que a Meta ecoou de volta em texto livre.
    for palavra in RESPOSTA_MARCADA.split():
        assert palavra not in str(exc)
    assert TELEFONE not in str(exc)
    assert TOKEN_FALSO not in str(exc)
    # Sem a cadeia: o objeto do httpx (com a request dentro) não viaja pro log.
    assert exc.__suppress_context__ is True
    assert exc.__cause__ is None


async def test_erro_de_rede_vira_envio_recusado_sem_status():
    import httpx

    from app.whatsapp_api import EnvioRecusado, enviar_texto

    def cair(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"conexão recusada para {TELEFONE}", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(cair)) as c:
        with pytest.raises(EnvioRecusado) as exc_info:
            await enviar_texto(PHONE_NUMBER_ID, TELEFONE, RESPOSTA, client=c)

    assert exc_info.value.status is None
    assert "ConnectError" in str(exc_info.value)
    assert TELEFONE not in str(exc_info.value)


async def test_200_com_corpo_estranho_nao_e_falha():
    """Um 200 com JSON inesperado é envio BEM-SUCEDIDO: a mensagem saiu.

    Perder o wamid da resposta é perder correlação, não perder a entrega — e levantar aqui
    faria a varredura contar como falha (e REENVIAR) algo que já chegou na pessoa.
    """
    import httpx

    from app.whatsapp_api import enviar_texto

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, text="ok"))
    ) as c:
        assert await enviar_texto(PHONE_NUMBER_ID, TELEFONE, RESPOSTA, client=c) is None


def test_get_access_token_falha_alto_e_nomeia_a_variavel(monkeypatch):
    """`RuntimeError`, NÃO `HTTPException`: os dois consumidores rodam fora de um request."""
    from fastapi import HTTPException

    from app.whatsapp_api import get_access_token

    monkeypatch.delenv("WHATSAPP_ACCESS_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="WHATSAPP_ACCESS_TOKEN") as exc_info:
        get_access_token()
    assert not isinstance(exc_info.value, HTTPException)


# ---------------------------------------------------------------------------
# O gatilho em background
# ---------------------------------------------------------------------------


async def test_varredura_em_background_nunca_levanta(engine, monkeypatch, caplog):
    """Ela roda DEPOIS de a resposta ter sido enviada — levantar viraria um "Exception in
    ASGI application" fantasma pendurado num request que deu 200.
    """
    import app.inbox as inbox_mod

    Session = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.SessionLocal", lambda **kw: Session(**kw))

    async def _explode(*_a, **_kw):
        raise RuntimeError("o grafo caiu")

    monkeypatch.setattr(inbox_mod, "processar_pendentes", _explode)

    with caplog.at_level(logging.ERROR, logger="app.inbox"):
        assert await varredura_real() is None                   # não levantou

    erros = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(erros) == 1
    assert "RuntimeError" in erros[0].getMessage()


async def test_a_rereivindicacao_tem_as_DUAS_metades(sessions):
    """As duas metades do anti-envio-duplo, exercitadas direto no statement.

    O teste de concorrência acima não alcança esta janela: lá a varredura B é barrada já na
    PRIMEIRA reivindicação, porque A segura a trava durante a chamada paga. A janela que
    `_rereivindicar` protege é outra e é estreita — entre o commit da fase 1 (que SOLTA a
    trava) e o começo do envio —, e reproduzi-la com duas varreduras exigiria um gancho
    dentro do laço. Então prova-se o statement, que é onde a garantia mora. (Verificado: sem
    este teste, as duas mutações abaixo passam com a suíte inteira verde.)

    - **Sem `skip_locked`**: o passo 3 BLOQUEARIA em vez de voltar vazio, e a segunda
      varredura enviaria assim que a primeira soltasse — a segunda cópia no telefone de
      alguém. O `wait_for` é o que transforma esse deadlock em vermelho.
    - **Sem o `AND status = 'computed'`**: o passo 4 devolveria a linha e a varredura
      mandaria de novo uma mensagem que já foi entregue. O predicado tem que estar no SQL e
      não num `if`: em READ COMMITTED o EvalPlanQual RE-CHECA a linha contra o WHERE depois
      de esperar outra transação, e é isso que torna "eu li isso há pouco" em "ainda vale".
    """
    from app.inbox import _rereivindicar

    async with sessions() as s:
        id_ = await planta(s, wamid="wamid.RECLAIM", status=COMPUTADA, answer=RESPOSTA)
        await s.commit()

    async with sessions() as x:
        # Controle: com a linha livre e `computed`, ela VOLTA — senão os dois vazios abaixo
        # seriam vácuo e o teste não mediria nada.
        assert (await x.execute(_rereivindicar(id_))).mappings().first() is not None

        # (1) travada por outra conexão -> vazio, sem bloquear
        async with sessions() as y:
            vazio = await asyncio.wait_for(y.execute(_rereivindicar(id_)), 10)
            assert vazio.mappings().first() is None
            await y.rollback()
        await x.rollback()

    # (2) já não é mais `computed` -> vazio, mesmo sem trava nenhuma
    async with sessions() as s:
        await s.execute(
            WhatsAppMessage.__table__.update()
            .where(WhatsAppMessage.id == id_)
            .values(status=RESPONDIDA)
        )
        await s.commit()
    async with sessions() as z:
        assert (await z.execute(_rereivindicar(id_))).mappings().first() is None


async def test_falha_de_banco_no_meio_para_a_varredura_e_nao_loga_PII(db_session, caplog):
    """Erro de banco: a passada PARA, o log identifica sem PII, e nada explode pra fora.

    Parar é a decisão: `_falhar` grava na MESMA transação que ainda segura a trava, então
    uma varredura que não consegue registrar a tentativa reivindicaria a mesma linha em laço
    dentro da mesma passada, sem progresso nenhum.

    O log é montado à mão e **não** por `logger.exception`, e a exceção deste teste é uma de
    verdade, com os valores ligados visíveis no texto — que é exatamente o que o `exc_info`
    imprimiria. `hide_parameters=True` na engine tapa isso em produção, mas a garantia não
    pode DEPENDER de um flag de engine: quem construir a sessão de outro jeito (este teste,
    um worker futuro) a perde sem aviso. Mesma regra do `_resumo_do_erro` da W2a.
    """
    erro = await _erro_de_update_com_pii(RESPOSTA_MARCADA)
    # Pré-condição: o teste só prova algo se a exceção de fato contiver a PII.
    assert RESPOSTA_MARCADA in str(erro), "o caso ruim não foi reproduzido"

    await planta(db_session, wamid="wamid.DB1", text=PERGUNTA_MARCADA,
                 received_at=text("now() - interval '1 minute'"))
    await planta(db_session, wamid="wamid.DB2", text="segunda pergunta")
    await db_session.commit()

    class SessaoQueQuebraNoUpdate:
        """Passa o SELECT da reivindicação e estoura no UPDATE da resposta."""

        def __init__(self, real):
            self._real = real

        async def execute(self, stmt, *a, **kw):
            if stmt.__visit_name__ == "update":
                raise erro
            return await self._real.execute(stmt, *a, **kw)

        async def commit(self):
            await self._real.commit()

        async def rollback(self):
            await self._real.rollback()

    with caplog.at_level(logging.DEBUG, logger="app.inbox"):
        r = await processar_pendentes(
            SessaoQueQuebraNoUpdate(db_session), limite=5,
            graph=GrafoFalso(resposta=RESPOSTA_MARCADA), sender=SenderFalso(),
        )

    # Parou na primeira: a segunda linha não chegou a ser reivindicada.
    assert r["reivindicadas"] == 1 and r["enviadas"] == 0

    formatter = logging.Formatter()
    tudo = "\n".join(formatter.format(rec) for rec in caplog.records)

    assert id_curto("wamid.DB1") in tudo        # controle: a busca funciona
    assert "a varredura PARA" in tudo
    assert "sqlstate=" in tudo                  # o operador continua servido
    for palavra in RESPOSTA_MARCADA.split() + PERGUNTA_MARCADA.split():
        assert palavra not in tudo, f"{palavra!r} vazou no log"


async def test_falha_de_envio_nao_loga_a_resposta(db_session, caplog):
    """Exceção GENÉRICA do sender: o WARNING leva só a classe, nunca o texto.

    `sender` é injetável e o log não pode DEPENDER de a exceção ser estéril — um envio
    futuro, ou um erro do httpx que escape, traria o corpo junto. Este é o ramo conservador;
    o de baixo é o outro, onde `str(exc)` PODE sair porque a `EnvioRecusado` é estéril por
    construção.
    """
    await planta(db_session, wamid="wamid.ENV", text=PERGUNTA_MARCADA)
    await db_session.commit()

    explosivo = SenderFalso(erro=RuntimeError(f"falhou mandando {RESPOSTA_MARCADA}"))
    with caplog.at_level(logging.DEBUG, logger="app.inbox"):
        await processar_pendentes(
            db_session, limite=5,
            graph=GrafoFalso(resposta=RESPOSTA_MARCADA), sender=explosivo,
        )

    formatter = logging.Formatter()
    tudo = "\n".join(formatter.format(rec) for rec in caplog.records)

    assert "envio falhou" in tudo and id_curto("wamid.ENV") in tudo    # controle
    for palavra in RESPOSTA_MARCADA.split() + PERGUNTA_MARCADA.split():
        assert palavra not in tudo, f"{palavra!r} vazou no log"


async def _erro_de_update_com_pii(resposta: str):
    """Um erro de UPDATE real, com os valores ligados visíveis no texto dele."""
    from sqlalchemy.ext.asyncio import create_async_engine

    eng = create_async_engine(os.environ["DATABASE_URL"], hide_parameters=False)
    try:
        async with async_sessionmaker(eng)() as s:
            try:
                await s.execute(
                    text("UPDATE whatsapp_message SET answer = :a, nao_existe = 1")
                    .bindparams(a=resposta)
                )
            except BaseException as exc:
                return exc
    finally:
        await eng.dispose()
    raise AssertionError("o UPDATE não falhou")


# ---------------------------------------------------------------------------
# Erro de banco DEPOIS de gastar: o orçamento tem que ser cobrado
# ---------------------------------------------------------------------------


class SessaoQueQuebra:
    """Deixa passar tudo menos o statement escolhido, na enésima vez que ele aparece."""

    def __init__(self, real, alvo: str, no_commit: bool = False, pular_commits: int = 0):
        self._real, self._alvo, self._no_commit = real, alvo, no_commit
        self._pular = pular_commits
        self.armado = True

    async def execute(self, stmt, *a, **kw):
        nome = getattr(stmt, "__visit_name__", "")
        if self.armado and not self._no_commit and nome == self._alvo:
            self.armado = False
            raise await _erro_de_update_com_pii(RESPOSTA_MARCADA)
        return await self._real.execute(stmt, *a, **kw)

    async def commit(self):
        if self.armado and self._no_commit:
            if self._pular > 0:
                self._pular -= 1
            else:
                self.armado = False
                raise await _erro_de_update_com_pii(RESPOSTA_MARCADA)
        await self._real.commit()

    async def rollback(self):
        await self._real.rollback()


async def test_perder_a_resposta_do_grafo_no_banco_COBRA_a_tentativa(db_session):
    """A resposta foi paga e não coube no banco: sem cobrar, a próxima varredura REPAGA.

    Este é o caminho que o `attempts` não cobria: a linha ficava `pending` com `attempts=0`,
    e como a reivindicação é `ORDER BY received_at LIMIT 1` ela seria a primeira escolhida em
    toda varredura futura — pagando o grafo de novo, sem teto. Mutação: tirar o `teto=teto`
    daquele `_erro_de_banco` e este teste fica vermelho em `attempts == 0`.
    """
    id_ = await planta(db_session, wamid="wamid.PERDE1")
    await db_session.commit()

    grafo = GrafoFalso(resposta=RESPOSTA_MARCADA)
    r = await processar_pendentes(
        SessaoQueQuebra(db_session, "update"), limite=5, graph=grafo, sender=SenderFalso()
    )

    linha = await le(db_session, id_)
    assert linha["attempts"] == 1, "a rodada paga não foi cobrada do orçamento"
    assert linha["status"] == PENDENTE          # a resposta se perdeu; a pergunta continua
    assert linha["answer"] is None
    assert r["interrompidas"] == 1 and r["falhas"] == 1


async def test_falha_no_commit_do_answered_COBRA_a_tentativa(db_session):
    """A mensagem SAIU e o `answered` não commitou: sem cobrar, ela é reenviada PARA SEMPRE.

    É a janela at-least-once. A documentação promete que `attempts` limita o estrago, e antes
    da correção isso era falso: aquele caminho ia pro `_erro_de_banco`, que não tocava o
    contador — então toda varredura futura reenviava a mesma mensagem à mesma pessoa, sem
    teto. Cobrar transforma "pode duplicar" em "duplica no máximo N vezes".
    """
    id_ = await planta(db_session, wamid="wamid.PERDE2", status=COMPUTADA, answer=RESPOSTA)
    await db_session.commit()

    sender = SenderFalso()
    r = await processar_pendentes(
        # O primeiro commit é o que SOLTA a linha órfã, antes do envio; o alvo é o segundo,
        # que é o `answered` — depois de a mensagem já ter saído.
        SessaoQueQuebra(db_session, "update", no_commit=True, pular_commits=1),
        limite=5, graph=GrafoFalso(), sender=sender,
    )

    assert len(sender.chamadas) == 1            # a mensagem de fato saiu
    linha = await le(db_session, id_)
    assert linha["attempts"] == 1, "o reenvio ficaria sem teto"
    assert linha["status"] == COMPUTADA         # vai ser reenviada — mas contada
    assert r["interrompidas"] == 1


async def test_erro_de_banco_aparece_no_relatorio(db_session):
    """`interrompidas` existe pro `scripts/process_pending.py` não imprimir `[ok]` limpo
    depois de perder trabalho pago — a mesma mentira plausível que o guard do `--limite 0`
    impede, entrando pelo caminho de erro.
    """
    await planta(db_session, wamid="wamid.REL")
    await db_session.commit()
    r = await processar_pendentes(
        SessaoQueQuebra(db_session, "update"), limite=5,
        graph=GrafoFalso(), sender=SenderFalso(),
    )
    assert r["interrompidas"] == 1
    assert any(v for k, v in r.items() if k != "reivindicadas"), \
        "o relatório sairia com tudo zerado menos as reivindicadas"


# ---------------------------------------------------------------------------
# Estado malformado do grafo e teto do corpo de saída
# ---------------------------------------------------------------------------


async def test_estado_malformado_do_grafo_conta_tentativa_e_nao_escapa(db_session):
    """`final_answer` fica DENTRO do `try` do grafo.

    Fora dele, um estado sem `messages` sai como `KeyError` por cima de `processar_pendentes`
    inteiro: nenhuma tentativa contada, a linha reivindicada primeiro em toda varredura
    futura (a inanição que o `attempts` existe pra impedir), e traceback cru no CLI.
    """
    class GrafoQueDevolveLixo(GrafoFalso):
        async def ainvoke(self, state):
            self.calls.append(state["messages"][-1].content)
            return {"iterations": 1, "next": "", "messages": []}

    id_ = await planta(db_session, wamid="wamid.LIXO")
    await db_session.commit()

    r = await processar_pendentes(
        db_session, limite=5, graph=GrafoQueDevolveLixo(), sender=SenderFalso()
    )   # não levanta

    assert (await le(db_session, id_))["attempts"] == 1
    assert r["falhas"] == 1


async def test_resposta_gigante_e_cortada_com_aviso_em_vez_de_virar_silencio(db_session):
    """Corpo acima do teto da Cloud API sai 400, e a resposta é DETERMINÍSTICA — as três
    tentativas falhariam idênticas e a pessoa ficaria sem nada.

    O caminho é real: com o synthesizer caído o grafo degrada pra saída CRUA do worker (as k
    cláusulas do RAG, ou até 100 linhas do `run_query`). Cortar aqui é o oposto de truncar a
    PERGUNTA: o que se corta é a resposta, e o corte é ANUNCIADO no texto que a pessoa lê.
    """
    from app.inbox import RESPOSTA_TRUNCADA
    from app.whatsapp_api import MAX_BODY_CHARS

    gigante = "x" * (MAX_BODY_CHARS + 5000)
    id_ = await planta(db_session, wamid="wamid.GIGANTE")
    await db_session.commit()

    sender = SenderFalso()
    await processar_pendentes(
        db_session, limite=5, graph=GrafoFalso(resposta=gigante), sender=sender
    )

    corpo = sender.chamadas[0][2]
    assert len(corpo) <= MAX_BODY_CHARS
    assert corpo.endswith(RESPOSTA_TRUNCADA)     # a pessoa é avisada de que foi cortado
    # E o que ficou gravado é o que SAIU: reenviar tem que mandar a mesma coisa.
    assert (await le(db_session, id_))["answer"] == corpo
    assert (await le(db_session, id_))["status"] == RESPONDIDA


async def test_resposta_no_tamanho_normal_nao_e_tocada(db_session):
    """O controle: o corte só age acima do teto — senão toda resposta ganharia o aviso."""
    await planta(db_session, wamid="wamid.NORMAL")
    await db_session.commit()
    sender = SenderFalso()
    await processar_pendentes(db_session, limite=5, graph=GrafoFalso(), sender=sender)
    assert sender.chamadas[0][2] == RESPOSTA


# ---------------------------------------------------------------------------
# Uma varredura em voo por processo, e o encadeamento de rodadas
# ---------------------------------------------------------------------------


async def test_varredura_concorrente_no_mesmo_processo_desiste(engine, monkeypatch):
    """Duas entregas quase simultâneas não abrem duas varreduras.

    Cada varredura segura uma conexão do pool com transação ABERTA durante a chamada ao grafo
    e o POST — N entregas viravam N conexões travadas, o pool default (5+10) esgotava, o
    `SessionLocal()` do próprio webhook estourava `TimeoutError`, a rota devolvia 500 e a Meta
    reentregava, agendando mais varreduras. Laço de realimentação que acaba com a inscrição
    desabilitada.
    """
    import app.inbox as inbox_mod

    Session = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.SessionLocal", lambda **kw: Session(**kw))

    entrou, libera = asyncio.Event(), asyncio.Event()
    em_voo, pico = [0], [0]

    async def _devagar(*_a, **_kw):
        # Mede CONCORRÊNCIA, não contagem: desde que o disparo engolido passou a ser
        # atendido com uma volta a mais, duas chamadas sequenciais são o comportamento
        # certo — o que não pode acontecer é duas ao mesmo tempo, que é o que esgota o pool.
        em_voo[0] += 1
        pico[0] = max(pico[0], em_voo[0])
        try:
            entrou.set()
            await libera.wait()
            return _contagens_de_teste()
        finally:
            em_voo[0] -= 1

    monkeypatch.setattr(inbox_mod, "processar_pendentes", _devagar)

    async def segunda():
        await entrou.wait()
        try:
            # `wait_for` porque a mutação interessante (tirar a checagem de `_varrendo`) não
            # falha: a segunda passada fica esperando o `libera` que só esta corotina solta —
            # deadlock, e suíte pendurada é o pior tipo de vermelho.
            await asyncio.wait_for(varredura_real(), 5)   # tem que DESISTIR, não esperar
        finally:
            libera.set()



    await asyncio.gather(varredura_real(), segunda())
    assert pico[0] == 1, f"{pico[0]} passadas em paralelo — cada uma trava uma conexão do pool"


def _contagens_de_teste():
    from app.inbox import _contagens_zeradas

    return _contagens_zeradas()


async def test_lote_maior_que_o_limite_e_drenado_no_mesmo_disparo(sessions, monkeypatch):
    """A Meta entrega em LOTE, e o agendamento só dispara de uma entrega com pergunta.

    Com um teto rígido de `limite`, um lote de 5 com `limite=2` deixaria 3 paradas até a
    PRÓXIMA entrega — que pode não vir. Por isso `varrer_em_background` encadeia rodadas,
    **mas só enquanto a rodada foi cheia E inteiramente bem-sucedida**: as linhas que falharam
    não estão no `vistas` de uma passada nova, então uma rodada seguinte as pegaria de volta e
    queimaria o orçamento em milissegundos — o mesmo bug que o `excluir` mata dentro de uma
    passada. O teste seguinte cobre esse lado.
    """
    import app.inbox as inbox_mod

    monkeypatch.setattr("app.db.SessionLocal", lambda **kw: sessions(**kw))

    async with sessions() as s:
        for n in range(5):
            await planta(s, wamid=f"wamid.LOTE{n}", text=f"pergunta {n}")
        await s.commit()

    grafo, sender = GrafoFalso(), SenderFalso()
    real = inbox_mod.processar_pendentes

    async def com_falsos(session, limite=2, **_kw):
        return await real(session, limite=limite, graph=grafo, sender=sender)

    monkeypatch.setattr(inbox_mod, "processar_pendentes", com_falsos)
    await varredura_real(limite=2)

    async with sessions() as s:
        restantes = (await s.execute(
            select(WhatsAppMessage).where(WhatsAppMessage.status != RESPONDIDA)
        )).scalars().all()
    assert restantes == [], f"{len(restantes)} mensagem(ns) ficaram sem resposta"
    assert len(sender.chamadas) == 5


async def test_o_encadeamento_PARA_na_primeira_falha(sessions, monkeypatch):
    """Rodada com falha não encadeia — senão a retentativa seria imediata.

    Sem esta condição, a linha que acabou de falhar voltaria na rodada seguinte (o `vistas` é
    por passada) e o orçamento inteiro sumiria em milissegundos, transformando um soluço da
    Cloud API em `failed`. O intervalo entre tentativas é o intervalo entre DISPAROS.
    """
    import app.inbox as inbox_mod

    monkeypatch.setattr("app.db.SessionLocal", lambda **kw: sessions(**kw))
    monkeypatch.setenv("WHATSAPP_MAX_ATTEMPTS", "3")

    async with sessions() as s:
        for n in range(4):
            await planta(s, wamid=f"wamid.PARA{n}", text=f"pergunta {n}")
        await s.commit()

    grafo = GrafoFalso()
    sender = SenderFalso(erro=RuntimeError("502"))
    real = inbox_mod.processar_pendentes

    async def com_falsos(session, limite=2, **_kw):
        return await real(session, limite=limite, graph=grafo, sender=sender)

    monkeypatch.setattr(inbox_mod, "processar_pendentes", com_falsos)
    await varredura_real(limite=2)

    async with sessions() as s:
        linhas_ = (await s.execute(
            select(WhatsAppMessage).order_by(WhatsAppMessage.id)
        )).scalars().all()
    # Uma rodada só: duas linhas tentadas UMA vez cada, e nenhuma aposentada.
    assert [l.attempts for l in linhas_] == [1, 1, 0, 0]
    assert all(l.status != FALHOU for l in linhas_)


# ---------------------------------------------------------------------------
# Segunda rodada de review: os caminhos de erro do próprio caminho de erro
# ---------------------------------------------------------------------------


async def test_falha_ao_registrar_a_tentativa_nao_estoura(db_session, caplog):
    """`_falhar` chamando `_erro_de_banco`: a assinatura estava errada e virava `TypeError`.

    O caminho é alcançável — o UPDATE da contabilidade também falha numa conexão morta — e o
    `TypeError` escapava de `processar_pendentes` inteiro: nenhuma tentativa contada, a linha
    reivindicada primeiro em toda varredura futura (a inanição que o `attempts` existe pra
    impedir) e traceback cru no CLI. Mutação: voltar `linha["wamid"]` no lugar de `linha` e
    tirar o `contagens` — este teste fica vermelho.
    """
    await planta(db_session, wamid="wamid.CONTAB")
    await db_session.commit()

    class SessaoQueQuebraNaContabilidade:
        """Deixa o SELECT e o UPDATE da resposta passarem; quebra o UPDATE da tentativa."""

        def __init__(self, real):
            self._real, self._updates = real, 0

        async def execute(self, stmt, *a, **kw):
            if getattr(stmt, "__visit_name__", "") == "update":
                self._updates += 1
                if self._updates == 2:            # 1 = grava a resposta, 2 = conta tentativa
                    raise RuntimeError("conexão morta")
            return await self._real.execute(stmt, *a, **kw)

        async def commit(self):
            await self._real.commit()

        async def rollback(self):
            await self._real.rollback()

        def in_transaction(self):
            return self._real.in_transaction()

    with caplog.at_level(logging.DEBUG, logger="app.inbox"):
        r = await processar_pendentes(              # não levanta
            SessaoQueQuebraNaContabilidade(db_session), limite=5,
            graph=GrafoFalso(), sender=SenderFalso(erro=RuntimeError("502")),
        )

    assert r["interrompidas"] == 1
    assert "registrar a tentativa" in "\n".join(rec.getMessage() for rec in caplog.records)


async def test_cobranca_nao_desfaz_o_que_outra_varredura_concluiu(db_session):
    """A cobrança pós-rollback roda SEM a trava, então precisa da guarda de status.

    Cenário: A envia, o commit de `answered` falha, A solta a trava; B reivindica, reenvia e
    marca `answered`; A então cobra a tentativa. Sem `WHERE status IN ('pending','computed')`
    no `_contar_tentativa`, o `case((esgotou, FALHOU))` sobrescreveria um estado TERMINAL —
    uma mensagem entregue registrada como nunca entregue.

    Aqui o estado terminal é plantado direto, que é o mesmo que B teria deixado.
    """
    from app.inbox import _contar_tentativa

    id_ = await planta(db_session, wamid="wamid.TERM", status=RESPONDIDA,
                       answer=RESPOSTA, attempts=2)
    await db_session.commit()

    assert (await db_session.execute(_contar_tentativa(id_, 3))).first() is None
    linha = await le(db_session, id_)
    assert linha["status"] == RESPONDIDA, "a cobrança sobrescreveu um estado terminal"
    assert linha["attempts"] == 2


async def test_entrega_feita_com_registro_perdido_nao_diz_que_ninguem_respondeu(
    db_session, caplog
):
    """A linha de ERROR da aposentadoria é a ÚNICA evidência de que alguém ficou sem resposta.

    No caminho em que a mensagem SAIU e só o `answered` não commitou, usar essa mesma frase
    mandaria o operador caçar um usuário que foi respondido — talvez mais de uma vez.
    """
    id_ = await planta(db_session, wamid="wamid.ENTREGUE", status=COMPUTADA,
                       answer=RESPOSTA, attempts=2)
    await db_session.commit()

    sender = SenderFalso()
    with caplog.at_level(logging.DEBUG, logger="app.inbox"):
        r = await processar_pendentes(
            SessaoQueQuebra(db_session, "update", no_commit=True, pular_commits=1),
            limite=5, graph=GrafoFalso(), sender=sender,
        )

    assert len(sender.chamadas) == 1
    assert (await le(db_session, id_))["status"] == FALHOU
    tudo = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "FOI ENTREGUE" in tudo
    assert "NÃO recebeu resposta" not in tudo
    # E não conta como desistência: ninguém ficou sem resposta.
    assert r["desistidas"] == 0


async def test_token_sumindo_no_meio_nao_gasta_tentativa(db_session, caplog):
    """`enviar_texto` lê o token a CADA envio, então ele some depois do fail-closed.

    Cobrar tentativa aí aposentaria o lote inteiro como `failed` por erro de CONFIGURAÇÃO —
    o mesmo desfecho que o fail-closed prévio existe pra impedir, só que mais tarde.
    """
    from app.whatsapp_api import ConfiguracaoAusente

    id_ = await planta(db_session, wamid="wamid.TOKEN")
    await db_session.commit()

    async def sender_sem_token(*_a):
        raise ConfiguracaoAusente("WHATSAPP_ACCESS_TOKEN ausente ou vazia")

    with caplog.at_level(logging.DEBUG, logger="app.inbox"):
        r = await processar_pendentes(
            db_session, limite=5, graph=GrafoFalso(), sender=sender_sem_token
        )

    linha = await le(db_session, id_)
    assert linha["attempts"] == 0, "erro de configuração gastou o orçamento da mensagem"
    assert linha["status"] == COMPUTADA          # a resposta paga sobrevive
    assert r["falhas"] == 0 and r["desistidas"] == 0
    assert "CONFIGURAÇÃO" in "\n".join(rec.getMessage() for rec in caplog.records)


async def test_pedido_durante_a_varredura_nao_e_perdido(engine, monkeypatch):
    """Um disparo engolido pelo `_varrendo` tem que ser atendido antes de a varredura sair.

    Sem a marca, a entrega que chegou durante uma varredura em curso ficava esperando a
    PRÓXIMA — que pode não vir —, e o README prometia o contrário.
    """
    import app.inbox as inbox_mod

    Session = async_sessionmaker(engine, expire_on_commit=False)
    monkeypatch.setattr("app.db.SessionLocal", lambda **kw: Session(**kw))

    entrou, libera = asyncio.Event(), asyncio.Event()
    rodadas = []

    async def _passada_falsa(*_a, **_kw):
        rodadas.append(1)
        if len(rodadas) == 1:
            entrou.set()
            await libera.wait()
        return _contagens_de_teste()

    monkeypatch.setattr(inbox_mod, "processar_pendentes", _passada_falsa)

    async def segunda_entrega():
        await entrou.wait()
        try:
            await asyncio.wait_for(varredura_real(), 5)   # engolida — mas marcada
            # A saída antecipada NÃO pode ter soltado a trava: quem a segura é a varredura
            # que continua rodando, e soltá-la deixaria a próxima entrega abrir uma passada
            # em paralelo — o esgotamento de pool que a trava existe pra impedir. É invisível
            # pelo resultado (as duas rodadas saem certas de qualquer jeito), então a
            # asserção é sobre o ESTADO, no instante certo.
            assert inbox_mod._varrendo is True, "a saída antecipada soltou a trava alheia"
        finally:
            # `finally`, e não a linha seguinte: se o assert acima falhar, a outra corotina
            # fica presa em `await libera.wait()` PARA SEMPRE e a suíte PENDURA em vez de
            # ficar vermelha — e não há `pytest-timeout` neste projeto. É a mesma armadilha
            # que os `asyncio.wait_for` deste módulo evitam, e ela escapou aqui uma vez.
            libera.set()

    await asyncio.gather(varredura_real(), segunda_entrega())
    assert len(rodadas) == 2, "o disparo perdido não foi atendido"


async def test_varredura_em_background_nunca_levanta_nem_no_import(monkeypatch, caplog):
    """O `if _varrendo` e o import ficam DENTRO do `try`.

    Fora dele, um `ImportError` de `app.db` (deploy parcial, ciclo novo) escapava pela
    BackgroundTask e virava o "Exception in ASGI application" fantasma pendurado num 200 —
    exatamente o que o `except` existe pra matar.
    """
    import builtins

    real_import = builtins.__import__

    def _import_quebrado(nome, *a, **kw):
        if nome == "app.db":
            raise ImportError("app.db quebrado")
        return real_import(nome, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _import_quebrado)
    with caplog.at_level(logging.ERROR, logger="app.inbox"):
        assert await varredura_real() is None      # não levantou

    assert "ImportError" in "\n".join(rec.getMessage() for rec in caplog.records)


async def test_envio_recusado_leva_o_diagnostico_da_meta_pro_log(db_session, caplog):
    """O outro ramo: a `EnvioRecusado` é estéril, então o resumo dela SAI no log.

    Sem isso o operador via só `EnvioRecusado http=401` e não distinguia token expirado
    (code=190) de janela de 24h (131047) de erro de template — e toda falha de rede saía
    idêntica, sem status nenhum. O `_resumo_da_falha` de `app/whatsapp_api.py` virava código
    morto: montado com cuidado e jogado fora no call site.
    """
    from app.whatsapp_api import EnvioRecusado

    await planta(db_session, wamid="wamid.DIAG", text=PERGUNTA_MARCADA)
    await db_session.commit()

    recusa = SenderFalso(erro=EnvioRecusado("HTTPStatusError http=401 code=190", 401))
    with caplog.at_level(logging.DEBUG, logger="app.inbox"):
        await processar_pendentes(
            db_session, limite=5,
            graph=GrafoFalso(resposta=RESPOSTA_MARCADA), sender=recusa,
        )

    tudo = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "code=190" in tudo and "http=401" in tudo
    # E continua sem PII: a EnvioRecusado é estéril por construção, e é ISSO que autoriza
    # o `str(exc)` aqui — ver `test_erro_de_envio_nao_carrega_o_corpo`.
    for palavra in RESPOSTA_MARCADA.split() + PERGUNTA_MARCADA.split():
        assert palavra not in tudo


async def test_o_pedido_extra_NAO_repega_a_linha_que_acabou_de_falhar(sessions, monkeypatch):
    """O `_pedida_de_novo` não pode furar a guarda do encadeamento.

    Ele existe pra atender a entrega que chegou durante a varredura — mas a rodada extra
    começa com `_reivindicar` de novo, e sem um `vistas` compartilhado ela repegaria a linha
    que ACABOU de falhar: as três tentativas queimadas em milissegundos, e no caminho em que
    o POST deu certo e só o registro falhou, a MESMA mensagem reenviada três vezes seguidas.
    """
    import app.inbox as inbox_mod

    monkeypatch.setattr("app.db.SessionLocal", lambda **kw: sessions(**kw))
    monkeypatch.setenv("WHATSAPP_MAX_ATTEMPTS", "3")

    async with sessions() as s:
        await planta(s, wamid="wamid.EXTRA")
        await s.commit()

    grafo = GrafoFalso()
    sender = SenderFalso(erro=RuntimeError("502"))
    real = inbox_mod.processar_pendentes
    rodadas = [0]

    async def com_falsos(session, limite=1, vistas=None, **_kw):
        rodadas[0] += 1
        if rodadas[0] == 1:
            # Simula a entrega que chega no meio: o disparo dela é engolido pelo `_varrendo`.
            inbox_mod._pedida_de_novo = True
        return await real(session, limite=limite, graph=grafo, sender=sender, vistas=vistas)

    monkeypatch.setattr(inbox_mod, "processar_pendentes", com_falsos)
    await varredura_real(limite=1)

    async with sessions() as s:
        linha = (await s.execute(select(WhatsAppMessage))).scalar_one()
    assert rodadas[0] >= 2, "o pedido extra não foi atendido"
    assert linha.attempts == 1, (
        f"a rodada extra repegou a linha e gastou {linha.attempts} tentativas de uma vez"
    )
    assert linha.status == COMPUTADA
    assert len(sender.chamadas) == 1, "a mesma mensagem foi enviada mais de uma vez"


async def test_falha_ao_REIVINDICAR_e_reportada_como_as_outras(db_session, caplog):
    """A reivindicação era a única chamada de banco da passada sem tratamento.

    Sem o `try`, um Postgres reiniciado ali escapava de `processar_pendentes` inteiro: sem
    `interrompidas`, sem a linha com classe + SQLSTATE, e saindo do
    `scripts/process_pending.py` como traceback cru — os dois desfechos que os comentários
    deste módulo dizem impedir. Todas as outras chamadas já tinham o seu.
    """
    erro = await _erro_de_update_com_pii(RESPOSTA_MARCADA)

    class SessaoQueQuebraNaReivindicacao:
        def __init__(self, real):
            self._real = real

        async def execute(self, stmt, *a, **kw):
            if getattr(stmt, "__visit_name__", "") == "select":
                raise erro
            return await self._real.execute(stmt, *a, **kw)

        async def commit(self):
            await self._real.commit()

        async def rollback(self):
            await self._real.rollback()

        def in_transaction(self):
            return self._real.in_transaction()

    await planta(db_session, wamid="wamid.CLAIM", text=PERGUNTA_MARCADA)
    await db_session.commit()

    with caplog.at_level(logging.DEBUG, logger="app.inbox"):
        r = await processar_pendentes(          # não levanta
            SessaoQueQuebraNaReivindicacao(db_session), limite=5,
            graph=GrafoFalso(), sender=SenderFalso(),
        )

    assert r["interrompidas"] == 1
    tudo = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "reivindicar a próxima linha" in tudo
    assert "sqlstate=" in tudo                   # o operador continua servido
    # E sem PII, como todo o resto dos caminhos de erro deste módulo.
    for palavra in PERGUNTA_MARCADA.split() + RESPOSTA_MARCADA.split():
        assert palavra not in tudo
