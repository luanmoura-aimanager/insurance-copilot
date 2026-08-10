"""A camada que fala com a Voyage (`app/rag/embedding.py`) — sem banco e sem rede.

O `FakeVoyage` daqui é o cliente falso de toda a R2b (o de `tests/test_embed.py`
também): determinístico, conta chamadas e **nunca** sai da máquina. Nenhum teste da
suíte gasta dinheiro.
"""

from types import SimpleNamespace

import pytest
import voyageai.error

from app.models import EMBEDDING_DIM
from app.rag import embedding
from app.rag.embedding import (
    EMBED_MODEL,
    RATE_LIMIT_ESPERA_INICIAL,
    RATE_LIMIT_ESPERA_QUERY,
    RATE_LIMIT_TENTATIVAS,
    RATE_LIMIT_TENTATIVAS_QUERY,
    embed_documents,
    embed_query,
    get_client,
)


class FakeVoyage:
    """Cliente falso da Voyage.

    Determinístico: o vetor de um texto é função do texto (`len`), então a mesma entrada
    dá sempre a mesma saída e os testes podem afirmar *qual* chunk recebeu *qual* vetor.
    Registra cada chamada em `self.calls` — contar chamadas é o que prova que a segunda
    passada não gasta nada; olhar só o banco não provaria (uma passada que re-embedda
    tudo e regrava o mesmo valor deixa o banco idêntico e a fatura maior).
    """

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        falha_no_lote: int | None = None,
        rate_limit_nas_primeiras: int = 0,
    ):
        self.dim = dim
        self.falha_no_lote = falha_no_lote
        # Quantas das primeiras CHAMADAS (não lotes) respondem 429. Uma tentativa
        # rejeitada continua entrando em `self.calls`: é uma ida à API, e é isso que os
        # testes de retry contam.
        self.rate_limit_nas_primeiras = rate_limit_nas_primeiras
        self.calls: list[dict] = []

    def embed(self, texts, model=None, input_type=None, truncation=True):
        self.calls.append(
            {
                "texts": list(texts),
                "model": model,
                "input_type": input_type,
                "truncation": truncation,
            }
        )
        if len(self.calls) <= self.rate_limit_nas_primeiras:
            raise voyageai.error.RateLimitError("rate limit (falso)")
        if self.falha_no_lote is not None and len(self.calls) == self.falha_no_lote:
            raise RuntimeError("boom: a API da Voyage caiu no meio da passada")
        return SimpleNamespace(
            embeddings=[[float(len(t) % 100)] + [0.0] * (self.dim - 1) for t in texts],
            total_tokens=sum(max(1, len(t) // 4) for t in texts),
        )


def test_embed_documents_manda_input_type_document():
    """`input_type="document"` é o lado do par assimétrico que a indexação usa.

    A R3 vai consultar com `input_type="query"`. Errar isso não levanta erro nenhum —
    só degrada a recuperação em silêncio —, então o contrato fica travado por teste.
    """
    fake = FakeVoyage()
    vetores, tokens = embed_documents(["cláusula A", "cláusula B"], client=fake)

    assert len(fake.calls) == 1
    assert fake.calls[0]["input_type"] == "document"
    assert fake.calls[0]["model"] == EMBED_MODEL
    assert len(vetores) == 2
    assert all(len(v) == EMBEDDING_DIM for v in vetores)
    assert tokens > 0


def test_embed_query_manda_input_type_query():
    """O OUTRO lado do par assimétrico. A cláusula foi indexada como "document"; a
    pergunta tem que ir como "query", senão a Voyage a coloca perto de outras perguntas
    em vez das cláusulas que a respondem — sem erro nenhum, só piores vizinhos."""
    fake = FakeVoyage()
    vetor = embed_query("granizo está coberto?", client=fake)

    assert len(fake.calls) == 1
    assert fake.calls[0]["input_type"] == "query"
    assert fake.calls[0]["model"] == EMBED_MODEL
    assert fake.calls[0]["truncation"] is False   # o retry e os parâmetros são os mesmos
    assert len(vetor) == EMBEDDING_DIM            # o vetor, não uma lista de vetores


def test_embed_query_valida_a_dimensao():
    """Mesma validação de `embed_documents`: a mensagem tem que falar do MODELO. Aqui
    nem existe coluna pra o Postgres reclamar depois — o vetor errado iria direto pro
    `<=>` e voltaria com "different vector dimensions" no meio de uma busca."""
    with pytest.raises(ValueError, match=f"esperado {EMBEDDING_DIM}"):
        embed_query("granizo?", client=FakeVoyage(dim=512))


def test_embed_query_recusa_pergunta_vazia():
    """Espaço em branco custa uma chamada e devolve um vetor de ruído que casaria com
    qualquer cláusula do corpus — pior que não buscar."""
    fake = FakeVoyage()
    for ruim in ("", "   "):
        with pytest.raises(ValueError, match="pergunta vazia"):
            embed_query(ruim, client=fake)
    assert fake.calls == []


def test_embed_documents_desliga_a_truncagem():
    """`truncation=False` contra o padrão do SDK (`True`).

    Truncando, um texto maior que o contexto do modelo é cortado **em silêncio**: o
    `clause_chunk.text` guarda a cláusula inteira, o vetor cobre só o começo dela e
    `embedding_model` diz `voyage-4-lite` como em qualquer linha correta. A busca casa
    por um prefixo e cita a cláusula toda — o "índice que mente" de novo. Como a R2a não
    tem splitter por decisão medida, é este parâmetro que decide, no dia da primeira
    cláusula longa, entre um erro barulhento e um índice podre.
    """
    fake = FakeVoyage()
    embed_documents(["cláusula A"], client=fake)
    assert fake.calls[0]["truncation"] is False


def test_get_client_escolhe_o_timeout(monkeypatch):
    """O SDK herda 600s se ninguém escolher, e aqui 10 minutos não são só espera: o lote
    fica travado com FOR UPDATE e a transação aberta durante a chamada inteira."""
    get_client.cache_clear()
    monkeypatch.setenv("VOYAGE_API_KEY", "chave-de-teste")
    client = get_client()
    assert client._params["request_timeout"] == embedding.EMBED_TIMEOUT
    assert client.max_retries == 0   # quem retenta é _embed_com_retry, e só rate limit
    get_client.cache_clear()


def test_lista_vazia_nao_chama_a_api():
    """Um lote vazio é uma chamada paga por nada — e a API rejeitaria mesmo."""
    fake = FakeVoyage()
    assert embed_documents([], client=fake) == ([], 0)
    assert fake.calls == []


def test_dimensao_errada_levanta_erro_claro():
    """O Postgres barraria o vetor errado de qualquer jeito (`expected 1024 dimensions`),
    mas só depois do lote estar pago e dentro de uma transação, com uma mensagem que
    fala de coluna em vez de falar de modelo. A checagem tem que ser aqui."""
    fake = FakeVoyage(dim=512)
    with pytest.raises(ValueError, match=f"esperado {EMBEDDING_DIM}"):
        embed_documents(["cláusula A"], client=fake)


def test_quantidade_de_vetores_diferente_levanta_erro():
    """Um lote que volta curto alinharia vetor com chunk errado no `zip` do chamador:
    o chunk receberia, em silêncio, o vetor do vizinho — e nada no banco denunciaria."""

    class Curto(FakeVoyage):
        def embed(self, texts, model=None, input_type=None, truncation=True):
            resp = super().embed(
                texts, model=model, input_type=input_type, truncation=truncation
            )
            resp.embeddings = resp.embeddings[:-1]
            return resp

    with pytest.raises(ValueError, match="sem correspondência 1:1"):
        embed_documents(["a", "b"], client=Curto())


@pytest.fixture
def esperas(monkeypatch):
    """Captura os `time.sleep` do backoff em vez de esperar de verdade.

    Sem isto o teste de retry levaria ~30s de relógio (20s + 40s de espera real) e a
    suíte inteira viraria refém do backoff.
    """
    registradas: list[float] = []
    monkeypatch.setattr(embedding.time, "sleep", registradas.append)
    return registradas


def test_rate_limit_e_retentado_com_backoff(esperas):
    """Rate limit é a única falha que se resolve sozinha esperando — é uma janela que
    reabre. Duas tentativas recusadas e a terceira responde: o lote tem que sair inteiro,
    e as esperas têm que CRESCER (backoff), com jitter dentro da faixa esperada."""
    fake = FakeVoyage(rate_limit_nas_primeiras=2)

    vetores, tokens = embed_documents(["cláusula A"], client=fake)

    assert len(fake.calls) == 3          # duas recusadas + a que passou
    assert len(vetores) == 1 and len(vetores[0]) == EMBEDDING_DIM
    assert tokens > 0

    # Jitter de metade fixa + metade aleatória: a n-ésima espera cai em [base/2, base].
    base = RATE_LIMIT_ESPERA_INICIAL
    assert len(esperas) == 2
    assert base / 2 <= esperas[0] <= base
    assert base <= esperas[1] <= base * 2


def test_query_tem_orcamento_de_espera_proprio_e_curto(esperas):
    """A busca roda dentro de um request e segura um worker do `to_thread` enquanto
    dorme. Herdar as 6 tentativas do backfill (~190–380s de espera, mais os timeouts)
    penduraria o `/ask` por minutos e, com algumas buscas simultâneas, esgotaria o
    executor — travando também as chamadas síncronas ao Postgres dos outros nós."""
    fake = FakeVoyage(rate_limit_nas_primeiras=RATE_LIMIT_TENTATIVAS_QUERY + 1)

    with pytest.raises(voyageai.error.RateLimitError):
        embed_query("granizo?", client=fake)

    assert len(fake.calls) == RATE_LIMIT_TENTATIVAS_QUERY
    assert len(fake.calls) < RATE_LIMIT_TENTATIVAS        # menos que o do backfill
    # E a espera é curta: a do backfill começa em 20s, esta em 2s.
    assert esperas and all(e <= RATE_LIMIT_ESPERA_QUERY for e in esperas)


def test_erro_que_nao_e_rate_limit_nao_e_retentado(esperas):
    """Chave inválida, texto grande demais ou API fora do ar não melhoram com espera:
    retentar um erro real só faz o backfill levar 7 minutos pra dar a mesma mensagem."""

    class ChaveInvalida(FakeVoyage):
        def embed(self, texts, model=None, input_type=None, truncation=True):
            super().embed(
                texts, model=model, input_type=input_type, truncation=truncation
            )
            raise voyageai.error.AuthenticationError("chave inválida (falso)")

    fake = ChaveInvalida()
    with pytest.raises(voyageai.error.AuthenticationError):
        embed_documents(["cláusula A"], client=fake)

    assert len(fake.calls) == 1
    assert esperas == []


def test_rate_limit_persistente_sobe_depois_das_tentativas(esperas):
    """Desistir é o desenho, não falha: o commit por lote de `embed_pending` já preservou
    tudo que foi pago, e a re-execução retoma sozinha pelos `embedding IS NULL`. Engolir
    a exceção aqui é que seria ruim — a passada terminaria "com sucesso" e chunks."""
    fake = FakeVoyage(rate_limit_nas_primeiras=RATE_LIMIT_TENTATIVAS + 1)

    with pytest.raises(voyageai.error.RateLimitError):
        embed_documents(["cláusula A"], client=fake)

    assert len(fake.calls) == RATE_LIMIT_TENTATIVAS
    assert len(esperas) == RATE_LIMIT_TENTATIVAS - 1   # não se espera depois da última


def test_get_client_sem_chave_falha_alto(monkeypatch):
    """Fail-closed em configuração: sem `VOYAGE_API_KEY` a passada tem que morrer aqui,
    e não no meio dela com um 401 genérico — aí já haveria lotes pagos."""
    get_client.cache_clear()   # o lru_cache guardaria um cliente de outro teste
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        get_client()
    get_client.cache_clear()


def test_get_client_com_chave_vazia_tambem_falha(monkeypatch):
    """`VOYAGE_API_KEY=` num .env mal preenchido é ausência, não configuração."""
    get_client.cache_clear()
    monkeypatch.setenv("VOYAGE_API_KEY", "")
    with pytest.raises(RuntimeError, match="VOYAGE_API_KEY"):
        get_client()
    get_client.cache_clear()
