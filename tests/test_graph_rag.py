"""R3b — o worker de RAG dentro do grafo. ZERO chamada real (nem Anthropic, nem Voyage).

O client Anthropic é falso (mesma superfície de `tests/test_synth.py`: `.messages.create`
despachando por `tools`), `search_clauses` é falso, e `SessionLocal` é um objeto vazio —
o nó abre a própria sessão, mas com a busca trocada ninguém chega a consultar o banco, o
que deixa este módulo rodar sem container, como `tests/test_ask.py`.

O que se prova aqui são as três SAÍDAS que a fatia criou, e o fato de elas serem
distinguíveis entre si: a resposta sintetizada em cima de cláusulas, o "procurei e não
achei" (NADA_RELEVANTE) e o "não trato desse assunto" (FORA_DE_ESCOPO). Trocar uma pela
outra não quebra nada — só mente pro usuário —, então cada uma tem seu teste.
"""
from types import SimpleNamespace

import pytest
from httpx import ASGITransport, AsyncClient
from langchain_core.messages import AIMessage, HumanMessage

from app.agents import graph as graph_mod
from app.rag.search import MAX_DISTANCE_PADRAO, Hit

TOKEN = "token-de-teste"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

FRASE = "A cláusula exclui danos por infiltração de janela aberta."

HITS = [
    Hit(
        chunk_id=11,
        document_id=3,
        exclusion_id=42,
        coverage_id=None,
        text="Exclusão geral da apólice: danos por infiltração de janela aberta.",
        distance=0.21,
    ),
    Hit(
        chunk_id=12,
        document_id=3,
        exclusion_id=None,
        coverage_id=7,
        text="Regra de franquia da cobertura Vendaval: 10% dos prejuízos.",
        distance=0.44,
    ),
]


# --- Fakes do client Anthropic ---
class _FakeToolUse:
    type = "tool_use"

    def __init__(self, payload: dict):
        self.input = payload


class _FakeText:
    type = "text"

    def __init__(self, text: str):
        self.text = text


class _FakeResponse:
    def __init__(self, blocks: list):
        self.content = blocks


class _FakeMessages:
    def __init__(self, decisions: list[str]):
        self._decisions = decisions
        self.supervisor_calls = 0
        self.synth_calls = 0
        self.calls = 0                       # TODA chamada de LLM, de qualquer nó
        self.synth_prompts: list[str] = []   # o que o synthesizer recebeu pra sintetizar

    async def create(self, **kwargs):
        self.calls += 1
        # Sem tools = synthesizer: é o único nó que pede texto livre.
        if not kwargs.get("tools"):
            self.synth_calls += 1
            self.synth_prompts.append(kwargs["messages"][0]["content"])
            return _FakeResponse([_FakeText(FRASE)])

        i = self.supervisor_calls
        self.supervisor_calls += 1
        nxt = self._decisions[min(i, len(self._decisions) - 1)]
        return _FakeResponse([_FakeToolUse({"next": nxt, "reasoning": f"fake #{i}"})])


class _FakeClient:
    def __init__(self, decisions: list[str]):
        self.messages = _FakeMessages(decisions)


class _FakeSession:
    """`async with SessionLocal() as s` sem banco nenhum atrás.

    Com `search_clauses` falso, a session nunca é usada — ela só precisa existir e saber
    entrar e sair do `async with`, que é o contrato que o nó realmente exercita.
    """

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


@pytest.fixture
def fake_graph(monkeypatch):
    """Instala LLM falso + busca falsa e devolve (client, chamadas_da_busca)."""

    def _install(decisions: list[str], hits: list[Hit] | None = None, erro: str | None = None):
        client = _FakeClient(decisions)
        monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)
        monkeypatch.setattr("app.db.SessionLocal", lambda **kw: _FakeSession())

        buscas: list[dict] = []

        async def _busca(session, question, *, k=5, max_distance=None, **kw):
            buscas.append({"question": question, "k": k, "max_distance": max_distance})
            if erro is not None:
                raise RuntimeError(erro)
            return list(hits if hits is not None else HITS)

        monkeypatch.setattr(graph_mod, "search_clauses", _busca)

        # Custo vira no-op explícito: os fakes não têm .model/.usage e este módulo não
        # tem fixture de banco (mesmo motivo de tests/test_ask.py).
        async def _sem_custo(**kwargs):
            return None

        monkeypatch.setattr(graph_mod, "record_call_cost", _sem_custo)
        return SimpleNamespace(client=client, buscas=buscas)

    return _install


@pytest.fixture
async def ask_client(monkeypatch):
    """Client HTTP puro: com a busca falsa, este módulo inteiro roda sem Postgres."""
    monkeypatch.setenv("API_TOKENS", f"teste:{TOKEN}")
    from app.main import app

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def _perguntar(ask_client, pergunta: str):
    return await ask_client.post("/ask", json={"question": pergunta}, headers=AUTH)


async def test_rota_rag_worker_alimenta_o_synthesizer(ask_client, fake_graph):
    """O caminho feliz da fatia: supervisor -> rag_worker -> supervisor -> synthesizer.

    Não basta a busca ter rodado: o que importa é que o TEXTO recuperado chegue ao
    synthesizer. Sem essa asserção, um synthesizer que ignorasse o rag_worker (por
    exemplo procurando só por `sql_worker`, como antes desta fatia) passaria verde —
    ele devolveria a frase do fake do mesmo jeito.
    """
    f = fake_graph(["rag_worker", "END"])

    r = await _perguntar(ask_client, "Infiltração por janela aberta é coberta?")

    assert r.status_code == 200
    assert r.json()["answer"] == FRASE

    # A busca rodou uma vez, com a pergunta do usuário e SEM limiar: o corte é do nó,
    # pra que as distâncias descartadas continuem visíveis (ver test_corte_pelo_limiar).
    assert len(f.buscas) == 1
    assert f.buscas[0]["question"] == "Infiltração por janela aberta é coberta?"
    assert f.buscas[0]["k"] == graph_mod.RAG_K
    assert f.buscas[0]["max_distance"] is None

    # E o que o synthesizer recebeu foi o resultado do rag_worker — com a origem junto,
    # que é o que torna a citação possível.
    prompt = f.client.messages.synth_prompts[0]
    assert "infiltração de janela aberta" in prompt.lower()
    assert "chunk 11" in prompt and "doc 3" in prompt


async def test_unsupported_nao_gasta_nada_depois_da_decisao(ask_client, fake_graph):
    """Fora de escopo: uma chamada de LLM no total (a do supervisor) e frase fixa.

    O contador global `calls` é o ponto: `synth_calls == 0` sozinho não impediria um
    worker de ter rodado e queimado uma chamada antes de o grafo desistir.
    """
    f = fake_graph(["unsupported"])

    r = await _perguntar(ask_client, "Seguro de vida cobre suicídio?")

    assert r.status_code == 200
    assert r.json()["answer"] == graph_mod.FORA_DE_ESCOPO
    assert f.client.messages.calls == 1          # só o supervisor decidiu; nada além disso
    assert f.client.messages.synth_calls == 0
    assert f.buscas == []                        # nem a busca (que também é paga) rodou


async def test_busca_sem_hits_diz_que_nao_encontrou(ask_client, fake_graph):
    """Todos os vizinhos acima do limiar: "procurei e não achei" != "não trato disso".

    As duas frases são estáticas e as duas dispensam o LLM, o que as tornaria fáceis de
    confundir na implementação — daí a asserção explícita de que são diferentes.
    """
    f = fake_graph(["rag_worker", "END"], hits=[])

    r = await _perguntar(ask_client, "Meteorito é coberto?")

    assert r.status_code == 200
    answer = r.json()["answer"]
    assert answer == graph_mod.NADA_RELEVANTE
    assert answer != graph_mod.FORA_DE_ESCOPO
    assert answer != graph_mod.NO_ANSWER
    assert len(f.buscas) == 1                    # a busca ROdou (é o que a distingue)
    assert f.client.messages.synth_calls == 0    # não se paga pra reescrever "não achei"


async def test_corte_pelo_limiar_acontece_no_no(ask_client, fake_graph):
    """A busca traz `k` vizinhos; quem descarta os distantes é o `rag_worker`.

    Sem o corte no nó, a cláusula a 0,9 entraria no contexto do synthesizer como se
    fosse resposta — e o limiar existe exatamente pra impedir isso.
    """
    perto = HITS[0]
    longe = Hit(
        chunk_id=99,
        document_id=3,
        exclusion_id=1,
        coverage_id=None,
        text="Exclusão geral da apólice: danos por guerra.",
        distance=0.9,   # acima de MAX_DISTANCE_PADRAO
    )
    f = fake_graph(["rag_worker", "END"], hits=[perto, longe])

    r = await _perguntar(ask_client, "Infiltração por janela aberta é coberta?")

    assert r.status_code == 200
    prompt = f.client.messages.synth_prompts[0]
    assert "chunk 11" in prompt
    assert "chunk 99" not in prompt   # descartado pelo limiar
    assert "guerra" not in prompt


async def test_todos_acima_do_limiar_viram_nada_relevante(ask_client, fake_graph):
    """A busca vetorial SEMPRE devolve vizinhos — "não achei" é produzido pelo corte."""
    longe = Hit(
        chunk_id=99, document_id=3, exclusion_id=1, coverage_id=None,
        text="Exclusão geral da apólice: danos por guerra.", distance=0.9,
    )
    f = fake_graph(["rag_worker", "END"], hits=[longe])

    r = await _perguntar(ask_client, "Minha bicicleta foi roubada na rua?")

    assert r.status_code == 200
    assert r.json()["answer"] == graph_mod.NADA_RELEVANTE
    assert f.client.messages.synth_calls == 0


async def test_texto_do_hit_e_normalizado(ask_client, fake_graph):
    """Cláusula terminando em quebra de linha não pode fechar a mensagem do worker.

    Ela vira o último turno `assistant` da chamada seguinte ao supervisor, e a API
    recusa (400) um turno assistente final com espaço em branco no fim — o que
    derrubaria o /ask com 500 DEPOIS de o embedding já ter sido pago.
    """
    sujo = Hit(
        chunk_id=11, document_id=3, exclusion_id=42, coverage_id=None,
        text="Exclusão geral da apólice: danos por infiltração.\n\n", distance=0.21,
    )
    fake_graph(["rag_worker", "END"], hits=[sujo])

    r = await _perguntar(ask_client, "Infiltração é coberta?")

    assert r.status_code == 200
    conteudo = graph_mod._formatar_hits([sujo])
    assert conteudo == conteudo.rstrip()


async def test_unsupported_tardio_nao_apaga_o_trabalho_ja_pago(ask_client, fake_graph):
    """O supervisor decide DE NOVO depois do worker — e pode classificar mal aí.

    O gatilho realista é ele ler o "não encontrei" (ou as próprias cláusulas) e concluir
    que o assunto é de outro ramo. Se `unsupported` tivesse prioridade sobre o resultado,
    uma pergunta legítima receberia "só respondo sobre seguro residencial" depois de a
    busca ter sido paga — a pior das três frases, porque manda o usuário embora.
    """
    f = fake_graph(["rag_worker", "unsupported"])

    r = await _perguntar(ask_client, "Infiltração por janela aberta é coberta?")

    assert r.status_code == 200
    assert r.json()["answer"] == FRASE
    assert r.json()["answer"] != graph_mod.FORA_DE_ESCOPO
    assert f.client.messages.synth_calls == 1   # o resultado foi sintetizado, não descartado


async def test_erro_na_busca_vira_mensagem_e_nao_derruba_o_ask(ask_client, fake_graph):
    """Mesmo contrato do sql_worker: falha de infra vira texto no histórico, não 500."""
    fake_graph(["rag_worker", "END"], erro="pgvector fora do ar")

    r = await _perguntar(ask_client, "Vendaval tem franquia?")

    assert r.status_code == 200
    assert r.json()["answer"] == FRASE   # o synthesizer sintetiza até o erro; ninguém quebra


def test_route_fail_closed_continua_indo_pro_synthesizer():
    """Valor inválido em `next` → END → synthesizer (nunca um nó inexistente).

    O enum do structured output deveria impedir isso; o route() é o suspensório. As duas
    metades importam: o valor devolvido E o destino desse valor no mapa de arestas — um
    `route` correto com o mapa errado mandaria o fail-closed pra lugar nenhum.
    """
    from langgraph.graph import END

    assert graph_mod.route({"iterations": 1, "next": "worker_que_nao_existe"}) == END
    assert graph_mod.route({"iterations": 1, "next": "unsupported"}) == END
    assert graph_mod.builder.branches["supervisor"]["route"].ends[END] == "synthesizer"


async def test_synthesizer_usa_o_worker_MAIS_RECENTE(monkeypatch):
    """Com os dois workers no histórico, sintetiza o último — aqui, o rag_worker.

    Chamado direto (sem grafo) porque o que se testa é a leitura do histórico: montar
    essa situação via /ask exigiria um roteiro de supervisor que passasse pelos dois
    workers, e aí o teste dependeria do roteamento pra provar algo que é do synthesizer.
    """
    client = _FakeClient([])
    monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)

    async def _sem_custo(**kwargs):
        return None

    monkeypatch.setattr(graph_mod, "record_call_cost", _sem_custo)

    state = {
        "iterations": 3,
        "next": "END",
        "messages": [
            HumanMessage(content="Vendaval tem franquia?"),
            AIMessage(content="SQL: SELECT 1\nResult: [(1,)]", name="sql_worker"),
            AIMessage(content=f"Cláusulas recuperadas:\n{graph_mod._formatar_hits(HITS)}",
                      name="rag_worker"),
        ],
    }

    out = await graph_mod.synthesizer(state)

    assert out["messages"][0].content == FRASE
    prompt = client.messages.synth_prompts[0]
    assert "Cláusulas recuperadas" in prompt
    assert "SELECT 1" not in prompt   # o resultado ANTIGO não é o que se sintetiza


async def test_nada_relevante_nao_apaga_resultado_de_outro_worker(monkeypatch):
    """"Não encontrei" só é a resposta final quando NÃO há outro resultado.

    Se o sql_worker respondeu e o rag_worker (rodando depois) veio vazio, o dado certo
    está no histórico — devolver "não encontrei nenhuma cláusula" jogaria fora uma
    resposta já paga e mentiria sobre o que o sistema sabe.
    """
    client = _FakeClient([])
    monkeypatch.setattr(graph_mod, "get_async_client", lambda: client)

    async def _sem_custo(**kwargs):
        return None

    monkeypatch.setattr(graph_mod, "record_call_cost", _sem_custo)

    state = {
        "iterations": 4,
        "next": "END",
        "messages": [
            HumanMessage(content="Quantas seguradoras cobrem vendaval?"),
            AIMessage(content="SQL: SELECT count(*)\nResult: [(7,)]", name="sql_worker"),
            AIMessage(content=graph_mod.NADA_RELEVANTE, name="rag_worker"),
        ],
    }

    out = await graph_mod.synthesizer(state)

    assert out["messages"][0].content == FRASE
    assert out["messages"][0].content != graph_mod.NADA_RELEVANTE
    assert "Result: [(7,)]" in client.messages.synth_prompts[0]
