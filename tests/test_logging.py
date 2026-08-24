"""O log estruturado: uma linha JSON por registro, em stdout, correlacionada por request.

Dois regimes, e a separação é deliberada.

Os testes que mexem em CONFIGURAÇÃO GLOBAL rodam em SUBPROCESSO. Dentro do pytest,
`dictConfig` arranca os handlers da raiz — onde vive o `LogCaptureHandler` que alimenta o
`caplog` —, e o sintoma é o pior possível: teste de log que passa sem ver log nenhum. É a
mesma armadilha que `tests/conftest.py` já documenta a respeito do `fileConfig` do Alembic.
Em subprocesso o processo é limpo e a asserção é sobre o stdout de verdade.

Os demais trocam o `stream` do NOSSO handler por um StringIO: exercitam o formatter, o
filtro e o nível como estão configurados, em vez de reconstruí-los à mão.
"""
import io
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import text

# Fixtures e constantes reaproveitadas: o `/ask` com grafo falso já está montado ali, e
# duplicá-lo aqui criaria uma segunda versão pra manter. Mesmo padrão de
# `tests/test_whatsapp_inbox.py`, que importa de `tests/test_whatsapp.py`.
from tests.test_cost_graph import (  # noqa: F401 — as fixtures entram pelo namespace
    AUTH,
    CLIENTE,
    TOKEN,
    _eventos,
    ask_client,
    cost_rows,
    fake_graph,
)
from tests.test_whatsapp import TELEFONE, TEXTO, WAMID

REPO_ROOT = Path(__file__).resolve().parent.parent


def _run_child(code: str, tmp_path, **extra_env) -> subprocess.CompletedProcess:
    """Roda `code` num Python com env limpo (sem DATABASE_URL) e fora do repo.

    Mesmo molde de `tests/test_lazy_db.py`: env por allowlist (não `os.environ.copy()`),
    cwd fora do repo pra não pegar o `.env` do dev. A diferença é que aqui o STDOUT do
    filho é o que está sob teste, então ele é asserido no pai.
    """
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": str(REPO_ROOT),
        **extra_env,
    }
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


# Preâmbulo comum dos filhos. As duas neutralizações são o ponto do teste de cima:
#
# - `dotenv.load_dotenv`: o filho tem que enxergar o mesmo mundo do CI (sem .env), e a
#   heurística do `find_dotenv` muda com a versão do python-dotenv. Mesmo motivo do
#   `tests/test_lazy_db.py`.
# - `logging.basicConfig`: é O ACIDENTE. `mcp_servers/postgres_mcp_server.py` constrói um
#   FastMCP no nível do módulo e o `__init__` dele chama `basicConfig`; como
#   `app/agents/graph.py` importa dali, `import app.main` acabava configurando a raiz em
#   INFO — em stderr, sem formato — de graça. Sem matar isso, este teste passaria verde
#   contra um app que não configura logging nenhum.
_PREAMBULO = """
import dotenv
dotenv.load_dotenv = lambda *a, **k: False
import logging
logging.basicConfig = lambda *a, **k: None
"""


def _linhas(saida: str) -> list[dict]:
    return [json.loads(l) for l in saida.splitlines() if l.strip()]


def test_emissao_sem_o_acidente(tmp_path):
    """Com o `basicConfig` do FastMCP morto, `import app.main` ainda emite INFO em stdout.

    Este é o teste da fatia inteira. Antes dela ele é vermelho: a raiz fica sem handler
    nenhum (`[]`, nível WARNING) e o INFO desaparece — o `lastResort` do Python só cobre
    WARNING pra cima, e em stderr. Ou seja, todo o log em INFO do webhook (o da W1 e o
    "entrega guardada" da W2a) dependia de um objeto de servidor MCP que o processo web
    nunca serve.

    A asserção é sobre STDOUT de propósito: o acidente escrevia em stderr, então exigir
    stdout separa "o log existe" de "o acidente ainda está de pé".
    """
    proc = _run_child(
        _PREAMBULO + """
import app.main
logging.getLogger("app.main").info("linha de prova")
""",
        tmp_path,
    )

    assert proc.returncode == 0, f"o filho morreu\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"

    linhas = _linhas(proc.stdout)
    assert len(linhas) == 1, (
        "esperava exatamente uma linha de log em stdout\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert linhas[0]["level"] == "INFO"
    assert linhas[0]["logger"] == "app.main"
    assert linhas[0]["message"] == "linha de prova"
    assert "linha de prova" not in proc.stderr, "o log saiu em stderr, não em stdout"


def test_log_level_warning_silencia_info(tmp_path):
    """`LOG_LEVEL` manda — e o WARNING é o CONTROLE.

    Sem a segunda metade, um `configure_logging()` que não emitisse nada passaria verde
    aqui: "nenhum INFO" e "nenhuma linha" são indistinguíveis olhando só o silêncio.
    """
    proc = _run_child(
        _PREAMBULO + """
import app.main
log = logging.getLogger("app.main")
log.info("nao deveria aparecer")
log.warning("deveria aparecer")
""",
        tmp_path,
        LOG_LEVEL="WARNING",
    )

    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    linhas = _linhas(proc.stdout)
    assert [l["message"] for l in linhas] == ["deveria aparecer"]
    assert linhas[0]["level"] == "WARNING"


def test_log_level_invalido_cai_no_default_e_avisa(tmp_path):
    """Valor inválido não pode derrubar o boot nem silenciar a saída.

    Mesmo fail-open validado do `app/limits.py::_limit_from_env`: um typo no env do Railway
    não pode virar exceção no import (o web process não subiria) nem log mudo — vira o
    default MAIS um aviso, que só existe porque é emitido DEPOIS do dictConfig.
    """
    proc = _run_child(
        _PREAMBULO + """
import app.main
logging.getLogger("app.main").info("o INFO continua saindo")
""",
        tmp_path,
        LOG_LEVEL="VERBOSO",
    )

    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    mensagens = [l["message"] for l in _linhas(proc.stdout)]
    assert any("VERBOSO" in m and "INFO" in m for m in mensagens), mensagens
    assert "o INFO continua saindo" in mensagens


def test_configure_logging_duas_vezes_nao_duplica(tmp_path):
    """Chamar de novo não empilha handler — senão cada linha sairia N vezes.

    A propriedade vem do próprio `dictConfig`, que remove os handlers do logger antes de
    instalar os configurados. É o mesmo mecanismo que faz a fatia vencer o `basicConfig`
    do FastMCP quando ele chega primeiro.
    """
    proc = _run_child(
        _PREAMBULO + """
import app.main
from app.logging_config import configure_logging
configure_logging()
configure_logging()
logging.getLogger("app.main").info("uma vez só")
""",
        tmp_path,
    )

    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert [l["message"] for l in _linhas(proc.stdout)] == ["uma vez só"]


# ---------------------------------------------------------------------------
# In-process: o handler CONFIGURADO, com o stream trocado por um StringIO.
#
# Não é reconstrução — é o mesmo objeto que roda em produção, com o mesmo formatter,
# o mesmo filtro e o mesmo nível. Reconstruir `JsonFormatter()` na mão no teste provaria
# que a classe funciona e não que ela está LIGADA.
# ---------------------------------------------------------------------------
@pytest.fixture
def linhas_json():
    """Desvia a saída do nosso handler e devolve um parser das linhas emitidas."""
    import app.main  # noqa: F401 — o import é o que dispara configure_logging()

    from app.logging_config import JsonFormatter

    nossos = [h for h in logging.getLogger().handlers if isinstance(h.formatter, JsonFormatter)]
    # Também é a asserção estrutural de que a configuração aconteceu, e UMA vez só.
    assert len(nossos) == 1, f"esperava exatamente um handler nosso na raiz, achei {nossos}"
    handler = nossos[0]

    original, handler.stream = handler.stream, io.StringIO()
    try:
        yield lambda: [
            json.loads(l) for l in handler.stream.getvalue().splitlines() if l.strip()
        ]
    finally:
        handler.stream = original


@pytest_asyncio.fixture
async def http_client(monkeypatch):
    """Client sem banco: /health e o 401 do /ask não tocam no Postgres."""
    monkeypatch.setenv("API_TOKENS", f"{CLIENTE}:{TOKEN}")
    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


def test_toda_linha_e_json_com_os_campos_minimos(linhas_json):
    """Uma linha JSON por registro, com os quatro campos que todo registro tem."""
    logging.getLogger("app.main").info("uma linha %s", "interpolada")
    logging.getLogger("app.rag.search").warning("outra linha")

    linhas = linhas_json()
    assert [l["message"] for l in linhas] == ["uma linha interpolada", "outra linha"]
    assert [l["level"] for l in linhas] == ["INFO", "WARNING"]
    assert [l["logger"] for l in linhas] == ["app.main", "app.rag.search"]
    for l in linhas:
        # UTC explícito: "14:02" sem fuso não cruza com o relógio de outra máquina, e o
        # container não tem timezone garantida.
        assert l["ts"].endswith("+00:00"), l["ts"]
        assert datetime.fromisoformat(l["ts"]).tzinfo is not None


def test_uma_linha_por_registro_mesmo_com_quebra_de_linha(linhas_json):
    """Mensagem multi-linha não vira dois registros — o `json.dumps` escapa o `\\n`.

    É o que permite a um coletor de log ler a saída linha a linha. Sem isso, um traceback
    de vinte frames viraria vinte "registros", dezenove deles JSON inválido.
    """
    logging.getLogger("app.main").info("primeira\nsegunda\nterceira")

    (linha,) = linhas_json()
    assert linha["message"] == "primeira\nsegunda\nterceira"


def test_fora_de_request_nao_ha_request_id(linhas_json):
    """Sem request, a chave é AUSENTE — não `null`, não a string "None".

    A passada de embedding e a extração offline rodam fora de qualquer request, e as
    colunas de `cost_event` correspondentes são nullable pelo mesmo motivo. Chave ausente
    diz isso; `"None"` é ruído que alguém acabaria filtrando como se fosse um id.
    """
    logging.getLogger("app.rag.embedding").info("passada offline")

    (linha,) = linhas_json()
    assert "request_id" not in linha
    assert "client" not in linha
    assert "None" not in json.dumps(linha)


async def test_o_header_cobre_health_o_401_e_o_429(http_client, monkeypatch):
    """`X-Request-Id` sai em toda resposta, inclusive nas que morrem antes da rota.

    É a prova de que o middleware é o MAIS EXTERNO: o 401 vem da dependency de auth e o
    429 do teto por IP viria do SlowAPIMiddleware — as duas respostas em que quem reclama
    não tem outra pista pra dar ao operador.
    """
    from app.main import HEADER_REQUEST_ID

    # O 429 é gerado pelo PRÓPRIO SlowAPIMiddleware, que monta a Response dele — caminho
    # diferente do 401 (que vem da dependency, por dentro da app) através do `send_com_id`.
    # É ele que a ordem dos `add_middleware` decide, então sem este caso dá pra inverter os
    # dois registros com a suíte verde.
    monkeypatch.setenv("ASK_RATE_LIMIT_IP", "1/minute")
    ruim = {"Authorization": "Bearer errado"}

    saude = await http_client.get("/health")
    negado = await http_client.post("/ask", json={"question": "oi tudo bem"}, headers=ruim)
    estourado = await http_client.post("/ask", json={"question": "oi tudo bem"}, headers=ruim)

    assert saude.status_code == 200 and saude.headers[HEADER_REQUEST_ID]
    assert negado.status_code == 401 and negado.headers[HEADER_REQUEST_ID]
    assert estourado.status_code == 429 and estourado.headers[HEADER_REQUEST_ID]
    # Um id por request, não um por processo.
    assert len({r.headers[HEADER_REQUEST_ID] for r in (saude, negado, estourado)}) == 3


async def test_request_id_liga_log_header_e_cost_event(
    cost_rows, ask_client, fake_graph, linhas_json
):
    """O MESMO valor nos três lugares — é isso que a correlação significa.

    O id nasce no middleware, volta pro usuário no `X-Request-Id`, aparece em toda linha
    de log do request e é o que o `/ask` carimba em `cost_event`. Cunhar um id novo dentro
    do `/ask` (como era antes) partiria em dois exatamente o que liga "o que este request
    custou" a "o que este request logou" — e o usuário não teria como nomear nenhum dos
    dois.
    """
    fake_graph(["sql_worker"])

    r = await ask_client.post(
        "/ask", json={"question": "Quantos perigos existem?"}, headers=AUTH
    )
    assert r.status_code == 200
    from app.main import HEADER_REQUEST_ID

    header = r.headers[HEADER_REQUEST_ID]
    assert header

    correlacionadas = [l for l in linhas_json() if "request_id" in l]
    assert correlacionadas, "nenhuma linha de log saiu correlacionada"
    assert {l["request_id"] for l in correlacionadas} == {header}
    # O `client` vem do mesmo ContextVar, e só existe depois da auth.
    assert {l["client"] for l in correlacionadas if l["logger"].startswith("app.agents")} == {
        CLIENTE
    }

    eventos = await _eventos(cost_rows)
    assert len(eventos) == 3                       # supervisor -> sql_worker -> synthesizer
    assert {e.request_id for e in eventos} == {header}


async def test_um_ask_nao_escreve_linha_crua_no_stdout(
    cost_rows, ask_client, fake_graph, linhas_json, capsys
):
    """Os `print()` do grafo viraram log — senão "stdout é um fluxo JSON" seria falso.

    Duas metades, e as duas importam: nada cru sai em stdout, E o conteúdo continua
    existindo (como registro correlacionado). Só a primeira passaria verde se alguém
    tivesse simplesmente APAGADO as linhas em vez de convertê-las.
    """
    fake_graph(["sql_worker"])
    capsys.readouterr()                            # descarta o que veio antes

    r = await ask_client.post(
        "/ask", json={"question": "Quantos perigos existem?"}, headers=AUTH
    )
    assert r.status_code == 200

    bruto = capsys.readouterr().out
    assert bruto == "", f"saiu linha crua em stdout:\n{bruto}"

    mensagens = [l["message"] for l in linhas_json()]
    for marca in ("[supervisor]", "[sql_worker]", "[synthesizer]"):
        assert any(m.startswith(marca) for m in mensagens), (marca, mensagens)


async def _erro_de_insert_com_pii():
    """Uma exceção de INSERT real, com os valores ligados visíveis no texto dela.

    Engine PRÓPRIA com `hide_parameters=False`: a garantia tem que ser do FORMATTER, e
    `hide_parameters=True` em `app/db.py` é defesa em profundidade — quem construir a
    sessão de outro jeito a perderia sem aviso. Mesma receita de
    `tests/test_whatsapp_inbox.py`.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    eng = create_async_engine(os.environ["DATABASE_URL"], hide_parameters=False)
    try:
        async with async_sessionmaker(eng)() as s:
            try:
                await s.execute(
                    text(
                        "INSERT INTO whatsapp_message (wamid, from_phone, text, nao_existe) "
                        "VALUES (:w, :p, :t, 1)"
                    ).bindparams(w=WAMID, p=TELEFONE, t=TEXTO)
                )
            except BaseException as exc:
                return exc
    finally:
        await eng.dispose()
    raise AssertionError("o INSERT não falhou")


async def test_excecao_leva_tipo_e_frames_e_nunca_a_mensagem(db_url, linhas_json):
    """`logger.exception` rende tipo + frames, e NADA do texto da exceção.

    Regressão do achado da W2a: o texto de uma exceção do SQLAlchemy inclui
    `[parameters: (...)]` — o telefone e a mensagem inteira do usuário —, e é exatamente
    ele que `format_exception` imprimiria. `format_tb` formata só os frames.

    A asserção principal é uma AUSÊNCIA, e é conferida palavra por palavra: uma versão que
    truncasse a mensagem da exceção continuaria vazando host, usuário e o começo do texto.
    """
    erro = await _erro_de_insert_com_pii()
    # Pré-condição: o teste só prova algo se a exceção de fato contiver a PII.
    assert TELEFONE in str(erro) and TEXTO in str(erro), "o caso ruim não foi reproduzido"

    try:
        raise erro
    except BaseException:
        logging.getLogger("app.main").exception("falha ao gravar a entrega")

    (linha,) = linhas_json()
    bruto = json.dumps(linha, ensure_ascii=False)

    # O operador continua servido: qual erro, e onde.
    assert linha["exc_type"] == "sqlalchemy.exc.ProgrammingError", linha["exc_type"]
    assert "test_logging.py" in linha["exc_traceback"]
    assert linha["message"] == "falha ao gravar a entrega"

    assert "[parameters:" not in bruto
    assert "[SQL:" not in bruto
    for pedaco in (TELEFONE, WAMID, TEXTO, *TEXTO.split()):
        assert pedaco not in bruto, f"{pedaco!r} vazou pro log"


def test_debug_e_so_do_nosso_codigo(tmp_path):
    """`LOG_LEVEL=DEBUG` não pode ligar o DEBUG das bibliotecas — é ali que a PII entra.

    O `anthropic` emite `log.debug("Request options: %s", ...)` com o array `messages`
    inteiro: a pergunta literal do usuário e cada cláusula recuperada. Isso passaria pelo
    campo `message`, que é justamente o campo em que a lista fechada não consegue revisar
    nada. Então a raiz nunca desce abaixo de INFO, e quem desce é o logger `app`.
    """
    proc = _run_child(
        _PREAMBULO + """
import app.main, json
nomes = ["app.main", "app.agents.graph", "anthropic", "httpx", "httpcore", "voyageai"]
print(json.dumps({n: logging.getLevelName(logging.getLogger(n).getEffectiveLevel())
                  for n in nomes}))
""",
        tmp_path,
        LOG_LEVEL="DEBUG",
    )

    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    niveis = json.loads(proc.stdout.splitlines()[-1])
    assert niveis["app.main"] == "DEBUG" and niveis["app.agents.graph"] == "DEBUG"
    for terceiro in ("anthropic", "httpx", "httpcore", "voyageai"):
        assert niveis[terceiro] == "INFO", (terceiro, niveis)


def test_apertar_o_nivel_vale_pra_todo_mundo(tmp_path):
    """O piso é só na descida: `LOG_LEVEL=WARNING` cala terceiro também.

    Calar não vaza nada, então o piso de INFO da raiz não pode virar um chão que impeça o
    operador de silenciar a saída — que é o uso mais comum da variável.
    """
    proc = _run_child(
        _PREAMBULO + """
import app.main, json
print(json.dumps(logging.getLevelName(logging.getLogger("httpx").getEffectiveLevel())))
""",
        tmp_path,
        LOG_LEVEL="WARNING",
    )

    assert proc.returncode == 0, f"stderr:\n{proc.stderr}"
    assert json.loads(proc.stdout.splitlines()[-1]) == "WARNING"


def test_o_verify_token_do_whatsapp_e_redigido(linhas_json):
    """O log de acesso do uvicorn imprime a query string CRUA — e o handshake leva o segredo.

    `app/main.py` tira esse token do log dele de propósito ("um deles é segredo"), então
    adotar o `uvicorn.access` neste handler sem redigir traria de volta, por outra porta, o
    único segredo do sistema adivinhável por tentativa.
    """
    logging.getLogger("uvicorn.access").info(
        '127.0.0.1:1 - "GET /webhook/whatsapp?hub.mode=subscribe'
        '&hub.verify_token=%s&hub.challenge=9 HTTP/1.1" 200',
        "segredo-que-nao-pode-vazar",
    )

    (linha,) = linhas_json()
    assert "segredo-que-nao-pode-vazar" not in json.dumps(linha)
    assert "hub.verify_token=<redigido>" in linha["message"]
    # O resto da linha continua servindo o operador.
    assert "hub.challenge=9" in linha["message"]


def test_a_cadeia_da_excecao_entra_sem_a_mensagem(linhas_json):
    """Quem CAUSOU o erro tem tipo e frames no log — e continua sem texto de exceção.

    `format_tb` sozinho anda só no traceback mais externo: um `DBAPIError` embrulhando um
    erro do asyncpg reportaria os frames do SQLAlchemy e nada sobre o erro do driver, que é
    a identidade em disputa quando `_vale_reentregar` classifica errado.
    """
    # As mensagens são montadas em RUNTIME, e isso é o ponto do teste, não estilo:
    # `format_tb` inclui a LINHA DE FONTE de cada frame, então uma exceção cujo texto é um
    # literal no código apareceria no log pelo fonte — e o teste "provaria" um vazamento
    # que não existe. PII de verdade nasce em runtime, como aqui.
    causa_txt = "".join(["mensagem", "-da-causa"])
    embrulho_txt = "".join(["mensagem", "-do-embrulho"])

    try:
        try:
            raise ValueError(causa_txt)
        except ValueError as causa:
            raise RuntimeError(embrulho_txt) from causa
    except RuntimeError:
        logging.getLogger("app.main").exception("falhou")

    (linha,) = linhas_json()
    bruto = json.dumps(linha)

    assert linha["exc_type"] == "RuntimeError"          # o mais externo nomeia a linha
    assert "causado por ValueError" in linha["exc_traceback"]
    assert causa_txt not in bruto
    assert embrulho_txt not in bruto


def test_from_None_suprime_o_contexto_no_log_tambem(linhas_json):
    """`raise ... from None` é uma decisão de quem escreveu — o log não a desfaz."""
    try:
        try:
            raise ValueError("contexto-suprimido")
        except ValueError:
            raise RuntimeError("o que importa") from None
    except RuntimeError:
        logging.getLogger("app.main").exception("falhou")

    (linha,) = linhas_json()
    assert "causado por" not in linha["exc_traceback"]


async def test_excecao_nao_tratada_deixa_um_registro_correlacionado(
    http_client, linhas_json, monkeypatch
):
    """O 500 sai sem header (o ServerErrorMiddleware é mais externo) — mas COM correlação.

    Sem o `except` do middleware, o único registro do crash seria o
    "Exception in ASGI application" do uvicorn, emitido depois do reset do ContextVar e
    portanto sem `request_id`: a resposta que mais precisa ser rastreável seria a única sem
    header E sem id.
    """
    import app.main as main_mod

    class _GrafoQueExplode:
        async def ainvoke(self, _state):
            raise RuntimeError("boom")

    # O objeto `graph` é global do módulo e a rota o lê por nome, então trocar o atributo
    # basta — e a exceção sobe pela rota inteira, que é o caminho que se quer exercitar.
    monkeypatch.setattr(main_mod, "graph", _GrafoQueExplode())

    with pytest.raises(RuntimeError):
        await http_client.post("/ask", json={"question": "uma pergunta"}, headers=AUTH)

    falhas = [l for l in linhas_json() if l["level"] == "ERROR"]
    assert len(falhas) == 1, falhas
    assert falhas[0]["request_id"]
    assert falhas[0]["exc_type"] == "RuntimeError"
    assert "/ask" in falhas[0]["message"]


def test_client_vazio_ainda_e_um_cliente(linhas_json):
    """`client` falsy é um valor REAL, e omiti-lo mentiria sobre a origem da chamada.

    `API_TOKENS` é `"nome:token"`, então uma entrada malformada (`":tok"`) produz um nome
    vazio. Com truthiness a chave sumiria — e a ausência dela significa, por contrato,
    "isto rodou FORA de um request", o que jogaria as chamadas pagas desse cliente no balde
    do offline.
    """
    from app.agents.context import reset_request_context, set_request_context

    ctx = set_request_context("id-do-request", "")
    try:
        logging.getLogger("app.main").info("chamada paga")
    finally:
        reset_request_context(ctx)

    (linha,) = linhas_json()
    assert linha["request_id"] == "id-do-request"
    assert linha["client"] == ""


def test_as_anomalias_do_route_sao_WARNING(linhas_json):
    """Circuit breaker e fail-closed são ANOMALIA, não trace — e `LOG_LEVEL=WARNING` existe.

    Como `print`, essas duas linhas eram incondicionais. Em INFO, um deploy silenciado
    calaria as duas, e de fora o `NO_ANSWER` resultante é indistinguível de falha de infra —
    sem nada no log pra desempatar. O irmão delas (o supervisor devolvendo "END") já loga em
    WARNING pelo mesmo motivo.
    """
    from app.agents.graph import route

    assert route({"iterations": 99, "next": "sql_worker", "messages": []})
    assert route({"iterations": 0, "next": "valor-invalido", "messages": []})

    niveis = {l["message"]: l["level"] for l in linhas_json()}
    assert len(niveis) == 2, niveis
    assert set(niveis.values()) == {"WARNING"}, niveis
