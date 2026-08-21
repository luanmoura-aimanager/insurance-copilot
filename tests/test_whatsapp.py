"""
Testes da borda do WhatsApp (fatia W1): handshake, assinatura HMAC e rate limit.

Sem rede e sem banco — o webhook não toca no Postgres, então este módulo não pede a
fixture `client` do conftest (aquela sobe container). Nenhum teste gasta dinheiro:
não há chamada de LLM nem de embedding neste caminho.
"""
import base64
import hashlib
import hmac
import json
import logging

import pytest
from httpx import ASGITransport, AsyncClient

APP_SECRET = "segredo-de-app-do-teste"
VERIFY_TOKEN = "token-de-verificacao-do-teste"

TELEFONE = "5511999998888"
WAMID = "wamid.HBgNNTUxMTk5OTk5ODg4OA=="
TEXTO = "Quais seguradoras cobrem vendaval?"

PAYLOAD = {
    "object": "whatsapp_business_account",
    "entry": [
        {
            "id": "1234567890",
            "changes": [
                {
                    "field": "messages",
                    "value": {
                        "messaging_product": "whatsapp",
                        "metadata": {"display_phone_number": "551133334444",
                                     "phone_number_id": "999"},
                        "contacts": [{"profile": {"name": "Fulana"}, "wa_id": TELEFONE}],
                        "messages": [
                            {
                                "from": TELEFONE,
                                "id": WAMID,
                                "timestamp": "1771000000",
                                "type": "text",
                                "text": {"body": TEXTO},
                            }
                        ],
                    },
                }
            ],
        }
    ],
}


def corpo(payload=PAYLOAD) -> bytes:
    """Bytes exatos do corpo. A assinatura é sobre ELES, não sobre o dict."""
    return json.dumps(payload).encode("utf-8")


def assinar(body: bytes, secret: str = APP_SECRET) -> dict[str, str]:
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return {"X-Hub-Signature-256": f"sha256={digest}"}


@pytest.fixture
def webhook_client(monkeypatch):
    """Factory (não client): o teste aperta o env ANTES de abrir a conexão.

    Mesmo formato do `protected_client` de tests/test_auth.py, e pelo mesmo motivo —
    os limites e os segredos são lidos a cada request, então basta o env estar no
    lugar na hora da chamada. Limites folgados por padrão; quem testa limite aperta.
    """
    monkeypatch.setenv("WHATSAPP_APP_SECRET", APP_SECRET)
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", VERIFY_TOKEN)
    monkeypatch.setenv("WHATSAPP_RATE_LIMIT", "100/minute")
    monkeypatch.setenv("ASK_RATE_LIMIT_IP", "100/minute")
    # Só pro CONTROLE de test_teto_por_ip_nao_alcanca_o_webhook: sem isto o /ask sai
    # 500 (fail closed do API_TOKENS) em vez de 401, e o controle passaria a medir o
    # caminho errado antes de chegar no 429.
    monkeypatch.setenv("API_TOKENS", "controle:token-do-controle")

    from app.main import app

    async def _make():
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    return _make


@pytest.fixture
async def client(webhook_client):
    async with await webhook_client() as c:
        yield c


# --- handshake de verificação (GET) -----------------------------------------


async def test_get_com_token_certo_devolve_o_challenge(client):
    """A Meta só cadastra a URL se o challenge voltar cru, em text/plain."""
    r = await client.get(
        "/webhook/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": VERIFY_TOKEN,
                "hub.challenge": "1158201444"},
    )

    assert r.status_code == 200
    assert r.text == "1158201444"
    assert r.headers["content-type"].startswith("text/plain")


async def test_get_com_token_errado_403_e_nao_ecoa_o_challenge(client):
    """403 — e o challenge NÃO volta no corpo.

    A segunda metade é a que importa: devolver o eco junto do 403 faria da rota um
    refletor gratuito pra quem não tem o token nenhum.
    """
    r = await client.get(
        "/webhook/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": "nao-e-esse",
                "hub.challenge": "1158201444"},
    )

    assert r.status_code == 403
    assert "1158201444" not in r.text


async def test_get_sem_mode_subscribe_403(client):
    """Token certo não basta: o handshake é `hub.mode=subscribe`."""
    r = await client.get(
        "/webhook/whatsapp",
        params={"hub.mode": "unsubscribe", "hub.verify_token": VERIFY_TOKEN,
                "hub.challenge": "1158201444"},
    )

    assert r.status_code == 403


async def test_challenge_vazio_403(client):
    """`hub.challenge=` passava pelo `is None` e saía 200 com corpo vazio.

    A Meta compara o corpo com o challenge que mandou: um 200 errado é handshake FALHO
    reportado como sucesso, e mais difícil de depurar que um 403.
    """
    r = await client.get(
        "/webhook/whatsapp",
        params={"hub.mode": "subscribe", "hub.verify_token": VERIFY_TOKEN, "hub.challenge": ""},
    )

    assert r.status_code == 403


async def test_get_sem_verify_token_403(client):
    r = await client.get("/webhook/whatsapp", params={"hub.mode": "subscribe",
                                                      "hub.challenge": "1158201444"})

    assert r.status_code == 403


async def test_verify_token_ausente_no_ambiente_500(monkeypatch, webhook_client):
    """Config ausente é 500, não 403 — mesma regra do API_TOKENS."""
    monkeypatch.delenv("WHATSAPP_VERIFY_TOKEN", raising=False)

    async with await webhook_client() as c:
        r = await c.get("/webhook/whatsapp", params={"hub.mode": "subscribe",
                                                     "hub.verify_token": VERIFY_TOKEN,
                                                     "hub.challenge": "1"})

    assert r.status_code == 500


async def test_verify_token_ausente_sem_challenge_tambem_500(monkeypatch, webhook_client):
    """500 mesmo SEM `hub.challenge` — é o que fixa a ORDEM das checagens no GET.

    `verify_challenge` é a única chamada que lê configuração; com o teste do challenge
    antes dela, o `or` curto-circuitava e este request saía 403, escondendo um deploy
    quebrado atrás de "a Meta mandou errado". Gêmeo do teste do app secret sem header.
    """
    monkeypatch.delenv("WHATSAPP_VERIFY_TOKEN", raising=False)

    async with await webhook_client() as c:
        r = await c.get("/webhook/whatsapp", params={"hub.mode": "subscribe",
                                                     "hub.verify_token": VERIFY_TOKEN})

    assert r.status_code == 500


# --- assinatura (POST) ------------------------------------------------------


async def test_post_com_assinatura_valida_200(client):
    body = corpo()
    r = await client.post("/webhook/whatsapp", content=body, headers=assinar(body))

    assert r.status_code == 200


async def test_corpo_alterado_em_um_byte_403(client):
    """A assinatura cobre a MENSAGEM, não só o remetente.

    Este é o teste que separa "veio da Meta" de "veio da Meta e chegou íntegro": um
    byte trocado no corpo, com a assinatura original, tem que ser recusado. Sem ele,
    uma implementação que só conferisse o formato do header passaria verde.
    """
    body = corpo()
    headers = assinar(body)
    adulterado = body.replace(b"vendaval", b"vendavaL")
    assert adulterado != body and len(adulterado) == len(body)

    r = await client.post("/webhook/whatsapp", content=adulterado, headers=headers)

    assert r.status_code == 403


async def test_sem_header_de_assinatura_403(client):
    body = corpo()
    r = await client.post("/webhook/whatsapp", content=body)

    assert r.status_code == 403


@pytest.mark.parametrize(
    "header",
    [
        "abc",                                    # sem prefixo nenhum
        "sha1=" + "a" * 40,                       # o header legado, que não aceitamos
        "sha256=nao-e-hexadecimal",               # prefixo certo, hex inválido
        "sha256=",                                # prefixo e nada
        "sha256=" + "a" * 64,                     # hex bem formado, digest errado
        "sha256=çç",                              # NÃO-ASCII: o caso do compare_digest
    ],
)
async def test_header_malformado_403(client, header):
    """Header malformado é 403, nunca 500 — inclusive o não-ASCII.

    O último caso é o load-bearing: comparar as strings hex com `compare_digest`
    estouraria `TypeError` num valor escolhido pelo cliente, e o 403 viraria 500.
    É `bytes.fromhex` que fecha isso (mesmo buraco que app/auth.py fechou no Bearer).

    O header vai em bytes crus porque o httpx se recusa a codificar não-ASCII; o
    Starlette decodifica como latin-1, que é o que um cliente de verdade produziria.
    Mesmo arranjo de `test_token_nao_ascii_401` em tests/test_auth.py.
    """
    body = corpo()
    r = await client.post("/webhook/whatsapp", content=body,
                          headers={b"X-Hub-Signature-256": header.encode("latin-1")})

    assert r.status_code == 403


async def test_app_secret_ausente_no_ambiente_500(monkeypatch, webhook_client):
    """Segredo ausente é 500 mesmo SEM header de assinatura — e é o "sem header" que
    faz o teste valer.

    É o que fixa a ORDEM dentro de `verify_signature`: o segredo é lido antes de olhar
    o header. Invertida, este request sairia 403 ("a Meta mandou errado") escondendo
    um deploy quebrado, que é o disfarce que o 500 do API_TOKENS existe pra impedir.
    """
    monkeypatch.delenv("WHATSAPP_APP_SECRET", raising=False)

    async with await webhook_client() as c:
        r = await c.post("/webhook/whatsapp", content=corpo())

    assert r.status_code == 500


# --- leitura do payload -----------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        {"object": "whatsapp_business_account", "entry": [{"changes": [{"value": {
            "statuses": [{"id": WAMID, "status": "delivered"}]}}]}]},   # não é mensagem
        {"foo": "bar"},
        {"entry": "isto devia ser lista"},
        {"entry": [{"changes": [{"value": {"messages": [{"sem": "id"}]}}]}]},
        [],
        {},
    ],
)
async def test_payload_de_formato_inesperado_continua_200(client, caplog, payload):
    """Formato inesperado vira 200 silencioso, nunca exceção.

    A Meta manda no MESMO endpoint eventos que não são mensagem (os `statuses` de
    entrega/leitura) e adiciona campos sem aviso. Um não-2xx aqui faria ela retentar
    e, insistindo, desabilitar a inscrição.
    """
    body = corpo(payload)
    with caplog.at_level(logging.DEBUG, logger="app.main"):
        r = await client.post("/webhook/whatsapp", content=body, headers=assinar(body))

    assert r.status_code == 200
    assert [rec for rec in caplog.records if rec.levelno >= logging.ERROR] == []


async def test_mensagem_sem_identificacao_e_descartada():
    """`{"id": "", "from": ""}` não é mensagem identificável.

    `isinstance("", str)` é True, então a guarda por tipo deixava passar — e um wamid
    vazio colide com todos os outros na deduplicação que a W2 precisa, enquanto um
    telefone vazio vira "***" no log, igualzinho a um número mascarado de verdade.
    """
    from app.whatsapp import extract_messages

    payload = {"entry": [{"changes": [{"value": {"messages": [
        {"id": "", "from": "", "type": "text", "text": {"body": "oi"}}]}}]}]}

    assert extract_messages(payload) == []


async def test_corpo_grande_demais_e_recusado_sem_ler(client):
    """A app não bufferiza o que um anônimo mandar.

    O HMAC só pode ser calculado depois de ler os bytes, então até o teto ser
    verificado quem manda é um desconhecido — é o único ponto do projeto em que isso
    acontece, porque todo o resto está atrás do Bearer.
    """
    from app.whatsapp import MAX_BODY_BYTES

    gigante = b"x" * (MAX_BODY_BYTES + 1)
    r = await client.post("/webhook/whatsapp", content=gigante, headers=assinar(gigante))

    assert r.status_code == 413


async def test_uma_mensagem_ruim_nao_engole_as_seguintes(client, caplog, monkeypatch):
    """A Meta entrega em LOTE, e respondemos 200 — então ela nunca reentrega.

    Um try em volta do laço faria a falha na mensagem N descartar as N+1.. de vez.
    Aqui a 2ª de três explode no `mascarar_telefone`, e a 3ª ainda tem que ser logada.
    """
    import app.main as main_mod

    real = main_mod.mascarar_telefone

    def explode(numero: str) -> str:
        if numero.endswith("2"):
            raise RuntimeError("falha ao mascarar")
        return real(numero)

    monkeypatch.setattr(main_mod, "mascarar_telefone", explode)

    payload = {"entry": [{"changes": [{"value": {"messages": [
        {"id": f"wamid.{n}", "from": f"551199999000{n}", "type": "text", "text": {"body": "oi"}}
        for n in (1, 2, 3)
    ]}}]}]}
    body = corpo(payload)

    with caplog.at_level(logging.INFO, logger="app.main"):
        r = await client.post("/webhook/whatsapp", content=body, headers=assinar(body))

    from app.whatsapp import id_curto

    registrado = "\n".join(rec.getMessage() for rec in caplog.records)
    assert r.status_code == 200
    assert id_curto("wamid.1") in registrado
    assert id_curto("wamid.3") in registrado          # a que viria DEPOIS da falha
    assert f"falha ao registrar mensagem (id={id_curto('wamid.2')})" in registrado


async def test_corpo_que_nao_e_json_continua_200(client, caplog):
    """Assinatura válida sobre um corpo que não é JSON: 200, e o erro fica no log."""
    body = b"nao sou json"
    with caplog.at_level(logging.DEBUG, logger="app.main"):
        r = await client.post("/webhook/whatsapp", content=body, headers=assinar(body))

    assert r.status_code == 200
    assert any("payload não interpretado" in rec.message for rec in caplog.records)


async def test_mensagem_de_texto_e_logada_sem_pii(client, caplog):
    """O log identifica a mensagem, e NÃO carrega o texto nem o telefone inteiro.

    A ausência é a asserção principal: telefone e conteúdo são PII e o log fica retido
    na plataforma. O que o operador precisa é reconhecer o mesmo remetente entre duas
    linhas (daí os 4 últimos dígitos) e saber que a mensagem chegou — quem prova que o
    parse funciona é esta suíte, não o log de produção.
    """
    body = corpo()
    with caplog.at_level(logging.INFO, logger="app.main"):
        r = await client.post("/webhook/whatsapp", content=body, headers=assinar(body))

    from app.whatsapp import id_curto

    assert r.status_code == 200
    linhas = [rec.getMessage() for rec in caplog.records if rec.levelno == logging.INFO]
    inteiro = "\n".join(linhas)

    # Identifica a mensagem por um DIGEST, não pelo wamid — ver abaixo.
    assert id_curto(WAMID) in inteiro
    assert "8888" in inteiro          # os 4 últimos, que é o que se mascara PARA
    assert str(len(TEXTO)) in inteiro

    # E não carrega nada de PII. A busca é palavra por palavra porque uma versão que
    # truncasse o texto continuaria vazando o começo dele.
    assert TEXTO not in inteiro
    for palavra in TEXTO.split():
        assert palavra not in inteiro
    assert TELEFONE not in inteiro

    # A metade que faltava, e que fazia este teste ser CEGO por construção: o wamid
    # EMBUTE o telefone em base64, então logá-lo inteiro publicaria, reversível, o
    # mesmo número que `mascarar_telefone` acabou de esconder — e um `TELEFONE not in
    # inteiro` nunca perceberia, porque a busca é por substring e o número está
    # codificado. A asserção tem que ser sobre o wamid CRU.
    assert TELEFONE.encode() in base64.b64decode(WAMID.split(".", 1)[1])  # a premissa
    assert WAMID not in inteiro


async def test_phone_number_id_chega_junto():
    """O número que RECEBEU, não só o que mandou.

    Um app secret da Meta cobre todos os números da conta e o envio é
    `POST /{phone_number_id}/messages`; sem carregar isso na borda, a W2 fixaria um
    número no código — errado em silêncio no dia em que a conta ganhar o segundo.
    """
    from app.whatsapp import extract_messages

    (msg,) = extract_messages(PAYLOAD)

    assert msg.phone_number_id == "999"
    assert extract_messages({"entry": [{"changes": [{"value": {"messages": [
        {"id": "w", "from": "5511", "type": "text"}]}}]}]})[0].phone_number_id is None


@pytest.mark.parametrize("valor", ["²", "nao-e-numero", "-1", ""])
async def test_content_length_invalido_nao_vira_500(client, valor):
    """`"²".isdigit()` é True e `int("²")` levanta — 500 com traceback, antes do HMAC.

    Alcançável por qualquer anônimo, na rota pública, antes de qualquer verificação.
    """
    body = corpo()
    r = await client.post("/webhook/whatsapp", content=body,
                          headers={**assinar(body), b"Content-Length": valor.encode("latin-1")})

    assert r.status_code in (200, 400, 403, 413)   # o que não pode é 500


async def test_desconexao_no_meio_do_corpo_nao_vira_500():
    """`request.stream()` levanta ClientDisconnect quando o cliente aborta.

    Nada tratava isso: 500 e traceback por request, na única rota cujo contrato é
    nunca responder não-2xx — e se for o cliente da própria Meta que expirou, o 500 é
    o que empurra ela pra desabilitar a inscrição.
    """
    from app.main import _corpo_com_teto
    from starlette.requests import Request

    pedacos = iter([{"type": "http.request", "body": b"parcial", "more_body": True},
                    {"type": "http.disconnect"}])

    async def receive():
        return next(pedacos)

    req = Request({"type": "http", "method": "POST", "path": "/webhook/whatsapp",
                   "headers": []}, receive=receive)

    assert await _corpo_com_teto(req) == b""      # corpo incompleto = corpo vazio


async def test_corpo_continua_legivel_depois_da_leitura_com_teto():
    """Quem vier depois pode chamar `request.body()` — a W2 vai chamar.

    `Request.stream()` marca o stream como consumido sem popular `_body`, então sem o
    cache um `await request.body()` posterior levantaria `RuntimeError: Stream
    consumed` — numa rota documentada como sempre-200.
    """
    from app.main import _corpo_com_teto
    from starlette.requests import Request

    pedacos = iter([{"type": "http.request", "body": b"oi", "more_body": False}])

    async def receive():
        return next(pedacos)

    req = Request({"type": "http", "method": "POST", "path": "/webhook/whatsapp",
                   "headers": []}, receive=receive)

    assert await _corpo_com_teto(req) == b"oi"
    assert await req.body() == b"oi"


async def test_placeholder_do_env_example_conta_como_ausente(monkeypatch, webhook_client):
    """`CHANGE_ME` é truthy e passava — um `cp .env.example .env` deixaria de pé um
    webhook público cujo segredo está impresso no repositório."""
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "CHANGE_ME")

    async with await webhook_client() as c:
        r = await c.post("/webhook/whatsapp", content=corpo())

    assert r.status_code == 500


async def test_o_500_nao_diz_qual_variavel_falta(monkeypatch, webhook_client):
    """A rota é pública e sem auth: o corpo da resposta não entrega o schema de
    configuração do deploy nem avisa que ele está meio provisionado agora.

    O nome da variável fica no log, que é o canal do operador.
    """
    monkeypatch.delenv("WHATSAPP_APP_SECRET", raising=False)

    async with await webhook_client() as c:
        r = await c.post("/webhook/whatsapp", content=corpo())

    assert r.status_code == 500
    assert "WHATSAPP_APP_SECRET" not in r.text


# --- rate limit -------------------------------------------------------------


async def test_teto_por_ip_nao_alcanca_o_webhook(monkeypatch, webhook_client):
    """O teto por IP não pode estrangular a entrega da Meta.

    Com `default_limits` em 2/minute, 5 POSTs assinados têm que passar. O CONTROLE no
    fim é obrigatório: sem ele, um limiter desligado (ou um `reset` mal colocado)
    faria este teste passar verde sem provar nada — é o /ask saindo 429 no mesmo
    cliente que prova que o teto estava de fato ativo.
    """
    monkeypatch.setenv("ASK_RATE_LIMIT_IP", "2/minute")
    body = corpo()

    async with await webhook_client() as c:
        for _ in range(5):
            r = await c.post("/webhook/whatsapp", content=body, headers=assinar(body))
            assert r.status_code == 200

        pergunta = {"question": "quantos perigos existem?"}
        assert (await c.post("/ask", json=pergunta)).status_code == 401
        assert (await c.post("/ask", json=pergunta)).status_code == 401
        controle = await c.post("/ask", json=pergunta)

    assert controle.status_code == 429


async def test_teto_proprio_do_webhook_bloqueia(monkeypatch, webhook_client):
    """Escapar do teto por IP não é ficar sem teto: a rota é pública."""
    monkeypatch.setenv("WHATSAPP_RATE_LIMIT", "2/minute")
    body = corpo()

    async with await webhook_client() as c:
        assert (await c.post("/webhook/whatsapp", content=body,
                             headers=assinar(body))).status_code == 200
        assert (await c.post("/webhook/whatsapp", content=body,
                             headers=assinar(body))).status_code == 200

        r = await c.post("/webhook/whatsapp", content=body, headers=assinar(body))

    assert r.status_code == 429


async def test_handshake_continua_sob_o_teto_por_ip(monkeypatch, webhook_client):
    """O espelho, no GET, do `test_teto_por_ip_cobre_request_sem_auth` do /ask.

    O handshake é o único ponto do sistema onde um segredo pode ser ADIVINHADO por
    tentativa, e a decisão foi deixá-lo coberto pelo teto por IP — mas isso só vale
    porque GET e POST são handlers SEPARADOS: `_should_exempt` chaveia pelo nome do
    handler, então fundir os dois num `@app.api_route(methods=["GET","POST"])`, ou pôr
    qualquer `@limiter.limit` estático no `whatsapp_verify`, isentaria o handshake em
    silêncio, com a suíte inteira verde.
    """
    monkeypatch.setenv("ASK_RATE_LIMIT_IP", "2/minute")
    params = {"hub.mode": "subscribe", "hub.verify_token": "chute", "hub.challenge": "1"}

    async with await webhook_client() as c:
        assert (await c.get("/webhook/whatsapp", params=params)).status_code == 403
        assert (await c.get("/webhook/whatsapp", params=params)).status_code == 403

        r = await c.get("/webhook/whatsapp", params=params)

    assert r.status_code == 429


async def test_handshake_estourado_nao_derruba_a_entrega_da_meta(monkeypatch, webhook_client):
    """Os dois baldes são separados — e antes do `per_method` NÃO eram.

    O slowapi chaveia o balde pelo PATH e o valor do limite entra na chave, então com
    `ASK_RATE_LIMIT_IP == WHATSAPP_RATE_LIMIT` (o fixture põe os dois em 100/minute) o
    balde do GET e o do POST ficavam idênticos. Reproduzido com os dois em 5/minute:
    5 handshakes com token errado, de qualquer anônimo, faziam o POST assinado seguinte
    da Meta sair 429 — e a Meta desabilita a inscrição depois de falhas repetidas.
    """
    monkeypatch.setenv("ASK_RATE_LIMIT_IP", "2/minute")
    monkeypatch.setenv("WHATSAPP_RATE_LIMIT", "2/minute")
    params = {"hub.mode": "subscribe", "hub.verify_token": "chute", "hub.challenge": "1"}
    body = corpo()

    async with await webhook_client() as c:
        for _ in range(4):
            await c.get("/webhook/whatsapp", params=params)

        r = await c.post("/webhook/whatsapp", content=body, headers=assinar(body))

    assert r.status_code == 200


async def test_o_env_do_webhook_so_APERTA(monkeypatch, webhook_client):
    """A âncora é AVALIADA, não só registrada — e é isso que a documentação promete.

    `.env.example`, README e o docstring de `whatsapp_limit` dizem que
    `WHATSAPP_RATE_LIMIT` só aperta. Registrar a âncora (o teste abaixo) não prova
    isso: trocar a ordem dos dois decorators, passar `override_defaults=False`, ou uma
    mudança do slowapi em como `_route_limits` e `_dynamic_route_limits` se juntam,
    tiraria a âncora da avaliação com os dois testes vizinhos verdes — e o teto que um
    typo de env não podia levantar passaria a ser levantável sem limite.
    """
    monkeypatch.setenv("WHATSAPP_RATE_LIMIT", "100000/minute")
    body = corpo()
    headers = assinar(body)

    async with await webhook_client() as c:
        for _ in range(600):
            assert (await c.post("/webhook/whatsapp", content=body,
                                 headers=headers)).status_code == 200

        r = await c.post("/webhook/whatsapp", content=body, headers=headers)

    assert r.status_code == 429


async def test_a_ancora_estatica_esta_registrada():
    """Nomeia o mecanismo que faz o teste acima passar.

    `_should_exempt` (slowapi/middleware.py) tira a rota do `default_limits` olhando
    `limiter._route_limits`, e só limites ESTÁTICOS chegam lá. É atributo PRIVADO de
    uma dependência pinada: se um upgrade do slowapi renomeá-lo, é aqui que se
    descobre, em vez de o teto por IP voltar a cobrir o webhook em silêncio.
    """
    from app.limits import limiter
    from app.main import whatsapp_webhook

    nome = f"{whatsapp_webhook.__module__}.{whatsapp_webhook.__name__}"

    assert nome in limiter._route_limits
    assert nome in limiter._dynamic_route_limits      # o callable do env, em paralelo
