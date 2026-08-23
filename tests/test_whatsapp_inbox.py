"""A aceitação DURÁVEL do webhook (fatia W2a): a mensagem vira linha em
`whatsapp_message`, deduplicada por `wamid`.

Este módulo sobe container, ao contrário de `tests/test_whatsapp.py` — e essa separação
é o desenho, não acidente. Lá a sessão é falsa e o que se prova é a borda (assinatura,
parse, limites, PII); aqui a sessão é REAL E COMMITA, e o que se prova é que o dado
sobrevive ao request.

**Toda asserção lê por uma sessão SEPARADA**, e é isso que torna o commit da rota
load-bearing: `get_session` não commita (app/db.py), então sem o `await session.commit()`
explícito a linha seria descartada no fim do request e nenhuma outra conexão a veria.
Verificado por mutação — removendo o commit, `test_entrega_valida_grava_uma_linha_pending`
fica vermelho enquanto o resto da suíte segue verde.

Nenhum teste gasta dinheiro: o webhook desta fatia não chama LLM nem embedding.
"""
import json
import logging
import os

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.models import WhatsAppMessage
from app.whatsapp import id_curto
from tests.test_whatsapp import APP_SECRET, TELEFONE, TEXTO, WAMID, assinar

PHONE_NUMBER_ID = "999"


def payload_com(mensagens: list[dict], phone_number_id: str | None = PHONE_NUMBER_ID) -> dict:
    """Um envelope da Meta em volta das mensagens dadas."""
    value: dict = {"messaging_product": "whatsapp", "messages": mensagens}
    if phone_number_id is not None:
        value["metadata"] = {"phone_number_id": phone_number_id}
    return {
        "object": "whatsapp_business_account",
        "entry": [{"id": "1234567890", "changes": [{"field": "messages", "value": value}]}],
    }


def msg_texto(wamid: str = WAMID, texto: str = TEXTO, de: str = TELEFONE) -> dict:
    return {"from": de, "id": wamid, "timestamp": "1771000000",
            "type": "text", "text": {"body": texto}}


@pytest_asyncio.fixture
async def sessions(engine):
    """Fábrica de sessions REAIS sobre o engine do container, + limpeza da tabela.

    Autouse-equivalente por dependência: todo teste daqui pede `webhook` ou `sessions`.
    A limpeza é na mão (antes e depois) porque estas linhas são COMMITADAS — o rollback
    do `db_session` do conftest não as alcança, mesmo motivo do `cost_rows` de
    tests/test_cost_graph.py.
    """
    Session = async_sessionmaker(engine, expire_on_commit=False)

    async def _limpa():
        async with Session() as s:
            await s.execute(delete(WhatsAppMessage))
            await s.commit()

    await _limpa()
    yield Session
    await _limpa()


@pytest_asyncio.fixture
async def webhook(sessions, monkeypatch):
    """Client do webhook cuja `get_session` devolve uma sessão real, que commita.

    O `client` do conftest sobrepõe a sessão pela da transação de teste, em que
    `commit()` é NO-OP (join_transaction_mode resolve pra rollback_only quando a Session
    é ligada a uma Connection já em transação). Sob aquele regime, commitar e não
    commitar seriam indistinguíveis — exatamente a decisão que este módulo existe pra
    travar. Daí a sessão própria, ligada ao engine.

    O seam é `app.db.SessionLocal` (e não `dependency_overrides`) porque a rota abre a
    sessão ela mesma, depois da assinatura — ver o docstring de `whatsapp_webhook`.
    """
    monkeypatch.setenv("WHATSAPP_APP_SECRET", APP_SECRET)
    monkeypatch.setenv("WHATSAPP_RATE_LIMIT", "100/minute")
    monkeypatch.setenv("ASK_RATE_LIMIT_IP", "100/minute")

    from app.main import app

    monkeypatch.setattr("app.db.SessionLocal", lambda **kw: sessions(**kw))

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


async def entregar(client, payload: dict):
    """Um POST assinado, como a Meta faria."""
    body = json.dumps(payload).encode("utf-8")
    return await client.post("/webhook/whatsapp", content=body, headers=assinar(body))


async def linhas(Session) -> list[WhatsAppMessage]:
    """Lê por uma sessão SEPARADA — ver o docstring do módulo."""
    async with Session() as s:
        r = await s.execute(select(WhatsAppMessage).order_by(WhatsAppMessage.id))
        return list(r.scalars())


async def test_entrega_valida_grava_uma_linha_pending(webhook, sessions):
    """O teste da fatia: a mensagem existe DEPOIS do request, vista de outra conexão.

    É também o teste do commit. Sem `await session.commit()` na rota, a linha morre com
    a transação do request e este assert não acha nada — verificado por mutação.
    """
    r = await entregar(webhook, payload_com([msg_texto()]))
    assert r.status_code == 200

    todas = await linhas(sessions)
    assert len(todas) == 1
    linha = todas[0]

    assert linha.wamid == WAMID
    assert linha.from_phone == TELEFONE
    assert linha.text == TEXTO
    # De `value.metadata`, não da mensagem. É o que permite a W2b responder pelo número
    # certo sem fixar um no código — ver o comentário em app/whatsapp.py.
    assert linha.phone_number_id == PHONE_NUMBER_ID
    # Nada nesta fatia sai de 'pending': quem processa é a W2b.
    assert linha.status == "pending"
    assert linha.processed_at is None
    assert linha.received_at is not None


async def test_a_mesma_entrega_duas_vezes_deixa_uma_linha(webhook, sessions):
    """Idempotência por `wamid` — e é ela que autoriza o 500 do INSERT.

    A Meta REENTREGA quando não recebe 200 a tempo, e a rota provoca isso de propósito
    quando não consegue gravar. Sem o unique + ON CONFLICT, cada retentativa viraria
    linha nova e a W2b pagaria uma rodada de LLM por cópia.

    Os DOIS têm que sair 200: um 4xx/5xx na reentrega faria a Meta continuar tentando.
    """
    p = payload_com([msg_texto()])

    primeira = await entregar(webhook, p)
    segunda = await entregar(webhook, p)

    assert primeira.status_code == 200
    assert segunda.status_code == 200

    todas = await linhas(sessions)
    assert len(todas) == 1


async def test_tipo_nao_texto_nao_grava_e_devolve_200(webhook, sessions):
    """Imagem chega, é logada e não vira linha: não há pergunta pra W2b responder.

    200 porque retentativa não transforma imagem em texto — e não-2xx repetido
    desabilita a inscrição.
    """
    r = await entregar(webhook, payload_com([
        {"from": TELEFONE, "id": WAMID, "type": "image", "image": {"id": "media-1"}},
    ]))

    assert r.status_code == 200
    assert await linhas(sessions) == []


async def test_texto_vazio_nao_grava(webhook, sessions):
    """`type: "text"` com `body: ""` não é pergunta.

    O payload da borda é `str | None` sem `min_length`, então `""` chega até aqui. Uma
    linha assim seria um pendente que a W2b não tem como responder — e o /ask exige 3
    caracteres. Mesma truthiness do guard de `id`/`from` em `_mensagem`.
    """
    r = await entregar(webhook, payload_com([msg_texto(texto="")]))

    assert r.status_code == 200
    assert await linhas(sessions) == []


async def test_duas_mensagens_no_mesmo_post_gravam_duas_linhas(webhook, sessions):
    """A Meta entrega em LOTE: `value.messages` com várias, num POST só."""
    outro_wamid = "wamid.HBgNNTUxMTk4ODg4Nzc3Nw=="
    r = await entregar(webhook, payload_com([
        msg_texto(),
        msg_texto(wamid=outro_wamid, texto="E granizo, tem cobertura?"),
    ]))

    assert r.status_code == 200

    todas = await linhas(sessions)
    assert len(todas) == 2
    assert {l.wamid for l in todas} == {WAMID, outro_wamid}
    assert {l.text for l in todas} == {TEXTO, "E granizo, tem cobertura?"}


async def test_lote_com_texto_e_nao_texto_grava_so_o_texto(webhook, sessions):
    """O filtro é por mensagem, não por entrega: a imagem no meio não descarta o texto."""
    outro_wamid = "wamid.HBgNNTUxMTk4ODg4Nzc3Nw=="
    r = await entregar(webhook, payload_com([
        {"from": TELEFONE, "id": outro_wamid, "type": "audio", "audio": {"id": "a-1"}},
        msg_texto(),
    ]))

    assert r.status_code == 200

    todas = await linhas(sessions)
    assert len(todas) == 1
    assert todas[0].wamid == WAMID


async def test_sem_metadata_grava_com_phone_number_id_nulo(webhook, sessions):
    """Falta de `phone_number_id` não descarta a mensagem — mesma regra da borda.

    A pergunta continua sendo uma pergunta; quem lida com "de qual número respondo?" é
    a W2b, e perder a mensagem seria estritamente pior do que não saber o remetente.
    """
    r = await entregar(webhook, payload_com([msg_texto()], phone_number_id=None))

    assert r.status_code == 200

    todas = await linhas(sessions)
    assert len(todas) == 1
    assert todas[0].phone_number_id is None


async def test_status_so_aceita_os_tres_valores(sessions):
    """O check é o que impede a W2b de inventar um quarto estado em silêncio."""
    from sqlalchemy.exc import IntegrityError

    async with sessions() as s:
        s.add(WhatsAppMessage(wamid="wamid.check", from_phone=TELEFONE,
                              text=TEXTO, status="processando"))
        with pytest.raises(IntegrityError, match="ck_whatsapp_message_status"):
            await s.flush()


async def test_o_log_distingue_entrega_NOVA_de_REENTREGA(caplog, webhook, sessions):
    """O único consumidor do `RETURNING`, e sem este teste ele não tinha nenhum.

    `_registrar_mensagens` podia devolver `set()` — jogando fora a razão inteira de o
    INSERT ter `RETURNING` — e a suíte ficava verde: toda entrega passaria a ser logada
    como 0 novas, e o operador perderia o único sinal que separa tráfego normal de uma
    tempestade de reentrega da Meta. Verificado: com `return set()`, este teste é o único
    que fica vermelho.

    Precisa de banco de verdade — a sessão falsa de tests/test_whatsapp.py devolve
    `scalars()` vazio, então lá esta asserção mediria o próprio falso.
    """
    p = payload_com([msg_texto()])

    with caplog.at_level(logging.INFO, logger="app.main"):
        await entregar(webhook, p)
        primeira = [r.getMessage() for r in caplog.records if "entrega guardada" in r.getMessage()]

        caplog.clear()
        await entregar(webhook, p)
        segunda = [r.getMessage() for r in caplog.records if "entrega guardada" in r.getMessage()]

    assert primeira == ["webhook whatsapp: entrega guardada (1 nova(s) de 1)"]
    # Mesmo wamid: nada entrou, e o log tem que dizer isso — é o bit que importa.
    assert segunda == ["webhook whatsapp: entrega guardada (0 nova(s) de 1)"]


async def test_lote_com_uma_nova_e_uma_reentrega_conta_certo(caplog, webhook, sessions):
    """A contagem é por LOTE, e um lote misto é o caso que distingue contar de chutar."""
    outro = "wamid.HBgNNTUxMTk4ODg4Nzc3Nw=="
    await entregar(webhook, payload_com([msg_texto()]))

    with caplog.at_level(logging.INFO, logger="app.main"):
        await entregar(webhook, payload_com([msg_texto(), msg_texto(wamid=outro)]))
        linhas_log = [r.getMessage() for r in caplog.records if "entrega guardada" in r.getMessage()]

    assert linhas_log == ["webhook whatsapp: entrega guardada (1 nova(s) de 2)"]
    assert len(await linhas(sessions)) == 2


async def test_mensagem_curta_e_mensagem_longa_sao_guardadas(webhook, sessions):
    """O filtro da borda é "tem conteúdo?", não os limites do `/ask` (3..500).

    Um "ok" e um texto de 4.000 caracteres (o WhatsApp permite 4.096) são mensagens que o
    usuário de fato mandou: descartá-las na borda deixaria o remetente sem resposta E sem
    registro de que falou. Quem decide o que consegue responder é a W2b, com a linha na
    mão — e este teste é o que impede alguém de "consertar" isso espelhando o AskRequest.
    """
    outro = "wamid.HBgNNTUxMTk4ODg4Nzc3Nw=="
    longo = "e" * 4000
    r = await entregar(webhook, payload_com([
        msg_texto(texto="ok"),
        msg_texto(wamid=outro, texto=longo),
    ]))

    assert r.status_code == 200
    guardadas = {l.wamid: l.text for l in await linhas(sessions)}
    assert guardadas == {WAMID: "ok", outro: longo}


async def test_classificacao_bate_com_o_driver_de_verdade():
    """Os tipos que `_vale_reentregar` classifica são os que o asyncpg REALMENTE levanta.

    Este é o teste que a versão anterior não tinha, e a ausência dele custou caro: a lista
    era `(OperationalError, InterfaceError, DisconnectionError, TimeoutError)` e o
    `OperationalError` — o palpite óbvio — o driver asyncpg **não emite nunca**. Os testes
    do 500 injetavam um construído à mão e ficavam verdes enquanto, em produção, banco fora
    do ar caía no ramo PERMANENTE e a pergunta do usuário era descartada com 200.

    Então aqui os erros são PROVOCADOS num Postgres de verdade e classificados como a
    rota classificaria. Se alguém trocar o critério por classe de exceção de novo, ou o
    SQLAlchemy mudar o mapeamento, é aqui que aparece.
    """
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    from app.main import _vale_reentregar

    # Engine PRÓPRIA, descartada no fim. Este teste faz duas coisas que envenenam um pool
    # compartilhado: deixa `statement_timeout` setado numa conexão e MATA outra pelo pid.
    # Ligado ao `engine` da sessão do conftest, as duas vazavam pra quem pegasse aquela
    # conexão depois — verificado: o teste passava sozinho e derrubava a suíte inteira.
    eng = create_async_engine(os.environ["DATABASE_URL"])
    S = async_sessionmaker(eng, expire_on_commit=False)

    async def provoca(fn):
        async with S() as s:
            try:
                await fn(s)
            except BaseException as exc:
                return exc
        raise AssertionError("o erro não aconteceu — o teste deixou de exercitar o caso")

    # --- TRANSITÓRIOS: reentregar tem chance -------------------------------
    async def timeout(s):
        await s.execute(text("SET statement_timeout = '80ms'"))
        await s.execute(text("SELECT pg_sleep(2)"))

    erro = await provoca(timeout)                      # 57014, chega como DBAPIError
    assert _vale_reentregar(erro), f"statement timeout classificado como permanente: {erro!r}"

    # Conexão morta no meio do statement (banco reiniciado) chega como InterfaceError —
    # MEDIDO nesta base derrubando o backend com `pg_terminate_backend`. A provocação ao
    # vivo não fica aqui de propósito: matar o próprio backend no meio da suíte faz o
    # asyncpg acusar uso concorrente da conexão em vez do erro que se quer medir, e o
    # teste passa a falhar 1 em 3 execuções — flakiness que ensina a ignorar a suíte.
    # O que importa checar é a CLASSIFICAÇÃO, e ela é determinística.
    from sqlalchemy.exc import InterfaceError

    assert _vale_reentregar(InterfaceError("SELECT 1", {}, Exception("connection was closed")))

    # Banco inalcançável: o asyncpg levanta ConnectionRefusedError CRU, que o SQLAlchemy
    # NÃO embrulha — o caso mais provável de todos, e o que passava batido.
    assert _vale_reentregar(ConnectionRefusedError(61, "Connection refused"))

    # --- PERMANENTES: reentregar não conserta ------------------------------
    # NUL no texto (22021) chega como DBAPIError, igualzinho ao timeout acima: é só o
    # SQLSTATE que os separa, e é essa a razão de a classificação não ser por classe.
    erro = await provoca(lambda s: s.execute(text("SELECT cast(:v as text)").bindparams(v="a\x00b")))
    assert not _vale_reentregar(erro), f"texto com NUL classificado como transitório: {erro!r}"

    erro = await provoca(lambda s: s.execute(text("SELECT nao_existe FROM whatsapp_message")))
    assert not _vale_reentregar(erro), f"coluna inexistente classificada como transitória: {erro!r}"

    await eng.dispose()


async def test_o_log_de_falha_nao_carrega_PII(caplog, webhook, sessions, monkeypatch):
    """Mesmo recebendo uma exceção que CONTÉM o telefone e a mensagem, a rota não os loga.

    O teste planta o pior caso de propósito: um erro de INSERT de verdade, gerado por uma
    engine SEM `hide_parameters`, cujo texto carrega `[parameters: (wamid, telefone,
    mensagem)]`. É o que um `logger.exception` imprimiria — e é o mesmo dado que o laço da
    W1 mascara com cuidado e que `id_curto` existe pra manter fora do log, voltando pela
    porta do traceback.

    A garantia tem que ser da ROTA, não da engine: `hide_parameters=True` em app/db.py é
    defesa em profundidade, mas quem construir a sessão de outro jeito (este teste, um
    worker da W2b) a perderia sem aviso.
    """
    import app.main as main_mod

    erro = await _erro_de_insert_com_pii()
    # Pré-condição: o teste só prova algo se a exceção de fato contiver a PII.
    assert TELEFONE in str(erro) and TEXTO in str(erro), "o caso ruim não foi reproduzido"

    async def _explode(_session, _perguntas):
        raise erro

    monkeypatch.setattr(main_mod, "_registrar_mensagens", _explode)

    with caplog.at_level(logging.INFO, logger="app.main"):
        await entregar(webhook, payload_com([msg_texto()]))

    formatter = logging.Formatter()
    tudo = "\n".join(formatter.format(r) for r in caplog.records)

    assert TELEFONE not in tudo
    assert TEXTO not in tudo
    assert WAMID not in tudo          # o wamid embute o telefone em base64
    # E o operador continua servido: classe + SQLSTATE + o id curto da mensagem.
    assert "sqlstate=" in tudo
    assert id_curto(WAMID) in tudo


async def _erro_de_insert_com_pii():
    """Uma exceção de INSERT real, com os valores ligados visíveis no texto dela."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    eng = create_async_engine(os.environ["DATABASE_URL"], hide_parameters=False)
    try:
        async with async_sessionmaker(eng)() as s:
            try:
                await s.execute(text(
                    "INSERT INTO whatsapp_message (wamid, from_phone, text, nao_existe) "
                    "VALUES (:w, :p, :t, 1)"
                ).bindparams(w=WAMID, p=TELEFONE, t=TEXTO))
            except BaseException as exc:
                return exc
    finally:
        await eng.dispose()
    raise AssertionError("o INSERT não falhou")


async def test_worker_sql_nao_enxerga_whatsapp_message(db_session):
    """A tabela guarda PII (telefone + texto de terceiro) e o worker SQL não a alcança.

    Não é omissão do `get_schema()`: a 4b285ffad59b deixou um `ALTER DEFAULT PRIVILEGES
    ... GRANT SELECT ON TABLES TO insurance_ro` permanente, então toda tabela nova JÁ
    NASCE legível pela role. Sem o REVOKE da migration, um `SELECT * FROM
    whatsapp_message` escrito pelo LLM despejaria conversa de usuário no contexto da
    chamada seguinte.

    O que este teste NÃO prova: que o worker conecta *como* insurance_ro — `_conninfo()`
    cai pro DATABASE_URL admin quando DATABASE_URL_RO não está setada. Quem cobre esse
    caso é a allowlist de tabela (tests/test_sql_allowlist.py), e quem exercita o REVOKE
    por conexão real autenticada é mcp_servers/test_readonly_role.py.
    """
    pode_ler_inbox = await db_session.scalar(
        text("SELECT has_table_privilege('insurance_ro', 'whatsapp_message', 'SELECT')")
    )
    assert pode_ler_inbox is False

    # Controle obrigatório: um revoke largo demais derrubaria o worker inteiro e este
    # teste passaria verde do mesmo jeito.
    pode_ler_exclusion = await db_session.scalar(
        text("SELECT has_table_privilege('insurance_ro', 'exclusion', 'SELECT')")
    )
    assert pode_ler_exclusion is True


async def test_insurance_ro_le_EXATAMENTE_as_tabelas_de_dominio(db_session):
    """Guarda o fail-open que já cobrou duas vezes: o default é LEGÍVEL.

    A `4b285ffad59b` deixou um `ALTER DEFAULT PRIVILEGES ... GRANT SELECT ON TABLES TO
    insurance_ro` permanente, então toda tabela nova nasce ao alcance do worker SQL e cada
    uma precisa lembrar de revogar na mão — `clause_chunk` precisou, `whatsapp_message`
    precisou. Os testes por tabela não pegam a PRÓXIMA: um `conversation_turn` da W2b, com
    mais texto de usuário dentro, entraria legível com a suíte inteira verde.

    Este teste é sobre o CONJUNTO, e por isso não precisa ser atualizado quando alguém
    adiciona uma tabela — ele já falha, apontando o nome dela. Quem for adicionar decide:
    ou é domínio e entra no `TABLES` do MCP server, ou não é e leva o REVOKE.
    """
    from mcp_servers.postgres_mcp_server import TABLES

    r = await db_session.execute(text(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
    ))
    todas = {linha[0] for linha in r} - {"alembic_version"}

    legiveis = set()
    for tabela in todas:
        pode = await db_session.scalar(
            text("SELECT has_table_privilege('insurance_ro', :t, 'SELECT')").bindparams(t=tabela)
        )
        if pode:
            legiveis.add(tabela)

    assert legiveis == set(TABLES), (
        f"o alcance do worker SQL divergiu do allowlist. A mais: {sorted(legiveis - set(TABLES))} "
        f"(falta REVOKE na migration). A menos: {sorted(set(TABLES) - legiveis)} "
        f"(o worker parou de enxergar tabela de domínio)."
    )



async def test_indice_de_status_existe_na_coluna_certa(db_session):
    """A varredura de pendentes da W2b depende dele.

    A asserção é sobre a DEFINIÇÃO, não sobre a existência da linha em `pg_indexes`:
    com `is not None` bastava criar o índice na coluna errada pro teste passar verde —
    o nome do índice não prova o conteúdo dele.
    """
    indexdef = await db_session.scalar(
        text("SELECT indexdef FROM pg_indexes WHERE tablename = 'whatsapp_message' "
             "AND indexname = :nome").bindparams(nome="ix_whatsapp_message_status")
    )
    assert indexdef is not None
    assert "(status)" in indexdef
