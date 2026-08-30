"""Log estruturado: uma linha JSON por registro, em stdout, correlacionada por request.

**Por que este módulo existe.** Até aqui o projeto nunca configurou logging, e o INFO
aparecer era ACIDENTE: `mcp_servers/postgres_mcp_server.py` constrói um `FastMCP` no nível
do módulo, o `__init__` dele chama `logging.basicConfig`, e `app/agents/graph.py` importa
`get_schema`/`run_query` de lá — então `import app.main` acabava configurando a raiz em
INFO de carona num servidor MCP que o processo web NUNCA serve. Medido: sem esse import a
raiz fica `[] / WARNING`; com ele, `StreamHandler(stderr) / INFO`, formato `"%(message)s"`.
Ou seja, todo o log em INFO do webhook (o da W1 e o "entrega guardada" da W2a) dependia de
um efeito colateral, saía sem timestamp e sem nível, e teria sumido em silêncio no dia em
que alguém tornasse aquela construção preguiçosa.

**O contrato.** `configure_logging()` é chamado no import de `app/main.py`, antes de
qualquer import de terceiro. A ordem não é o que faz esta configuração VENCER — o
`dictConfig` remove os handlers existentes, então ela venceria mesmo chegando depois
(verificado). O que a ordem compra é que o handler do `basicConfig` nunca chegue a
existir, e que o que for logado DURANTE os imports restantes já saia formatado.
"""
import json
import logging
import logging.config
import os
import re
import traceback
from datetime import datetime, timezone

from dotenv import load_dotenv

from app.agents.context import get_client_name, get_request_id

# O que o operador lê pra achar uma linha, e o que ele digita pra mudar o volume.
NIVEL_PADRAO = "INFO"
VAR_NIVEL = "LOG_LEVEL"

# Allowlist EXPLÍCITA, e não `logging.getLevelNamesMapping()`: aquele mapa aceita `NOTSET`
# (que vira "herda", ou seja, INFO — indistinguível de configuração certa, sem aviso), os
# apelidos `WARN`/`FATAL`, e qualquer nível que uma lib tenha registrado com
# `addLevelName` antes do import (o uvicorn registra `TRACE`). O fallback só significa
# alguma coisa se recusar o que não dá pra usar.
NIVEIS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# O uvicorn instala handlers PRÓPRIOS nestes três (com propagate=False) quando o servidor
# sobe, e a configuração dele roda ANTES do import da app. Sem reconfigurá-los aqui,
# metade da saída do processo sai em texto e metade em JSON — e é justamente a metade que
# registra os requests que ficaria de fora.
LOGGERS_DO_UVICORN = ("uvicorn", "uvicorn.error", "uvicorn.access")

# O pacote inteiro do projeto — é o nó em que `LOG_LEVEL=DEBUG` para de subir.
LOGGER_DO_APP = "app"


# O handshake da Meta chega como
# `GET /webhook/whatsapp?hub.mode=subscribe&hub.verify_token=<SEGREDO>&hub.challenge=...`,
# e o log de acesso do uvicorn monta a mensagem com a query string CRUA. Adotar o
# `uvicorn.access` neste handler traria o token pra dentro de um fluxo que se anuncia como
# revisado — enquanto `app/main.py` tira esse mesmo token do log dele de propósito, porque é
# o único segredo do sistema adivinhável por tentativa.
#
# A regra é por SUFIXO do nome do parâmetro, e substituiu uma lista de nomes EXATOS. A versão
# anterior casava o literal `hub.verify_token=` e mais nada — com o ponto escapado, então nem
# a grafia vizinha `hub_verify_token=` entrava. Só que a Meta manda as DUAS na mesma query
# string, e a que escapava saía em claro. O modo de falhar é o que condena aquele desenho: a
# lista continuava "cobrindo" o parâmetro, e a metade descoberta passava sem erro, sem aviso e
# sem teste vermelho. Uma lista de nomes precisa de uma entrada por grafia, e quem escreve a
# lista não controla quem escolhe as grafias.
#
# O que torna o sufixo seguro não é a precisão, é a DIREÇÃO do erro: ele redige a mais, nunca
# a menos. O custo conhecido é cosmético e chega pelo `exc_traceback`, porque `format_tb`
# inclui a linha de FONTE de cada frame: `app/rag/embedding.py` sai com
# `voyageai.Client(api_key=<redigido> timeout=...)` — o nome da VARIÁVEL, não um segredo — e é
# o único que de fato aparece, porque aquele frame está no caminho vivo do `rag_worker`; um
# `mapped_column(primary_key=True)` de `app/models.py` faria o mesmo. Tipo, arquivo, linha e
# cadeia de causas ficam intactos. `input_tokens=`/`max_tokens=` escapam só pelo `s` do plural,
# e é a única coisa que os separa do mesmo destino.
_SUFIXOS_SENSIVEIS = ("token", "secret", "signature", "password", "key")

# Sufixo e NÃO substring: o `=` logo depois da alternância é o que mantém `token_id=42` — um
# id, não um segredo — legível. Nada de âncora `[?&]`: o alvo também chega pela linha de fonte
# de um frame (`RuntimeError("hub.verify_token=...")`), onde o caractere anterior é `"` —
# exigir delimitador de query reabriria o vazamento pelo traceback, que é o buraco que
# `test_a_redacao_cobre_o_traceback_e_nao_so_a_mensagem` fechou.
#
# O lookbehind é OUTRA coisa e é OBRIGATÓRIO: ele não exige delimitador, só proíbe COMEÇAR no
# meio de um nome — as posições que só geram backtracking. Sem ele o `[^&\s"=]*` retenta em
# cada posição de um corridão sem separador e o casamento vira O(n²). Não é teórico: o
# `uvicorn.access` monta a mensagem com o path CRU (`get_path_with_query_string`), o teto de
# uma request line é 16 KB (o default do `h11`, que é o parser em uso porque `httptools` não
# está instalado), e 16 KB de path custavam 4 SEGUNDOS de CPU na thread do event loop — num GET
# anônimo pro webhook público, que é a rota mais exposta do sistema. Com o lookbehind: 0,7 ms,
# saída idêntica (fuzz diferencial de 300 mil strings, zero divergência).
#
# `re.escape` em cada sufixo é no-op hoje (são cinco palavras minúsculas) e existe pelo dia em
# que a lista crescer, que é o ponto inteiro dela: um `api.key` interpolado cru viraria
# curinga e passaria a redigir `apiXkey=`, e um `secret(` levantaria `re.error` NO IMPORT de
# `app.logging_config` — a primeira linha de `app/main.py`, ou seja, o web process não sobe.
# Uma lista de configuração não pode ter esse gatilho: é o oposto do fail-open barulhento que
# `_nivel_do_env` pratica logo abaixo.
#
# `IGNORECASE` porque o nome do parâmetro é escolhido por quem chama, não por nós:
# `?HUB_VERIFY_TOKEN=` é o mesmo segredo, e casar só minúscula seria a mesma falha de grafia
# que motivou a fatia, um nível abaixo.
_SEGREDOS_NA_URL = re.compile(
    r"(?<![^&\s\"=])([^&\s\"=]*(?:"
    + "|".join(re.escape(sufixo) for sufixo in _SUFIXOS_SENSIVEIS)
    + r")=)[^&\s\"]+",
    re.IGNORECASE,
)


def _redigir(mensagem: str) -> str:
    return _SEGREDOS_NA_URL.sub(r"\1<redigido>", mensagem)


class JsonFormatter(logging.Formatter):
    """Uma linha JSON por registro, com uma LISTA FECHADA de campos.

    Nada de `record.__dict__`: um dump genérico é como PII entra num log sem ninguém ter
    decidido que ela entra. Cada chave abaixo foi escolhida.

    `ensure_ascii=False` porque as mensagens são pt-BR — escapar acento só faz o operador
    ler `\\u00e7`. E o `json.dumps` escapa `\\n`, então um traceback de vinte frames vira
    UMA linha, que é o que um coletor de log espera de uma saída por linha.
    """

    def format(self, record: logging.LogRecord) -> str:
        linha = {
            # Timestamp em UTC e ISO: o container não tem timezone garantida, e "14:02"
            # sem fuso é inútil pra cruzar com o relógio de outra máquina.
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Postos pelo ContextFilter. OMITIDOS quando não há request: a chave ausente diz
        # "isto rodou fora de um request" (extração offline, `python -m app.agents.graph`),
        # enquanto `null` ou a string "None" seriam ruído que alguém acabaria filtrando
        # errado. `getattr` com default porque um registro pode chegar por um handler que
        # não tem o filtro (o `caplog` do pytest, por exemplo).
        for chave in ("request_id", "client"):
            valor = getattr(record, chave, None)
            # `is not None` e não truthiness: um cliente chamado "0" (ou um nome vazio
            # vindo de uma entrada malformada em API_TOKENS) é um valor REAL, e omiti-lo
            # atribuiria as chamadas pagas dele ao balde do offline — que é justamente o
            # que a ausência da chave significa aqui.
            if valor is not None:
                linha[chave] = valor

        exc_info = record.exc_info
        if exc_info and exc_info[1] is not None:
            linha.update(_campos_da_excecao(exc_info[1]))

        # A redação cobre TODO campo de texto, não só o `message`: `format_tb` inclui a
        # linha de FONTE de cada frame, então um alvo da lista pode chegar pelo
        # `exc_traceback` com o `message` limpo.
        #
        # E é feita ANTES do `dumps`, não sobre a linha serializada — ali o texto já está
        # escapado, e a classe de caracteres do padrão engoliria a contrabarra de um `\"`,
        # produzindo uma aspa solta dentro da string e JSON inválido. (Aconteceu; é o que
        # `test_a_redacao_cobre_o_traceback_e_nao_so_a_mensagem` pegou.)
        redigida = {c: _redigir(v) if isinstance(v, str) else v for c, v in linha.items()}
        return json.dumps(redigida, ensure_ascii=False)


def _nome_do_tipo(exc: BaseException) -> str:
    cls = type(exc)
    if cls.__module__ == "builtins":
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def _cadeia(exc: BaseException, limite: int = 5) -> list[BaseException]:
    """A exceção e as que a causaram, respeitando `raise ... from None`.

    `__cause__` (o `from`) tem prioridade sobre `__context__` (o encadeamento
    implícito), e `__suppress_context__` é o que `from None` liga — ignorá-lo religaria
    no log um contexto que alguém suprimiu de propósito. O teto e o conjunto de `id()`
    existem porque uma cadeia pode ser circular.
    """
    elos: list[BaseException] = []
    vistos: set[int] = set()
    atual: BaseException | None = exc
    while atual is not None and id(atual) not in vistos and len(elos) < limite:
        vistos.add(id(atual))
        elos.append(atual)
        proximo = atual.__cause__
        if proximo is None and not atual.__suppress_context__:
            proximo = atual.__context__
        atual = proximo
    return elos


def _campos_da_excecao(exc: BaseException) -> dict:
    """Tipo e FRAMES da exceção — nunca a mensagem dela. Esta é a decisão de PII da fatia.

    `traceback.format_tb` formata só os frames; `format_exception` acrescentaria a última
    linha, que é onde `str(exc)` mora — e é exatamente ela que não pode entrar. O texto de
    uma exceção do SQLAlchemy inclui `[parameters: (...)]`, ou seja, o telefone e a
    mensagem inteira do usuário; o do psycopg inclui host, porta, usuário e banco; o do
    Postgres põe os valores da chave no DETAIL de uma violação de unique. Mensagem de
    exceção é texto livre que uma biblioteca monta com o que tiver em mãos — o único campo
    do registro cujo conteúdo é escolhido por terceiro —, então a regra da lista fechada
    vale pra ela também.

    O custo é real e consciente: o traceback que o `basicConfig` imprimia trazia, por
    exemplo, o status que a Anthropic devolveu, e isso agora fica de fora. Quem precisa de
    um detalhe da exceção NOMEIA esse detalhe no call site, que é o que
    `resumo_do_erro` (logo abaixo, neste mesmo módulo) já faz com o SQLSTATE.

    **A CADEIA entra, e isso não é exceção à regra — é a mesma regra.** `format_tb` sozinho
    anda só no traceback mais externo, então um `DBAPIError` do SQLAlchemy embrulhando um
    erro do asyncpg reportaria os frames do SQLAlchemy e NADA sobre qual erro do driver
    aconteceu — justamente a identidade que a classificação de `_vale_reentregar` disputa.
    O que se descartou por ser PII foi a MENSAGEM; o tipo e os frames de quem causou não
    carregam nada disso, então deixá-los de fora seria perda colateral, não escolha.
    """
    return {
        "exc_type": _nome_do_tipo(exc),
        "exc_traceback": traceback_da_cadeia(exc),
    }


def resumo_do_erro(exc: BaseException) -> str:
    """Identifica um erro de banco SEM nada que o usuário escreveu.

    O texto de uma exceção do SQLAlchemy inclui `[parameters: (...)]` — os valores ligados,
    que nas superfícies deste projeto são o telefone e a mensagem inteira —, e é justamente
    ele que um `logger.exception` imprime. `hide_parameters=True` na engine (app/db.py) tapa
    isso, mas nenhum call site pode DEPENDER de um flag de engine pra manter a promessa de
    PII: quem construir a sessão de outro jeito (um teste, um worker) a perde sem aviso.

    A mensagem do servidor também fica de fora, e não por zelo: o Postgres põe os valores da
    chave no DETAIL de uma violação de unique (`Key (wamid)=(...)`), e o wamid embute o
    telefone em base64. O SQLSTATE identifica a falha com precisão e não carrega dado nenhum.

    Mora aqui, e não em `app/main.py` (onde nasceu), porque tem DOIS chamadores — o webhook e
    a varredura do WhatsApp — e os dois obedecem à mesma regra. Duas cópias divergiriam no
    dia em que uma fosse endurecida (um `pgcode`, um driver que reporte o estado em outro
    lugar), e as duas metades da mesma superfície passariam a descrever a mesma falha de
    jeitos diferentes. É a regra que tirou o `final_answer` de `app/main.py`.
    """
    sqlstate = getattr(getattr(exc, "orig", None), "sqlstate", None)
    if isinstance(sqlstate, str):
        return f"{type(exc).__name__}(sqlstate={sqlstate})"
    return type(exc).__name__


def traceback_da_cadeia(exc: BaseException) -> str:
    """Tipos e frames da exceção e de quem a causou, SEM nenhuma mensagem.

    Público porque `app/main.py` monta o traceback do webhook à mão (ele não pode passar
    `exc_info`, ver o comentário lá) e precisa exatamente disto — sem esta função ele
    reportaria só o traceback mais externo, perdendo qual erro do driver aconteceu.
    """
    partes = []
    for i, elo in enumerate(_cadeia(exc)):
        cabecalho = _nome_do_tipo(elo) if i == 0 else f"causado por {_nome_do_tipo(elo)}"
        frames = "".join(traceback.format_tb(elo.__traceback__)).rstrip()
        partes.append(f"{cabecalho}\n{frames}" if frames else cabecalho)
    return "\n".join(partes)


class ContextFilter(logging.Filter):
    """Anexa `request_id` e `client` ao registro, lidos dos ContextVars do request.

    É isto que torna a correlação ESTRUTURAL: nenhum call site passa `request_id` à mão, e
    portanto nenhum call site pode esquecer. São as MESMAS chaves de `cost_event`, então
    "o que este request custou" e "o que este request logou" se encontram pelo mesmo valor
    — e ele volta pro usuário no `X-Request-Id`, que é o que torna um erro relatado por
    alguém rastreável até a linha.

    Mora no HANDLER e não nos loggers: assim qualquer registro que chegue à saída passa
    por aqui, inclusive os do `uvicorn.access`, que ninguém do nosso lado emite.

    Sempre devolve True — é um enriquecedor, não um filtro de verdade.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id()
        record.client = get_client_name()
        return True


def _nivel_do_env() -> tuple[str, str | None]:
    """Lê LOG_LEVEL no momento da CHAMADA (não no import), com fallback barulhento.

    Env lido no nível do módulo mataria a coleta do pytest (a lição do `DATABASE_URL`, que
    voltou 4×) e obrigaria o import a exigir configuração. Na prática, como o único chamador
    é o import de `app/main.py`, o valor vale pelo processo inteiro: mudar LOG_LEVEL exige
    reiniciar, não reconstruir a imagem.

    O fallback espelha o `app/limits.py::_limit_from_env`: valor inválido não pode virar
    exceção no boot (o web process não subiria) nem silêncio — vira o default MAIS um aviso,
    que só dá pra emitir depois da configuração.
    """
    bruto = os.environ.get(VAR_NIVEL)
    if not bruto:
        return NIVEL_PADRAO, None
    nivel = bruto.strip().upper()
    if nivel not in NIVEIS:
        return NIVEL_PADRAO, f"{VAR_NIVEL} inválido ({bruto!r}); usando {NIVEL_PADRAO}"
    return nivel, None


def _config(nivel: str) -> dict:
    # DEBUG vale só pro NOSSO código, e este piso é a razão de existirem dois níveis.
    #
    # Com o nível aplicado direto na raiz, `LOG_LEVEL=DEBUG` — um valor que o .env.example
    # anuncia — liga o DEBUG de TODA biblioteca: o `anthropic` emite
    # `log.debug("Request options: %s", ...)` com o array `messages` inteiro, ou seja, a
    # pergunta literal do usuário e cada cláusula recuperada (e, na W2b, a mensagem que
    # chegou pelo WhatsApp). Seria a PII que a lista fechada de campos existe pra barrar
    # entrando pelo `message`, que é o campo em que ela é impossível de revisar.
    #
    # Então a raiz nunca desce abaixo de INFO, e quem desce é o logger `app`. Apertar
    # continua valendo pra todo mundo (`LOG_LEVEL=WARNING` cala a raiz também), porque
    # calar não vaza nada.
    # O piso vale pra raiz E pros loggers do uvicorn: o uvicorn é biblioteca de terceiro
    # como qualquer outra, e ele é o ÚNICO que este config toca pelo nome — deixá-lo de
    # fora tornaria falsa, justo nele, a frase "DEBUG vale só pro nosso código".
    nivel_de_terceiro = nivel if logging.getLevelName(nivel) >= logging.INFO else "INFO"

    return {
        "version": 1,
        # OBRIGATÓRIO. O default marca `disabled=True` em todo logger já criado no
        # processo — o mesmo estrago que o `fileConfig` do Alembic fazia dentro do pytest,
        # cujo sintoma é o pior possível: teste de log que passa sem ver log nenhum.
        "disable_existing_loggers": False,
        "formatters": {"json": {"()": JsonFormatter}},
        "filters": {"contexto": {"()": ContextFilter}},
        "handlers": {
            "stdout": {
                "class": "logging.StreamHandler",
                # stdout e não stderr: log não é erro, e o convencionado em container é que
                # o processo despeje seus eventos em stdout pra quem coleta.
                "stream": "ext://sys.stdout",
                "formatter": "json",
                "filters": ["contexto"],
            }
        },
        "root": {"handlers": ["stdout"], "level": nivel_de_terceiro},
        "loggers": {
            # Sem handler próprio e sem mexer no `propagate`: o registro sobe pro handler
            # da raiz, e o que este nó define é só até onde o NOSSO código fala.
            LOGGER_DO_APP: {"level": nivel},
            **{
                nome: {
                    "handlers": ["stdout"],
                    "level": nivel_de_terceiro,
                    "propagate": False,
                }
                for nome in LOGGERS_DO_UVICORN
            },
        },
    }


def configure_logging() -> None:
    """Instala o handler único de stdout. Idempotente, e não pisa em handler alheio.

    **`load_dotenv()` primeiro, senão `LOG_LEVEL` no `.env` não vale nada.** Esta função
    roda no import de `app/main.py`, que é ANTES do `load_dotenv()` do `app/db.py` (ele só
    acontece algumas linhas abaixo, quando `app.db` é importado). Sem a chamada aqui, o
    setup local documentado — `cp .env.example .env` e editar — deixava a variável
    invisível, e nem o aviso de valor inválido disparava, porque não havia valor nenhum.
    Verificado. `load_dotenv` não sobrescreve env já existente, então chamar cedo não muda
    nada pra quem configura pelo ambiente (Railway, CI), e é no-op sem arquivo.

    **Handlers que não são nossos são PRESERVADOS.** `dictConfig` remove todos os handlers
    de cada logger que configura, e na raiz é onde vivem os handlers de captura do pytest
    (`caplog`, mas também `--log-file` e `--log-cli-level`, que não se reinstalam por fase).
    Como esta função roda no import de `app.main` — que várias fixtures fazem no MEIO da
    sessão —, sem esta reinstalação um `pytest --log-file` sairia vazio a partir dali: é o
    mesmo estrago que `tests/conftest.py` (`configure_logger=False`) e
    `tests/test_migration.py` já pagam pra impedir do lado do Alembic.

    Isso NÃO ressuscita o acidente do FastMCP: o `basicConfig` só instala handler quando a
    raiz está vazia (verificado), e raiz vazia é exatamente o caso em que não há nada a
    preservar. Os dois casos são mutuamente exclusivos.

    A idempotência continua vindo do `dictConfig`: o nosso handler é recriado, não somado.
    """
    load_dotenv()

    raiz = logging.getLogger()
    alheios = [h for h in raiz.handlers if not isinstance(h.formatter, JsonFormatter)]

    nivel, aviso = _nivel_do_env()
    logging.config.dictConfig(_config(nivel))

    for handler in alheios:
        raiz.addHandler(handler)

    if aviso:
        # Depois do dictConfig de propósito: antes dele o aviso não teria pra onde ir.
        logging.getLogger(__name__).warning(aviso)


__all__ = [
    "ContextFilter",
    "JsonFormatter",
    "configure_logging",
    "traceback_da_cadeia",
    "NIVEIS",
    "NIVEL_PADRAO",
    "VAR_NIVEL",
]
