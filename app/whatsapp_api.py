"""Borda de SAÍDA do WhatsApp (Meta Cloud API): mandar uma mensagem de texto.

Este módulo é o espelho do `app/whatsapp.py` — lá se decide se o evento veio da Meta,
aqui se manda a resposta de volta — e ele NÃO conhece banco, grafo nem request. Quem
orquestra é `app/inbox.py`.

Três decisões moram aqui, e as três são sobre o que NÃO pode escapar:

1. **O token é um segredo SEPARADO do app secret**, lido em call time e fail-closed. O
   app secret verifica quem entra; este autoriza o que sai. Vazar um não vaza o outro.
2. **A exceção de envio não carrega o corpo.** O corpo tem o telefone de uma pessoa e a
   resposta inteira — a mesma PII que `mascarar_telefone` e `id_curto` existem pra manter
   fora do log, e que um `raise_for_status()` publicaria de graça pelo `.request` que a
   exceção do httpx guarda.
3. **O fail-closed levanta `RuntimeError`, não `HTTPException`** — ver `get_access_token`.
"""
import os

import httpx

# Versão da Graph API, fixa no código de propósito: mudar de versão muda o CONTRATO
# (campos que somem, campos que aparecem), então é revisão de código e não env var.
GRAPH_API_VERSION = "v21.0"
GRAPH_BASE = "https://graph.facebook.com"

# Timeout ESCOLHIDO, não o default do httpx (que é 5s de connect e sem teto de leitura no
# `Timeout(None)` de quem constrói o cliente na mão). Mesma lição do `EMBED_TIMEOUT_QUERY`,
# e aqui é mais forte: o envio acontece com a LINHA TRAVADA e a transação aberta (ver
# `app/inbox.py`), então cada segundo esperando é um segundo de trava — exatamente onde um
# Postgres com `idle_in_transaction_session_timeout` cobra a conta.
ENVIO_TIMEOUT = 10.0

# Teto do corpo de uma mensagem de texto na Cloud API. Acima dele a Meta responde 400, e como
# a resposta da varredura é determinística (fica gravada em `answer`) todas as tentativas
# falhariam idênticas — o usuário receberia SILÊNCIO. Quem corta é `app/inbox.py`, que também
# anuncia o corte no texto; aqui a constante só declara o limite de quem a impõe.
MAX_BODY_CHARS = 4096


class ConfiguracaoAusente(RuntimeError):
    """Falta o token de envio. Tipo PRÓPRIO, e não um `RuntimeError` cru, por dois motivos.

    (1) A varredura precisa distinguir "a Meta recusou esta mensagem" (retentável, gasta
    tentativa) de "não temos como enviar NADA" (não é da mensagem, e gastar tentativa nela
    seria perder perguntas por erro de configuração). (2) O `scripts/process_pending.py`
    trata este caso com uma saída limpa e sem traceback; com `except RuntimeError` genérico
    ele engolia também um "Event loop is closed" ou um "Session is already flushing" e os
    imprimia como se fossem o token faltando, escondendo um crash de verdade num script que
    acabou de gastar dinheiro.

    Continua sendo `RuntimeError` pra não quebrar quem já captura o tipo largo.
    """


class EnvioRecusado(Exception):
    """A Cloud API não aceitou a mensagem — SEM nada do corpo enviado.

    `status` é o HTTP que voltou, ou `None` quando nem houve resposta (timeout, DNS,
    conexão recusada). O texto é montado à mão, nunca herdado da exceção do httpx: ver
    `_resumo_da_falha`.
    """

    def __init__(self, resumo: str, status: int | None = None):
        super().__init__(resumo)
        self.status = status


def get_access_token() -> str:
    """Token da Cloud API (`WHATSAPP_ACCESS_TOKEN`), lido NA CHAMADA. Fail-closed.

    **Levanta `RuntimeError`, e não `HTTPException` como o `_config` de `app/whatsapp.py`**
    — é essa diferença que justifica um módulo à parte em vez de mais um wrapper lá. Os dois
    consumidores deste token rodam FORA de um request: a varredura em background e o
    `scripts/process_pending.py`. Numa BackgroundTask a `HTTPException` subiria depois de a
    resposta já ter começado a ser enviada, e no CLI apareceria como
    `starlette.exceptions.HTTPException: 500: configuração ausente` — um tipo sem sentido e
    uma mensagem que não diz o que fazer. Precedente: `app/rag/embedding.py::get_client`.

    Sem `@lru_cache`, ao contrário do cliente da Voyage: o token da Cloud API expira e é
    rotacionável, e cachear obrigaria a reiniciar o processo pra trocá-lo — o oposto da regra
    de "env em call time" que o projeto inteiro segue.

    A mensagem NOMEIA a variável, ao contrário do 500 do webhook, e a diferença é o destino:
    lá o texto vai pro corpo de uma resposta HTTP pública (nomear entregaria o schema de
    configuração do deploy a um scanner); aqui vai pro log e pro terminal do operador, que é
    exatamente quem precisa do nome.
    """
    valor = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
    # Mesma lição do `_config`: o placeholder é truthy e passaria por um `if not valor`.
    if valor.startswith("CHANGE_ME"):
        valor = ""
    if not valor:
        raise ConfiguracaoAusente(
            "WHATSAPP_ACCESS_TOKEN ausente ou vazia — sem ela nenhuma resposta pode ser "
            "entregue. Pegue o token no painel da Meta (WhatsApp > Configuração da API) e "
            "ponha no .env / nas variáveis do deploy."
        )
    return valor


def _resumo_da_falha(exc: BaseException, resposta: httpx.Response | None) -> str:
    """Identifica a falha SEM o corpo, sem o telefone e sem o token.

    A exceção do httpx guarda `.request` (e `.response`, no caso de status), e o
    `.content` da request É o corpo que acabamos de mandar: o número de uma pessoa e a
    resposta inteira. Então nada dela é reaproveitado — nem `str(exc)`, que inclui a URL.

    Da resposta de erro da Meta entram só os campos ESTRUTURADOS (`error.code` e
    `error.error_subcode`, que são inteiros de um catálogo público). `error.message` fica
    de fora de propósito: é texto livre montado do outro lado, e `error_data.details`
    documentadamente ecoa contexto do pedido — seria o corpo voltando pela porta lateral.
    Quem precisa do significado do código consulta a tabela de erros da Meta.
    """
    partes = [type(exc).__name__]
    if resposta is not None:
        partes.append(f"http={resposta.status_code}")
        try:
            erro = resposta.json().get("error", {})
            for campo in ("code", "error_subcode"):
                valor = erro.get(campo)
                if isinstance(valor, int):
                    partes.append(f"{campo}={valor}")
        except (ValueError, AttributeError):
            # Corpo de erro não-JSON (um HTML de gateway, por exemplo). O status já basta, e
            # despejar o corpo aqui é justamente o que esta função existe pra impedir.
            # `json.JSONDecodeError` NÃO entra na tupla: ela É uma `ValueError`, e listar as
            # duas sugeriria dois casos distintos — convidando o próximo leitor a "limpar" a
            # tupla tirando a base, o que aí sim mudaria o comportamento.
            pass
    return " ".join(partes)


def _wamid_enviado(payload: object) -> str | None:
    """O `wamid` que a Meta deu à NOSSA mensagem, ou None se o formato mudar.

    Defensivo por construção, como `extract_messages`: nunca levanta. É o único elo entre a
    resposta que mandamos e os `value.statuses` (entregue/lido) que vão chegar no mesmo
    webhook, então vale extrair — mas não vale derrubar um envio bem-sucedido por causa de
    um campo que a Meta renomeou.
    """
    if not isinstance(payload, dict):
        return None
    mensagens = payload.get("messages")
    if not isinstance(mensagens, list) or not mensagens:
        return None
    primeira = mensagens[0]
    if not isinstance(primeira, dict):
        return None
    wamid = primeira.get("id")
    return wamid if isinstance(wamid, str) and wamid else None


async def enviar_texto(
    phone_number_id: str,
    to: str,
    body: str,
    client: httpx.AsyncClient | None = None,
) -> str | None:
    """Manda `body` para `to` pelo número `phone_number_id`. Devolve o wamid da resposta.

    `phone_number_id` vem da LINHA (`whatsapp_message.phone_number_id`), nunca de config: um
    app secret da Meta cobre todos os números da conta, e fixar um no código daria resposta
    saindo do remetente errado no dia do segundo número. É o handoff que a W1 anunciou e a
    W2a persistiu.

    **Levanta `EnvioRecusado` em qualquer falha, e só ela** — nenhuma exceção do httpx
    escapa (o `except` é sobre `Exception`, e o porquê está lá embaixo). O motivo é um só:
    `HTTPStatusError` e `RequestError` seguram `.request`, cujo `.content` é o corpo com o
    telefone e a resposta. O `from None` fecha a
    última fresta: `traceback_da_cadeia` (app/logging_config.py) respeita
    `__suppress_context__`, então o objeto httpx não volta ao log pela cadeia da exceção.

    O cliente é injetável (é o seam dos testes) e o default é criado AQUI, por chamada, em
    vez de cacheado no módulo. Um `AsyncClient` de módulo fica preso ao event loop que o
    criou, e este código roda em três loops diferentes — uvicorn, o `asyncio.run` do CLI e o
    loop de sessão do pytest. Os clientes da Voyage podem ser cacheados porque são
    SÍNCRONOS, vivendo atrás de um `to_thread`. O preço (um handshake por mensagem) é
    irrelevante no volume desta superfície.
    """
    url = f"{GRAPH_BASE}/{GRAPH_API_VERSION}/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to,
        "type": "text",
        # `preview_url=False` é decisão, não default copiado: o corpo é escrito por um
        # MODELO, e ligar o preview faria a plataforma buscar uma URL que o modelo inventou.
        "text": {"preview_url": False, "body": body},
    }
    headers = {"Authorization": f"Bearer {get_access_token()}"}

    proprio = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(ENVIO_TIMEOUT))
    try:
        resposta = None
        try:
            resposta = await client.post(url, json=payload, headers=headers)
            # `raise_for_status()` de propósito NÃO é usado como a saída da função: ele
            # levanta uma exceção que carrega a request inteira. Ele serve aqui só como
            # detector; quem sai é a `EnvioRecusado` montada abaixo.
            resposta.raise_for_status()
        except Exception as exc:
            # `except Exception`, e NÃO `except httpx.HTTPError`: `InvalidURL`,
            # `CookieConflict` e `StreamError` não descendem de `HTTPError`, e a primeira é
            # alcançável — `phone_number_id` vem de `value.metadata` e a borda só o valida
            # como `str | None`, então um valor com espaço ou caractere de controle produz
            # `InvalidURL`, cuja mensagem carrega a URL inteira. Escapando daqui ela contorna
            # a esterilização e o `from None`, que é exatamente o que este módulo existe pra
            # impedir. A promessa é "nada do httpx escapa", e ela só é verdadeira assim.
            resumo = _resumo_da_falha(exc, resposta)
            status = resposta.status_code if resposta is not None else None
            raise EnvioRecusado(resumo, status) from None
        return _wamid_enviado(_json_ou_none(resposta))
    finally:
        if proprio:
            await client.aclose()


def _json_ou_none(resposta: httpx.Response) -> object:
    """Corpo de sucesso como objeto, ou None se não for JSON — nunca levanta.

    Um 200 com corpo estranho é envio BEM-SUCEDIDO: a mensagem saiu. Perder o wamid da
    resposta é perder correlação, não perder a entrega, e trocar isso por uma exceção faria
    a varredura contar como falha (e retentar) algo que já chegou na pessoa.
    """
    try:
        return resposta.json()
    except ValueError:
        return None
