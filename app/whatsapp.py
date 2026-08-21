"""
Borda de entrada do WhatsApp (Meta Cloud API): quem é a Meta e o que ela mandou.

Este módulo NÃO conhece o grafo e NÃO responde ao usuário — isso é a fatia W2. Aqui
só se decide (a) se o evento veio mesmo da Meta e chegou íntegro e (b) quais campos
mínimos dá pra extrair dele.

A rota que usa isto é a PRIMEIRA do projeto sem Bearer token: ela é pública porque a
Meta não manda `Authorization`. O controle de acesso passa a ser a assinatura HMAC do
corpo (`X-Hub-Signature-256`), então é ela que faz o papel que o `app/auth.py` faz no
`/ask` — e é por isso que as decisões abaixo espelham as de lá (env lida no call time,
500 na configuração ausente, comparação de tempo constante sobre BYTES).
"""
import hashlib
import hmac
import logging
import os
from dataclasses import dataclass

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# Teto do corpo aceito antes de qualquer verificação. A rota é o primeiro endpoint
# público do projeto e o HMAC só pode ser calculado depois de ler os bytes, então sem
# teto um anônimo faz a app bufferizar o que quiser em memória. Os payloads da Meta
# são pequenos (mídia chega como id, nunca embutida); 128 KB é folga de sobra.
MAX_BODY_BYTES = 128 * 1024

# Header em que a Meta manda o HMAC-SHA256 do corpo. Existe também um
# `X-Hub-Signature` legado, em SHA-1, que ignoramos de propósito.
ASSINATURA_HEADER = "X-Hub-Signature-256"

_PREFIXO_ASSINATURA = "sha256="

# hub.mode do handshake de verificação. Qualquer outro valor é recusado.
_MODO_SUBSCRIBE = "subscribe"


def _config(var: str) -> str:
    """Lê uma variável obrigatória do ambiente NA CHAMADA, com 500 se faltar.

    Duas decisões, e as duas são cópias de lições que este projeto já pagou.

    (1) O env é lido AQUI DENTRO, nunca no nível do módulo: `import app.main` precisa
    continuar funcionando sem env var nenhuma, senão a *coleta* do pytest morre antes
    de qualquer fixture rodar (é o mesmo bug do `DATABASE_URL` em `app/db.py` e da
    `VOYAGE_API_KEY` em `app/rag/embedding.py`). De quebra, dá pra rotacionar o app
    secret sem rebuild.

    (2) Configuração ausente é **500, não 403** — exatamente como `_load_tokens`
    devolve 500 e não 401 quando `API_TOKENS` está vazio. Um 403 aqui esconderia um
    deploy quebrado atrás de uma resposta que parece "a Meta mandou assinatura
    errada", e os dois têm reação oposta: um é incidente nosso, o outro é request
    forjado de terceiro.

    Levantamos `HTTPException` (e não `RuntimeError`) porque é o que `app/auth.py` já
    faz e porque um erro não-HTTP sairia da borda como exceção não tratada em vez de
    resposta.
    """
    valor = os.getenv(var, "").strip()
    if not valor:
        # ERROR, não WARNING: no webhook um erro de configuração tem custo COMPOSTO —
        # a Meta reenvia em não-2xx e desabilita a inscrição depois de falhas
        # repetidas, e recadastrar é passo manual no console dela. Sem esta linha o
        # 500 seria mudo, e o operador só descobriria pelo webhook já desligado.
        logger.error("webhook whatsapp: %s ausente — o webhook vai recusar tudo", var)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"{var} não configurado",
        )
    return valor


def get_app_secret() -> str:
    """App Secret do app da Meta — a chave do HMAC que assina cada POST."""
    return _config("WHATSAPP_APP_SECRET")


def get_verify_token() -> str:
    """Token do handshake de verificação (GET), escolhido por nós e repetido no painel."""
    return _config("WHATSAPP_VERIFY_TOKEN")


def _compara(esperado: str, recebido: str | None) -> bool:
    """compare_digest sobre BYTES, com None tratado como "não confere".

    Em `str`, `compare_digest` estoura `TypeError` em qualquer caractere não-ASCII — e
    os dois valores que passam por aqui (o verify token na query string, o hex da
    assinatura no header) são escolhidos pelo cliente. Em `str`, um acento viraria 500
    em vez de 403. Mesmíssimo motivo do `.encode()` em `app/auth.py`.
    """
    if recebido is None:
        return False
    return hmac.compare_digest(esperado.encode("utf-8"), recebido.encode("utf-8"))


def verify_challenge(mode: str | None, token: str | None) -> bool:
    """Handshake de verificação: `hub.mode == "subscribe"` e o `hub.verify_token` bate.

    A comparação mora aqui, e não no `app/main.py`, para que os dois segredos do
    WhatsApp sejam manipulados num módulo só e a comparação seja de tempo constante,
    como a do Bearer. É o único ponto do sistema em que um segredo pode ser
    *adivinhado* por tentativa, o que é a razão de a rota GET ficar sob o teto por IP
    (ver o comentário na rota, em `app/main.py`).

    O token é lido ANTES de olhar o `mode` pelo mesmo motivo da ordem em
    `verify_signature`: config ausente tem que sair 500, não 403.
    """
    esperado = get_verify_token()
    if mode != _MODO_SUBSCRIBE:
        return False
    return _compara(esperado, token)


def verify_signature(raw_body: bytes, header: str | None) -> bool:
    """HMAC-SHA256 do corpo CRU contra o app secret. Header: "sha256=<hex>".

    O corpo tem que ser os bytes exatos que a Meta assinou — quem chama lê
    `await request.body()` antes de qualquer parse, porque `json.loads` seguido de
    re-serialização mudaria o que está sendo verificado.

    **O segredo é lido antes de olhar o header, e a ordem é a decisão.** Invertida,
    um request sem assinatura num ambiente mal configurado sairia 403 ("a Meta mandou
    errado") em vez de 500 ("nosso deploy está quebrado") — o mesmo disfarce que o 500
    do `API_TOKENS` existe pra impedir.

    Header ausente, sem o prefixo, ou com hex inválido devolvem False (não levantam):
    isso é request forjado, que é resposta 403, não incidente.
    """
    secret = get_app_secret().encode("utf-8")

    if not header or not header.startswith(_PREFIXO_ASSINATURA):
        return False

    # bytes.fromhex em vez de comparar as strings hex: fecha o mesmo buraco do
    # `_compara` (TypeError em não-ASCII num valor escolhido pelo cliente) e rejeita
    # de graça qualquer coisa que não seja hexadecimal.
    try:
        recebida = bytes.fromhex(header[len(_PREFIXO_ASSINATURA):])
    except ValueError:
        return False

    esperada = hmac.new(secret, raw_body, hashlib.sha256).digest()
    return hmac.compare_digest(esperada, recebida)


def mascarar_telefone(numero: str) -> str:
    """Últimos 4 dígitos: o bastante pra correlacionar dois eventos do mesmo remetente
    no log, sem gravar o número.

    Telefone e conteúdo de mensagem são PII e o log fica retido na plataforma; o que o
    operador precisa é reconhecer *o mesmo* remetente entre duas linhas, não saber
    quem é. O conteúdo da mensagem não vai pro log em nível nenhum — quem prova que o
    parse funciona é a suíte, não o log de produção.
    """
    return f"***{numero[-4:]}" if len(numero) > 4 else "***"


@dataclass(frozen=True)
class IncomingMessage:
    """Uma mensagem recebida, com só o que a borda precisa saber.

    `text` é None em tudo que não for `type == "text"` (áudio, imagem, botão, ...):
    a mensagem existe, mas não tem texto pra ler. A W2 é que decide o que fazer com
    isso — aqui só não se perde o evento.
    """

    wamid: str
    from_phone: str
    tipo: str
    text: str | None


def extract_messages(payload: object) -> list[IncomingMessage]:
    """Extrai as mensagens de um payload da Meta. NUNCA levanta.

    Defensiva por construção: `isinstance` em cada degrau de
    `entry[*].changes[*].value.messages[*]`, e formato inesperado vira lista vazia.
    Não é paranoia — a Meta manda no MESMO endpoint eventos que não são mensagem
    (`value.statuses`, os callbacks de entrega/leitura, que chegam a cada resposta que
    a W2 mandar) e adiciona campos sem aviso. Um `KeyError` aqui viraria não-2xx, e a
    Meta reenvia em não-2xx e desabilita a inscrição depois de falhas repetidas: um
    payload que não sabemos ler não pode derrubar a assinatura do webhook.
    """
    mensagens: list[IncomingMessage] = []

    if not isinstance(payload, dict):
        return mensagens

    for entry in _lista(payload.get("entry")):
        if not isinstance(entry, dict):
            continue
        for change in _lista(entry.get("changes")):
            if not isinstance(change, dict):
                continue
            value = change.get("value")
            if not isinstance(value, dict):
                continue
            for bruta in _lista(value.get("messages")):
                msg = _mensagem(bruta)
                if msg is not None:
                    mensagens.append(msg)

    return mensagens


def _lista(valor: object) -> list:
    return valor if isinstance(valor, list) else []


def _mensagem(bruta: object) -> IncomingMessage | None:
    """Uma entrada de `value.messages`, ou None se não der pra identificar.

    `id` e `from` são obrigatórios: sem os dois não há o que logar nem, na W2, pra
    quem responder — e um wamid vazio quebraria a deduplicação que ela vai precisar.
    """
    if not isinstance(bruta, dict):
        return None

    wamid = bruta.get("id")
    from_phone = bruta.get("from")
    # Truthiness, não só isinstance: `""` é str e passaria: um wamid vazio colidiria
    # com todos os outros na deduplicação da W2, e um telefone vazio sai do
    # `mascarar_telefone` como "***", indistinguível de um número mascarado de verdade.
    if not (isinstance(wamid, str) and wamid) or not (isinstance(from_phone, str) and from_phone):
        return None

    tipo = bruta.get("type")
    tipo = tipo if isinstance(tipo, str) else "desconhecido"

    texto = None
    if tipo == "text":
        corpo = bruta.get("text")
        if isinstance(corpo, dict) and isinstance(corpo.get("body"), str):
            texto = corpo["body"]

    return IncomingMessage(wamid=wamid, from_phone=from_phone, tipo=tipo, text=texto)
