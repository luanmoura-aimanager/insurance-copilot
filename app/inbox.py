"""Fatia W2b — a varredura que responde o que entrou pelo webhook.

A W2a fez a mensagem EXISTIR e mudou o significado do 200 para "aceito com durabilidade".
Essa promessa foi feita para este módulo: como a Meta considera a entrega concluída e nunca
mais a reenvia, quem responde somos nós, fora do request.

**Duas fases, e elas são a decisão da fatia:** `pending` -> `computed` (resposta gravada)
-> `answered` (enviada) | `failed`. A resposta é gravada ANTES de sair. O motivo é dinheiro:
se o envio falhar depois do grafo, um retry ingênuo re-pergunta e paga outra rodada de LLM.
Com a resposta em `whatsapp_message.answer`, a retentativa só reenvia.

**A varredura COMMITA, e esta é a SEGUNDA exceção consciente ao contrato de transação do
projeto** (a primeira é `embed_pending`). `persist_document` e `index_document` não commitam
porque repetir aquelas passadas é de graça — é CPU relendo texto que já está no banco. Aqui
o critério é o mesmo e o resultado é o oposto: a fronteira da transação acompanha o **custo
de repetir**, e repetir custa uma rodada de LLM *e* uma mensagem entregue de novo a uma
pessoa. Por isso o commit é por FASE, não por passada.

**Nem a pergunta nem a resposta saem daqui.** Não vão pro log (só o tamanho delas), não vão
pro `cost_event` (nem como `label`), e o erro de banco é montado à mão em vez de sair por
`logger.exception` — o UPDATE liga `answer` como parâmetro, e o texto de uma exceção do
SQLAlchemy inclui `[parameters: (...)]`. É a mesma regra do `_resumo_do_erro` da W2a.

**Disparo: `BackgroundTasks` do webhook + `scripts/process_pending.py` pra recuperação.** A
forma madura é um processo `worker:` no Procfile rodando a mesma função em laço, e a escolha
por BackgroundTask é consciente, com um custo conhecido: o trabalho morre com o processo (um
deploy no meio de uma varredura deixa linhas em `pending` ou `computed`), compete pelo event
loop do web process, e não tem agendamento próprio — uma varredura que morreu espera a
PRÓXIMA entrega ou uma execução manual do script. É aceitável porque o inbox durável, o
`SKIP LOCKED` e as duas fases tornam o trabalho **retomável por qualquer um**: o gatilho é
intercambiável, e trocá-lo por um worker muda um call site e nada da semântica.
"""
import logging
import os
import time
from uuid import uuid4

from langchain_core.messages import HumanMessage
from sqlalchemy import case, func, select, update

from app.agents.context import reset_request_context, set_request_context
from app.agents.answer import final_answer
from app.logging_config import resumo_do_erro, traceback_da_cadeia
from app.models import WhatsAppMessage
from app.whatsapp import id_curto, mascarar_telefone
from app.whatsapp_api import (
    MAX_BODY_CHARS,
    ConfiguracaoAusente,
    enviar_texto,
    get_access_token,
)

logger = logging.getLogger(__name__)

# Os quatro estados. Constantes e não literais espalhados: o check do banco
# (`ck_whatsapp_message_status`) é a autoridade, e um typo aqui viraria IntegrityError em
# produção em vez de erro de import.
PENDENTE = "pending"
COMPUTADA = "computed"
RESPONDIDA = "answered"
FALHOU = "failed"

LIMITE_PADRAO = 10

# Defaults conservadores, sobrescritos por env (ver `_inteiro_do_env`).
MAX_CHARS_PADRAO = 500
MAX_ATTEMPTS_PADRAO = 3

# A recusa da mensagem longa. RECUSAR, NUNCA TRUNCAR: uma pergunta cortada e respondida
# como se estivesse inteira produz uma resposta confiante para OUTRA pergunta — é o mesmo
# "índice que mente" da R2a entrando por outra porta, e aqui o resultado sai no telefone de
# alguém. O limite é interpolado do valor que de fato cortou, nunca de um literal na frase:
# um número fixo no texto viraria mentira no dia em que alguém mexesse no env, e a regra do
# "sem denominador não verificável" do rodapé vale aqui igual.
MENSAGEM_LONGA = (
    "Sua mensagem é longa demais para eu responder por aqui (o limite é de {limite} "
    "caracteres). Pode reenviar a pergunta numa versão mais curta?"
)


# Marca do corte do corpo de SAÍDA. Ver `_caber_no_whatsapp`.
RESPOSTA_TRUNCADA = "\n\n[resposta cortada: excedeu o limite de tamanho do WhatsApp]"


def _caber_no_whatsapp(resposta: str, wamid: str) -> str:
    """Garante que o corpo cabe no limite da Cloud API — cortando, e DIZENDO que cortou.

    **Isto não contradiz o "recusar, nunca truncar" da mensagem de entrada; é o oposto dele.**
    Lá o que seria truncado é a PERGUNTA, e responder a uma pergunta cortada como se estivesse
    inteira é responder com confiança a outra pergunta — o usuário não teria como saber. Aqui
    o que é cortado é a RESPOSTA, e o corte é anunciado no próprio texto que ele lê.

    Sem isto o caso ruim é silêncio total, não uma mensagem grande: a Cloud API recusa corpo
    acima de 4096 caracteres com 400, e como a resposta é DETERMINÍSTICA (está gravada em
    `answer`) as três tentativas falhariam idênticas e a linha viraria `failed`. O caminho que
    produz isso é real e não é exótico — quando o synthesizer cai, o grafo degrada para a
    saída CRUA do worker (`frase or resultado`): as k cláusulas do `rag_worker`, ou o despejo
    de até 100 linhas do `run_query`. Uma resposta feia é muito melhor do que nenhuma.

    O corte deixa espaço para a própria marca, senão o resultado estouraria o limite de novo.
    """
    if len(resposta) <= MAX_BODY_CHARS:
        return resposta
    logger.warning(
        "inbox: resposta cortada para caber no WhatsApp (id=%s, %d -> %d caracteres)",
        id_curto(wamid),
        len(resposta),
        MAX_BODY_CHARS,
    )
    return resposta[: MAX_BODY_CHARS - len(RESPOSTA_TRUNCADA)] + RESPOSTA_TRUNCADA


def _inteiro_do_env(var: str, default: int, minimo: int = 1) -> int:
    """Lê um inteiro do env NA CHAMADA, caindo no default se não der pra usar.

    Fail-**open** validado, cópia de `app/limits.py::_limit_from_env` e do `LOG_LEVEL`: um
    typo no env do Railway não pode derrubar a varredura nem, pior, virar `MAX_CHARS=0` e
    recusar todo mundo em silêncio. O aviso é o que impede o default de ser invisível.

    Contraste que vale registrar: o TOKEN é fail-CLOSED (`get_access_token`) e estes NÚMEROS
    são fail-open. Não é inconsistência — token ausente significa "nada pode ser entregue",
    e continuar gastaria dinheiro em respostas que ninguém vai receber; número inválido tem
    um default seguro que responde certo.
    """
    bruto = os.getenv(var, "").strip()
    if not bruto:
        return default
    try:
        valor = int(bruto)
    except ValueError:
        logger.warning("%s inválido (%r); usando o default %d", var, bruto, default)
        return default
    if valor < minimo:
        logger.warning(
            "%s = %d é menor que o mínimo %d; usando o default %d", var, valor, minimo, default
        )
        return default
    return valor


def _max_chars() -> int:
    return _inteiro_do_env("WHATSAPP_MAX_CHARS", MAX_CHARS_PADRAO)


def _max_attempts() -> int:
    return _inteiro_do_env("WHATSAPP_MAX_ATTEMPTS", MAX_ATTEMPTS_PADRAO)


# ---------------------------------------------------------------------------
# Os statements. Core e não ORM, de propósito: sem identity map e sem objetos que expiram no
# commit (esta função commita duas vezes por linha), e casa com o `pg_insert` que o webhook
# já usa em `_registrar_mensagens`.
# ---------------------------------------------------------------------------

_COLUNAS = (
    WhatsAppMessage.id,
    WhatsAppMessage.wamid,
    WhatsAppMessage.from_phone,
    WhatsAppMessage.text,
    WhatsAppMessage.phone_number_id,
    WhatsAppMessage.status,
    WhatsAppMessage.answer,
    WhatsAppMessage.attempts,
)


def _reivindicar(excluir: frozenset[int] = frozenset()):
    """A próxima linha a processar, TRAVADA — uma só, e a mais antiga.

    `excluir` são as linhas que ESTA passada já tocou. Sem isso, uma linha cujo envio falha
    volta pro filtro como `computed` e é reivindicada de novo na volta seguinte do MESMO
    laço — o orçamento inteiro de `attempts` queimado em milissegundos, o que não é
    retentativa nenhuma: não passou tempo, a condição transitória não teve chance de mudar, e
    a linha vira `failed` no primeiro soluço da Cloud API. **O intervalo entre tentativas é o
    intervalo entre varreduras**, e é o `excluir` que faz isso valer. (Verificado: sem ele,
    uma passada com `limite=5` sozinha aposenta a mensagem.)

    **`FOR UPDATE SKIP LOCKED` com `LIMIT 1`, e não um lote.** A trava do Postgres dura a
    TRANSAÇÃO, e esta varredura commita entre as fases; um `LIMIT 10 FOR UPDATE` perderia as
    dez travas no primeiro commit e uma varredura concorrente roubaria as linhas 2..10 —
    rodada de LLM em dobro e mensagem entregue duas vezes à mesma pessoa. Precedente do
    padrão: `app/rag/embed.py::_pendentes`, com o mesmo trade explicitado lá (a transação
    fica aberta durante a chamada paga).

    A alternativa madura seria uma coluna de *claim* commitada antes da chamada, e ela está
    fora por dois motivos: exigiria um quinto estado, e sem trava um processo que morre deixa
    a linha presa PARA SEMPRE (precisaria de um reaper com timeout). A trava do Postgres se
    cura sozinha — morreu a conexão, morreu a trava.

    `LIMIT 1` com `SKIP LOCKED` **não passa fome por causa do limite**: no plano do Postgres
    o `LockRows` fica ABAIXO do `Limit`, então linha pulada por trava não consome o limite —
    o executor continua puxando até conseguir uma. É isso que faz o padrão de fila funcionar
    com `LIMIT 1`.

    **Não filtra por `attempts < teto`, e isso é decisão.** Quem aposenta é o caminho de
    falha (`_contar_tentativa`). Com o predicado aqui, baixar `WHATSAPP_MAX_ATTEMPTS` deixaria
    linhas acima do novo teto invisíveis E não-`failed` — zumbis que ninguém contabiliza. Sem
    ele, essas linhas são pegas mais uma vez, entregam ou aposentam, e somem do filtro. Subir
    o teto também não ressuscita `failed`, que é o certo: revival é decisão de operador (um
    UPDATE na mão), não efeito colateral de env var.

    `computed` entra no filtro pra recuperar ÓRFÃ: um processo morto entre os dois commits
    deixa a resposta paga e não enviada, e ela tem que sair sem repassar pelo grafo.
    """
    q = select(*_COLUNAS).where(WhatsAppMessage.status.in_((PENDENTE, COMPUTADA)))
    if excluir:
        q = q.where(WhatsAppMessage.id.notin_(excluir))
    return (
        q.order_by(WhatsAppMessage.received_at, WhatsAppMessage.id)
        .limit(1)
        .with_for_update(skip_locked=True)
    )


def _rereivindicar(id_: int):
    """A MESMA linha, já computada, travada de novo pra atravessar o envio.

    **As duas metades daqui são o que impede envio duplo, e nenhuma cobre a outra.** A
    transação de envio (a) segura a TRAVA durante o POST e (b) reconfere `status='computed'`
    no WHERE, sob essa trava. Só a trava: depois que a outra varredura commita `answered` e
    solta, esta reivindicaria de novo e mandaria a segunda cópia. Só o status: as duas leriam
    `computed` ao mesmo tempo e enviariam as duas.

    O `status` precisa estar no SQL e não num `if` em Python por causa do EvalPlanQual: em
    READ COMMITTED, uma linha atualizada e commitada por outra transação é RE-CHECADA contra
    o WHERE antes de voltar. É o predicado no SQL que transforma "eu li isso há pouco" em
    "isto ainda é verdade agora".

    Voltar vazio significa "outra varredura está com ela" — segue-se em frente sem enviar.
    """
    return (
        select(*_COLUNAS)
        .where(WhatsAppMessage.id == id_, WhatsAppMessage.status == COMPUTADA)
        .with_for_update(skip_locked=True)
    )


def _gravar_resposta(id_: int, resposta: str):
    """Fim da fase 1: a resposta existe no banco ANTES de sair. É a fatia inteira."""
    return (
        update(WhatsAppMessage)
        .where(WhatsAppMessage.id == id_)
        .values(answer=resposta, status=COMPUTADA)
    )


def _marcar_respondida(id_: int):
    return (
        update(WhatsAppMessage)
        .where(WhatsAppMessage.id == id_)
        .values(status=RESPONDIDA, processed_at=func.now())
    )


def _contar_tentativa(id_: int, teto: int):
    """Gasta uma tentativa e, se foi a última, aposenta — no MESMO statement.

    `attempts` é uma contagem por MENSAGEM, compartilhada pelas duas fases: o orçamento é
    "quantas vezes tentamos responder esta pessoa". Ela sobe também quando o GRAFO falha, e
    não só o envio — sem isso a varredura passa fome, porque `ORDER BY received_at LIMIT 1`
    escolheria para sempre a mesma linha envenenada e nenhuma mensagem nova seria respondida.

    No `SET` do UPDATE **todas as expressões leem a linha ANTIGA** (o mesmo fato que o
    `DO UPDATE` do UPSERT da R2a já documenta), então as três colunas concordam sobre o valor
    de `attempts + 1`. O `RETURNING` evita uma segunda consulta pro log e pros contadores —
    mesmo argumento do `RETURNING` de `_registrar_mensagens`.

    `processed_at` é preenchido na aposentadoria porque ele marca estado TERMINAL, não "foi
    respondida": a retenção conta dias a partir dele, e uma linha `failed` com NULL ali seria
    PII que a política nunca conseguiria expirar.
    """
    esgotou = WhatsAppMessage.attempts + 1 >= teto
    return (
        update(WhatsAppMessage)
        # A guarda de status é obrigatória porque este UPDATE nem sempre roda sob a trava: no
        # caminho de erro de banco a transação que segurava a linha já foi desfeita, e entre o
        # rollback e a cobrança outra varredura pode ter reivindicado, reenviado e marcado
        # `answered`. Sem o `WHERE`, o `case((esgotou, FALHOU))` sobrescreveria um estado
        # TERMINAL: uma mensagem entregue (duas vezes) registrada como nunca entregue. É o
        # mesmo argumento do `AND status='computed'` de `_rereivindicar` — lá pra não enviar
        # duas vezes, aqui pra não desfazer o que a outra varredura concluiu.
        .where(WhatsAppMessage.id == id_,
               WhatsAppMessage.status.in_((PENDENTE, COMPUTADA)))
        .values(
            attempts=WhatsAppMessage.attempts + 1,
            status=case((esgotou, FALHOU), else_=WhatsAppMessage.status),
            processed_at=case((esgotou, func.now()), else_=WhatsAppMessage.processed_at),
        )
        .returning(WhatsAppMessage.status, WhatsAppMessage.attempts)
    )


def _aposentar(id_: int):
    """Vai direto pra `failed` — pra falha que retry nenhum conserta.

    Hoje o único caso é a linha sem `phone_number_id`: não há de qual número responder, e
    tentar três vezes só adiaria o mesmo desfecho, então a aposentadoria é imediata em vez de
    gastar o orçamento. `attempts` ainda sobe: a coluna conta TENTATIVAS FEITAS, e houve uma —
    manter zero diria "nunca foi tentada", que é falso e quebraria qualquer relatório que use
    `attempts = 0` pra achar o que ainda não foi processado.
    """
    return (
        update(WhatsAppMessage)
        .where(WhatsAppMessage.id == id_,
               WhatsAppMessage.status.in_((PENDENTE, COMPUTADA)))
        .values(status=FALHOU, attempts=WhatsAppMessage.attempts + 1, processed_at=func.now())
        .returning(WhatsAppMessage.status, WhatsAppMessage.attempts)
    )


# ---------------------------------------------------------------------------
# A varredura
# ---------------------------------------------------------------------------


def _contagens_zeradas() -> dict:
    return {
        "reivindicadas": 0,   # linhas que ESTA passada travou
        "computadas": 0,      # respostas gravadas agora (grafo OU recusa)
        "recusadas": 0,       # subconjunto de computadas: longas demais, custo zero de LLM
        "enviadas": 0,        # chegaram a 'answered'
        "falhas": 0,          # tentativas que terminaram sem entregar nesta passada
        "desistidas": 0,      # subconjunto de falhas: bateram no teto e viraram 'failed'
        "cedidas": 0,         # reivindicadas e perdidas pra outra varredura no meio
        "interrompidas": 0,   # a passada PAROU por erro de banco (0 ou 1)
    }


async def processar_pendentes(session, limite=LIMITE_PADRAO, graph=None, sender=None) -> dict:
    """Responde até `limite` mensagens. COMMITA (ver o docstring do módulo).

    `limite` é o teto do LAÇO — quantas linhas DISTINTAS esta passada reivindica —, não um
    `LIMIT n` no SQL: cada volta refaz o `LIMIT 1` excluindo o que já foi tocado, e essa
    query *é* o cursor (mesma frase do `embed_pending`: sem offset pra desalinhar). Distintas
    porque cada linha é tentada NO MÁXIMO UMA VEZ por passada — ver `_reivindicar`. A passada termina cedo quando a
    reivindicação volta vazia, o que significa "não há nada a fazer" **ou** "tudo que resta
    está travado por outra varredura" — efeito colateral consciente do `SKIP LOCKED`, e está
    certo: o outro processo está cuidando delas.

    `graph` e `sender` são injetáveis porque são os dois seams de teste; nenhum teste gasta
    dinheiro nem manda mensagem. O objeto real do grafo é resolvido NA CHAMADA, pelo mesmo
    motivo do import tardio de `app.db` em `record_call_cost`.

    Ordem por linha, e ela é a fatia:

        tx1  SELECT ... LIMIT 1 FOR UPDATE SKIP LOCKED       <- trava
             sem phone_number_id?   -> aposenta                    [não paga LLM]
             já 'computed'?         -> nada a computar (órfã)
             len(text) > MAX_CHARS? -> answer = a recusa           [não paga LLM]
             senão                  -> answer = final_answer(await graph.ainvoke(...))
             UPDATE answer, status='computed';  COMMIT       <- a resposta é durável
        tx2  SELECT ... WHERE id AND status='computed' FOR UPDATE SKIP LOCKED
             await sender(...)                               <- trava segurada aqui
             UPDATE status='answered', processed_at=now();  COMMIT

    **O envio é at-least-once, e isso é escolha.** Se o POST dá certo e o COMMIT de
    `answered` falha, a próxima varredura reenvia e a pessoa recebe duas vezes. A
    alternativa — marcar `answered` antes de enviar — troca duplicata por SILÊNCIO, que é
    exatamente a falha que a W2a inteira existe pra impedir. A janela é um commit depois de
    um HTTP bem-sucedido, e `attempts` limita o estrago.

    **`FALHA_INTERNA` vindo do grafo é RESPOSTA, não falha da varredura.** O grafo não levanta
    em falha de worker — devolve a frase —, então ela é enviada e a linha vira `answered`.
    Discriminar pelo texto seria controle de fluxo por string mágica, que é justamente o que o
    projeto rejeitou ao marcar a falha com `name="worker_error"` em vez de prefixo no
    conteúdo. No dia do worker process (que tem gatilho próprio) retentar aí fica atraente, e
    o discriminante certo continua sendo o `name` da mensagem.
    """
    if limite < 1:
        # Mesma lição do `_validar_limit` da R2b: `limite=0` sairia pelo laço na primeira
        # volta e devolveria um relatório de zeros indistinguível de "não havia nada a
        # fazer" — mentira plausível num caminho que gasta dinheiro.
        raise ValueError(f"limite tem que ser >= 1 (veio {limite})")

    if sender is None:
        # Fail-closed cobrado UMA VEZ, antes da primeira reivindicação. Sem isto, um deploy
        # sem token pagaria o grafo por todas as pendentes e depois queimaria `attempts` em
        # todas até `failed` — perder perguntas por erro de configuração. É o mesmo movimento
        # do `embed_pending` recusando antes da primeira chamada paga.
        get_access_token()
        sender = enviar_texto

    if graph is None:
        # Import tardio, mas o motivo NÃO é adiar a cadeia do grafo: `app.agents.answer` já a
        # traz no import deste módulo (ela precisa do `NO_ANSWER`), então importar `app.inbox`
        # nunca foi barato. O que este import compra é o SEAM: `graph` é resolvido no momento
        # da chamada, e não amarrado ao objeto que existia no import.
        from app.agents.graph import graph as graph_real

        graph = graph_real

    contagens = _contagens_zeradas()
    teto = _max_attempts()
    max_chars = _max_chars()

    vistas: set[int] = set()
    for _ in range(limite):
        linha = (await session.execute(_reivindicar(frozenset(vistas)))).mappings().first()
        if linha is None:
            break
        vistas.add(linha["id"])
        contagens["reivindicadas"] += 1

        # `request_id` NOVO por linha, e `client=None`. A varredura roda fora de um request,
        # então sem isto o `cost_event` e as linhas de log dela ficariam sem correlação
        # nenhuma — e com várias mensagens no mesmo lote seria impossível dizer qual chamada
        # foi de qual pergunta. `client` fica NULL porque ninguém se autenticou: a coluna
        # continua significando "identidade do token", e inventar um valor a faria significar
        # duas coisas. O reset no `finally` é obrigatório (o mesmo do `/ask`).
        ctx = set_request_context(str(uuid4()), None)
        try:
            continuar = await _uma_linha(session, linha, graph, sender, teto, max_chars, contagens)
        finally:
            reset_request_context(ctx)
        if not continuar:
            break

    # `_reivindicar` é um SELECT ... FOR UPDATE, então ele ABRE transação mesmo voltando
    # vazio — e a passada sai por aí (nada pendente, ou tudo travado por outra varredura).
    # Sem este rollback a conexão fica `idle in transaction` entre as rodadas de
    # `varrer_em_background` e até o fim do `async with`, que é exatamente onde um Postgres
    # gerenciado com `idle_in_transaction_session_timeout` derruba a sessão. Mesma lição do
    # `rollback()` por pergunta em `scripts/calibrate_search.py`.
    try:
        if session.in_transaction():
            await session.rollback()
    except Exception:
        logger.warning("inbox: falha ao soltar a transação no fim da passada")

    return contagens


async def _uma_linha(session, linha, graph, sender, teto, max_chars, contagens) -> bool:
    """Processa UMA linha já reivindicada. Devolve False quando a varredura deve parar."""
    inicio = time.monotonic()
    id_ = linha["id"]
    wamid = linha["wamid"]
    recusada = False

    if linha["status"] == PENDENTE:
        if not linha["phone_number_id"]:
            # Não há de qual número responder, e o payload que trazia o `value.metadata` já
            # não existe. Aposenta ANTES do grafo: pagar por uma resposta impossível de
            # entregar é queimar dinheiro. Um env de fallback está fora de propósito — é
            # exatamente o "fixar um número no código" que o comentário da W1 existe pra
            # impedir, e que erraria em silêncio no dia do segundo número.
            logger.error(
                "inbox: mensagem sem phone_number_id, impossível responder (id=%s)",
                id_curto(wamid),
            )
            return await _falhar(session, id_, contagens, linha, inicio, aposentar=True)

        if len(linha["text"]) > max_chars:
            resposta = MENSAGEM_LONGA.format(limite=max_chars)
            recusada = True
        else:
            try:
                estado = await graph.ainvoke({
                    "iterations": 0,
                    "next": "",
                    "messages": [HumanMessage(content=linha["text"])],
                })
                # `final_answer` fica DENTRO deste `try`, e não depois dele: ela lê
                # `state["messages"][-1]`, e um estado malformado (um caminho de erro do
                # LangGraph, uma mudança futura no grafo) sairia como `KeyError`/`IndexError`
                # por cima de `_uma_linha` e de `processar_pendentes` inteiros — sem tentativa
                # contada, ou seja exatamente a inanição que o `attempts` existe pra impedir, e
                # com traceback cru no `scripts/process_pending.py`.
                resposta = _caber_no_whatsapp(final_answer(estado), wamid)
            except Exception as exc:
                # O grafo trata falha de WORKER internamente (vira `FALHA_INTERNA`); o que
                # chega aqui é o que ele não cobre — o supervisor, que não tem `try` — mais um
                # estado malformado pego pelo `final_answer` acima. Conta tentativa: sem isso
                # esta linha seria reivindicada primeiro em toda varredura, para sempre, e a
                # fila inteira morreria atrás dela.
                logger.error(
                    "inbox: o grafo falhou (id=%s) — %s\n%s",
                    id_curto(wamid),
                    type(exc).__name__,
                    traceback_da_cadeia(exc),
                )
                return await _falhar(session, id_, contagens, linha, inicio, teto=teto)

        try:
            await session.execute(_gravar_resposta(id_, resposta))
            await session.commit()
        except Exception as exc:
            # A resposta JÁ FOI PAGA e não coube no banco. Cobrar a tentativa é obrigatório:
            # sem isso a linha continua `pending` com `attempts` intacto, e a próxima
            # varredura repaga o grafo pela MESMA pergunta, para sempre.
            return await _erro_de_banco(
                session, "gravar a resposta", linha, exc, contagens, teto=teto, inicio=inicio
            )
        contagens["computadas"] += 1
        if recusada:
            contagens["recusadas"] += 1
    else:
        # Órfã: já `computed` (processo morto entre os dois commits). Nada a computar, e a
        # resposta já foi paga. Solta a transação de leitura pra que a fase de envio abra a
        # sua — um caminho só, sempre duas transações: dois formatos de fluxo fariam uma
        # mutação que quebre a re-reivindicação ficar vermelha só num deles.
        try:
            await session.commit()
        except Exception as exc:
            # Nada pago nesta volta (a resposta da órfã já estava no banco), então não há
            # tentativa a cobrar — só parar.
            return await _erro_de_banco(session, "soltar a linha órfã", linha, exc, contagens)

    # --- fase 2: enviar, com a linha travada de novo ------------------------
    try:
        alvo = (await session.execute(_rereivindicar(id_))).mappings().first()
    except Exception as exc:
        return await _erro_de_banco(session, "re-reivindicar a linha", linha, exc, contagens)

    if alvo is None:
        # Outra varredura pegou a linha entre os dois commits. Está certo: ela vai enviar.
        contagens["cedidas"] += 1
        try:
            await session.commit()
        except Exception as exc:
            return await _erro_de_banco(session, "soltar a linha cedida", linha, exc, contagens)
        # Loga como qualquer outro fim de linha: sem isto, uma linha que ESTA passada
        # reivindicou some do log dela entre a reivindicação e o resumo — e no caso de dois
        # processos varrendo (onde o SKIP LOCKED é a defesa inteira) é justamente o evento
        # que o operador quer ver.
        _logar(wamid, linha, "cedida", linha["attempts"], 0, inicio)
        return True

    try:
        wamid_enviado = await sender(alvo["phone_number_id"], alvo["from_phone"], alvo["answer"])
    except ConfiguracaoAusente as exc:
        # NÃO é falha DESTA mensagem, e por isso não gasta tentativa dela. `enviar_texto` lê o
        # token a cada envio (ele é rotacionável), então ele pode sumir DEPOIS do fail-closed
        # de `processar_pendentes` — entre duas rodadas, ou no meio de um lote. Cobrando aqui,
        # cada mensagem restante levaria uma tentativa e o lote inteiro seria aposentado como
        # `failed`: exatamente o "perder perguntas por erro de configuração" que o fail-closed
        # existe pra impedir, só que mais tarde. A passada para; a linha continua `computed`.
        logger.error(
            "inbox: envio impossível por CONFIGURAÇÃO (id=%s) — a varredura PARA, e nenhuma "
            "tentativa é gasta — %s",
            id_curto(wamid),
            exc,
        )
        try:
            await session.rollback()
        except Exception:
            pass
        return False
    except Exception as exc:
        # WARNING e não ERROR: falha de envio é esperada e retentável, e o `attempts` é o
        # que a transforma em incidente quando insiste. Sem `str(exc)` e sem `exc_info` — a
        # exceção do httpx segura a request, cujo corpo tem o telefone e a resposta inteira.
        logger.warning(
            "inbox: envio recusado (id=%s, tentativa=%d/%d) — %s%s",
            id_curto(wamid),
            alvo["attempts"] + 1,
            teto,
            type(exc).__name__,
            f" http={exc.status}" if getattr(exc, "status", None) else "",
        )
        return await _falhar(session, id_, contagens, alvo, inicio, teto=teto)

    try:
        await session.execute(_marcar_respondida(id_))
        await session.commit()
    except Exception as exc:
        # A mensagem JÁ CHEGOU na pessoa; o que falhou foi registrar isso. A próxima varredura
        # vai reenviar — é a janela at-least-once —, e é por isso que a tentativa é COBRADA
        # aqui: sem ela a linha fica `computed` com `attempts` intacto e cada varredura futura
        # reenvia a MESMA mensagem, sem teto nenhum. Cobrar transforma "pode duplicar" em
        # "duplica no máximo `WHATSAPP_MAX_ATTEMPTS` vezes", que é o que a documentação promete.
        return await _erro_de_banco(
            session, "marcar como respondida (a mensagem SAIU)", alvo, exc, contagens,
            teto=teto, inicio=inicio, entregue=True,
        )

    contagens["enviadas"] += 1
    _logar(wamid, alvo, RESPONDIDA, alvo["attempts"], len(alvo["answer"] or ""), inicio,
           wamid_enviado=wamid_enviado)
    return True


async def _falhar(
    session, id_, contagens, linha, inicio, *, teto=None, aposentar=False, entregue=False
) -> bool:
    """**O ÚNICO lugar que gasta orçamento e aposenta.** Devolve False se a varredura deve parar.

    Existe um caminho só de propósito. A política de aposentadoria (quando vira `failed`, o
    que conta como desistência, o que sai no log) estava escrita duas vezes — aqui e dentro
    do tratamento de erro de banco — e duas cópias divergem: um contador novo, uma redação
    diferente, a guarda de status de `_contar_tentativa`, tudo teria que ser aplicado em dois
    lugares. É a mesma regra que tirou o `final_answer` de `app/main.py`.

    Normalmente roda na MESMA transação que ainda segura a trava — nada de `rollback()` antes,
    porque soltar a trava abriria uma janela pra outra varredura pagar o grafo de novo pela
    mesma linha. Vindo de `_erro_de_banco` roda depois do rollback, SEM trava; quem protege
    esse caso é a guarda de status dentro dos statements.

    `entregue=True` significa "a mensagem SAIU e só o registro falhou", e muda o que se diz: a
    linha de ERROR padrão afirma que alguém ficou sem resposta, e é a única evidência disso
    que existe — usá-la aqui mandaria o operador procurar um usuário que na verdade foi
    respondido (talvez mais de uma vez).
    """
    contagens["falhas"] += 1
    stmt = _aposentar(id_) if aposentar else _contar_tentativa(id_, teto)
    try:
        resultado = (await session.execute(stmt)).first()
        await session.commit()
    except Exception as exc:
        # A própria contabilidade da falha falhou — transação abortada, ou seja banco doente.
        # `teto` fica de fora de propósito: `_erro_de_banco` chamaria `_falhar` de volta e a
        # recursão seria mútua. Aqui só se registra e para.
        return await _erro_de_banco(session, "registrar a tentativa", linha, exc, contagens)

    if resultado is None:
        # A guarda de status barrou: outra varredura já levou a linha a um estado terminal
        # entre o rollback e esta cobrança. Não há o que corrigir — ela cuidou.
        logger.info(
            "inbox: a tentativa não foi cobrada (id=%s) — outra varredura já concluiu a linha",
            id_curto(linha["wamid"]),
        )
        return True

    novo_status, tentativas = resultado
    if novo_status == FALHOU:
        if entregue:
            # NÃO conta como desistência: a pessoa recebeu. O que se perdeu foi o registro.
            logger.error(
                "inbox: a mensagem (id=%s) FOI ENTREGUE mas o registro falhou %d vez(es) — a "
                "linha vira 'failed' e não será reenviada; ela pode ter sido entregue mais de "
                "uma vez",
                id_curto(linha["wamid"]),
                tentativas,
            )
        else:
            contagens["desistidas"] += 1
            # ERROR e com a franqueza do `PERDIDAS` da W2a: esta linha é o ÚNICO registro de
            # que alguém perguntou e não foi respondido. Sem ela, restaria um número.
            logger.error(
                "inbox: desistindo da mensagem (id=%s) após %d tentativa(s) — este usuário NÃO "
                "recebeu resposta",
                id_curto(linha["wamid"]),
                tentativas,
            )
    _logar(linha["wamid"], linha, novo_status, tentativas, len(linha["answer"] or ""), inicio)
    return True


async def _erro_de_banco(
    session, o_que: str, linha, exc: BaseException, contagens, *,
    teto=None, inicio=None, entregue=False,
) -> bool:
    """Loga um erro de banco SEM PII, COBRA a tentativa quando havia trabalho pago, e para.

    `logger.error` + resumo montado à mão, e **nunca** `logger.exception`: o UPDATE liga
    `answer` como parâmetro, então o texto da exceção do SQLAlchemy carregaria a resposta
    inteira em `[parameters: (...)]`. O resumo é o MESMO de `app/main.py`
    (`app.logging_config.resumo_do_erro`) porque os dois obedecem à mesma regra — duas cópias
    divergiriam no dia em que uma delas fosse endurecida.

    **`teto` é o que separa os dois tipos de erro daqui, e a diferença é dinheiro.** Nos
    caminhos em que nada foi pago (soltar a linha órfã, re-reivindicar) não há tentativa a
    cobrar: a linha volta intacta e a próxima varredura recomeça de graça. Nos DOIS em que já
    houve gasto — a resposta do grafo que não coube no banco, e o `answered` que não commitou
    depois de a mensagem ter SAÍDO — deixar `attempts` intacto transformaria uma falha
    transitória em repetição sem teto: repagar o grafo pela mesma pergunta em toda varredura,
    no primeiro caso, e reenviar a mesma mensagem à mesma pessoa para sempre, no segundo. É
    essa cobrança que torna verdadeira a frase "o `attempts` limita o estrago".

    A cobrança roda numa transação NOVA, depois do rollback (a que estourou está abortada e
    não aceita mais statement) e portanto SEM a trava — daí a guarda de status nos statements.
    """
    contagens["interrompidas"] += 1
    logger.error(
        "inbox: falha de banco ao %s (id=%s) — a varredura PARA — %s\n%s",
        o_que,
        id_curto(linha["wamid"]),
        resumo_do_erro(exc),
        traceback_da_cadeia(exc),
    )

    try:
        await session.rollback()
    except Exception:
        logger.warning("inbox: o rollback também falhou — %s", type(exc).__name__)
        return False

    if teto is not None:
        try:
            await _falhar(
                session, linha["id"], contagens, linha, inicio or time.monotonic(),
                teto=teto, entregue=entregue,
            )
        except Exception as cobranca_exc:
            # Fica registrado que o teto NÃO foi aplicado nesta volta: é o que explica uma
            # repetição, e insistir não ajuda — se a cobrança falhou, o banco está fora.
            logger.error(
                "inbox: a tentativa NÃO pôde ser cobrada (id=%s) — o teto não se aplica a "
                "esta volta e a próxima varredura vai repetir o trabalho — %s",
                id_curto(linha["wamid"]),
                type(cobranca_exc).__name__,
            )

    return False


def _logar(wamid, linha, status, tentativas, caracteres, inicio, wamid_enviado=None) -> None:
    """Uma linha por mensagem processada. Sem a pergunta, sem a resposta, sem o wamid cru.

    `id_curto` porque o wamid EMBUTE o telefone em base64 (ver `app/whatsapp.py`), telefone
    mascarado, e da resposta só o TAMANHO — exatamente o que a linha de INFO da W1 faz com o
    texto recebido. O que o operador precisa é reconhecer a mesma mensagem entre duas linhas
    e saber em que estado ela parou; quem prova que a resposta está certa é a suíte.
    """
    logger.info(
        "inbox: mensagem processada (id=%s, de=%s, status=%s, tentativas=%d, caracteres=%d, "
        "%.0fms%s)",
        id_curto(wamid),
        mascarar_telefone(linha["from_phone"]),
        status,
        tentativas,
        caracteres,
        (time.monotonic() - inicio) * 1000,
        # O id que a Meta deu à NOSSA mensagem, também em digest. É o único elo entre esta
        # resposta e os `value.statuses` (entregue/lido) que vão chegar no mesmo webhook —
        # sem ele, `enviar_texto` parseia o wamid de volta para ninguém. Digest pelo mesmo
        # motivo do outro: wamid embute telefone em base64.
        f", resposta={id_curto(wamid_enviado)}" if wamid_enviado else "",
    )


async def contar_a_fazer(session) -> dict:
    """Quantas linhas em cada estado. Para o `--dry-run` — e **sem `FOR UPDATE`**.

    Travar linhas num ensaio bloquearia a passada de verdade, que é a mesma regra do
    `pending_stats` da R2b.
    """
    r = await session.execute(
        select(WhatsAppMessage.status, func.count())
        .group_by(WhatsAppMessage.status)
    )
    por_status = dict(r.all())
    return {
        "pendentes": por_status.get(PENDENTE, 0),
        "computadas": por_status.get(COMPUTADA, 0),
        "respondidas": por_status.get(RESPONDIDA, 0),
        "falhas": por_status.get(FALHOU, 0),
    }


# Quantas passadas um único disparo encadeia. Ver `varrer_em_background`.
MAX_RODADAS = 5

# Uma varredura em voo por processo. NÃO é `Lock`: quem chega e encontra ocupado DESISTE em
# vez de esperar — a que está rodando refaz o `LIMIT 1` a cada volta e portanto já vai pegar
# as linhas que acabaram de entrar. Enfileirar seria pior de propósito: cada varredura segura
# uma conexão do pool com transação ABERTA durante a chamada ao grafo e o POST à Cloud API,
# então N entregas simultâneas viravam N conexões travadas — com o pool default (5 + 10) uma
# rajada da Meta esgotava o pool, o `SessionLocal()` do PRÓPRIO webhook estourava
# `TimeoutError`, `_vale_reentregar` classificava isso como transitório, a rota devolvia 500,
# a Meta reentregava e agendava mais varreduras. Um laço de realimentação que termina com a
# inscrição desabilitada — exatamente o desfecho que a âncora de rate limit existe pra impedir,
# entrando pela porta de dentro.
#
# Boolean simples e não semáforo porque a checagem e a marcação acontecem sem `await` entre
# elas, num único event loop: não há corrida a proteger. É por PROCESSO, não global — dois web
# processes podem varrer ao mesmo tempo, e é aí que o `SKIP LOCKED` faz o trabalho.
_varrendo = False

# "Alguem pediu uma varredura enquanto esta rodava." O disparo do webhook e engolido pelo
# `_varrendo`, e sem esta marca as mensagens daquela entrega ficariam esperando a PROXIMA —
# que pode nao vir. A varredura em curso le a marca antes de sair e da mais uma volta.
_pedida_de_novo = False


async def varrer_em_background(limite: int = LIMITE_PADRAO) -> None:
    """O que o webhook agenda depois do 200. **NUNCA levanta**, e roda UMA POR VEZ.

    Abre a PRÓPRIA sessão: quando esta função roda, a do request já foi fechada e a conexão
    devolvida ao pool. Import tardio de `app.db` pelo mesmo motivo de sempre — resolve
    `SessionLocal` no momento da chamada, que é o que faz o monkeypatch dos testes pegar.

    Não levanta porque a exceção subiria DEPOIS de a resposta ter sido enviada: o
    `RequestIdMiddleware` a logaria e re-levantaria, e em produção isso vira um "Exception in
    ASGI application" fantasma pendurado num request que deu 200. O canal do operador aqui é
    o log, como em todo o resto desta superfície.

    **É também o seam único da suíte** — `tests/conftest.py` troca esta função por um no-op
    autouse, pra que nenhum teste chame o grafo (e a Anthropic) por acidente ao POSTar no
    webhook. Ver o comentário lá.
    """
    global _varrendo, _pedida_de_novo

    # DOIS blocos, e a divisão é load-bearing nas duas direções.
    #
    # (1) A saída antecipada (já tem varredura em curso) NÃO pode passar por um `finally` que
    #     solta o `_varrendo`: ela soltaria a trava de OUTRA corotina, que continua rodando, e
    #     a próxima entrega abriria uma segunda passada em paralelo — exatamente o
    #     esgotamento de pool que a trava existe pra impedir. Aconteceu com um `try/finally`
    #     único, e quem mostrou foi `test_pedido_durante_a_varredura_nao_e_perdido` (o
    #     `_pedida_de_novo` voltava a False na saída antecipada). Então o `finally` só começa
    #     DEPOIS da reivindicação da trava.
    # (2) Mas o docstring promete que esta função NUNCA levanta, e um `ImportError` de
    #     `app.db` (deploy parcial, ciclo de import novo) escaparia pela BackgroundTask e
    #     viraria o "Exception in ASGI application" fantasma pendurado num request que já
    #     devolveu 200. Daí este primeiro bloco ser `try/except` — sem `finally`.
    try:
        if _varrendo:
        # Já tem uma rodando, e ela vai pegar o que acabou de entrar (a reivindicação é
        # refeita a cada volta). Ver o comentário de `_varrendo`: esperar aqui é o que
        # esgotava o pool numa rajada da Meta.
            # O pedido NÃO é descartado — fica marcado, e a que está rodando dá mais uma
            # volta antes de sair. Sem isso, a entrega que chegou durante uma varredura em
            # curso esperaria a PRÓXIMA, que pode não vir.
            _pedida_de_novo = True
            logger.debug("inbox: varredura já em curso, este disparo será atendido por ela")
            return

        from app.db import SessionLocal

        _varrendo = True
        _pedida_de_novo = False
    except Exception as exc:
        logger.error(
            "inbox: a varredura em background não conseguiu começar — %s\n%s",
            type(exc).__name__,
            traceback_da_cadeia(exc),
        )
        return

    try:
        totais = _contagens_zeradas()
        async with SessionLocal() as session:
            for _ in range(MAX_RODADAS):
                r = await processar_pendentes(session, limite=limite)
                for chave, valor in r.items():
                    totais[chave] += valor
                # Encadeia SÓ enquanto a rodada foi cheia E inteiramente bem-sucedida. O teto
                # `limite` sozinho deixaria um lote grande da Meta (ela entrega em lote, e a
                # W2a grava o lote todo) parado até a PRÓXIMA entrega, porque o agendamento só
                # dispara de uma entrega com pergunta. E a segunda condição é o que impede o
                # encadeamento de virar retentativa imediata: as linhas que falharam não estão
                # mais no `vistas` de uma passada nova, então uma rodada seguinte as pegaria de
                # volta e queimaria o orçamento em milissegundos — o mesmo bug que o `excluir`
                # de `_reivindicar` mata dentro de uma passada.
                if r["enviadas"] < limite:
                    # Rodada nao-cheia ou com falha: nao encadeia por conta propria. Mas se
                    # CHEGOU PEDIDO enquanto rodavamos, a linha nova pode estar depois do
                    # ponto onde esta rodada parou — e o disparo dela foi engolido pelo
                    # `_varrendo`. Uma volta a mais, e so uma por pedido, senao duas entregas
                    # concorrentes viravam um laco que se realimenta.
                    if _pedida_de_novo:
                        _pedida_de_novo = False
                        continue
                    break
        logger.info(
            "inbox: varredura concluída (%d reivindicada(s), %d enviada(s), %d falha(s), "
            "%d desistida(s), %d cedida(s))",
            totais["reivindicadas"], totais["enviadas"], totais["falhas"],
            totais["desistidas"], totais["cedidas"],
        )
    except Exception as exc:
        logger.error(
            "inbox: a varredura em background falhou — %s\n%s",
            type(exc).__name__,
            traceback_da_cadeia(exc),
        )
    finally:
        _varrendo = False
        _pedida_de_novo = False
