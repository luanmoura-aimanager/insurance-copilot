"""O único ponto que fala com a API de embedding (Voyage). Sem banco, sem ORM.

**Modelo: `voyage-4-lite`, 1024 dimensões.** A escolha já está gravada no schema — a
dimensão vive no *tipo* da coluna (`vector(1024)`), porque o Postgres precisa dela pra
indexar —, então trocar de modelo é migration, não config. O critério foi: multilíngue
(o corpus é pt-BR), barato, e dimensão pequena o bastante pra deixar o índice HNSW leve.

**`input_type` é o detalhe que faz a busca funcionar.** A Voyage treina o modelo de
forma *assimétrica*: documento e pergunta são embeddados com prefixos diferentes, e o
espaço é otimizado pra que uma *pergunta* fique perto dos *documentos* que a respondem
— não perto de outras perguntas. Por isso as duas funções públicas daqui não são
intercambiáveis: `embed_documents` indexa cláusulas com `input_type="document"` (R2b) e
`embed_query` embedda a pergunta com `input_type="query"` (R3a). É o par que precisa
bater: usar o mesmo dos dois lados degrada a recuperação em silêncio, sem erro nenhum
pra denunciar — nenhuma dimensão muda, nenhuma exceção sobe, só os vizinhos ficam
piores. Daí os dois lados estarem travados por teste.

**Rate limit é a única falha retentada aqui** (`_embed_com_retry`): é a única que se
resolve sozinha esperando, porque é uma janela que reabre. Todo o resto sobe na hora —
mascarar um erro real atrás de 6 tentativas só faz o backfill levar minutos pra dar a
mesma mensagem.

**`truncation=False`, contra o padrão do SDK.** Com o padrão (`True`), um texto maior
que o contexto do modelo é *cortado em silêncio* e embeddado pela metade. A linha
resultante fica indistinguível de uma correta — `clause_chunk.text` guarda a cláusula
inteira, o vetor cobre só o começo dela, e `embedding_model` diz `voyage-4-lite` como em
todas as outras. Aí a busca casa por um prefixo e cita a cláusula toda: exatamente o
"índice que mente" que a R2a já trata como inaceitável, entrando pela última porta que
faltava. Hoje isso é inalcançável (o maior chunk do corpus tem 934 caracteres), mas a
R2a **não tem splitter por decisão medida** — o dia em que o corpus trouxer uma cláusula
longa é o dia em que este parâmetro decide entre um erro barulhento e um índice podre.
"""

import logging
import os
import random
import time
from functools import lru_cache

import voyageai
import voyageai.error

from app.models import EMBEDDING_DIM

logger = logging.getLogger(__name__)

EMBED_MODEL = "voyage-4-lite"

# Teto de itens por requisição da própria API — o CLI valida `--batch-size` contra ele.
MAX_BATCH_API = 1000

# O teto da API é 1000 textos por requisição; 200 é deliberadamente menor. O que se
# perde num lote grande demais é granularidade de falha e de custo: cada lote é uma
# linha de `cost_event` e um commit, então um lote de 1000 que estoura no fim joga fora
# uma chamada inteira já paga. Com os ~4.4k chunks do corpus atual isso dá 22 chamadas.
MAX_BATCH = 200

# Os dois lados do modelo assimétrico — ver o docstring do módulo. Não são
# intercambiáveis: cláusula indexada é "document", pergunta de busca é "query".
INPUT_TYPE_DOCUMENT = "document"
INPUT_TYPE_QUERY = "query"

# Backoff do rate limit. Os números são grandes de propósito: o limite da Voyage é por
# minuto, então esperar 1s não resolve nada — só queima tentativa. 6 tentativas com base
# 20s dobrando até o teto de 120s cobrem ~7 minutos de janela fechada, que é mais do que
# qualquer reset por minuto precisa.
RATE_LIMIT_TENTATIVAS = 6
RATE_LIMIT_ESPERA_INICIAL = 20.0
RATE_LIMIT_FATOR = 2.0
RATE_LIMIT_ESPERA_MAX = 120.0

# Orçamento SEPARADO para a busca, e bem menor. Os números acima são de um backfill
# offline, onde ninguém espera e sete minutos de paciência são baratos; a busca roda
# dentro de um request, segurando um worker do `asyncio.to_thread` durante a chamada
# inteira. Herdar 6 tentativas ali significaria um `/ask` pendurado por minutos e, com
# algumas buscas simultâneas, o executor padrão esgotado — o que trava também as
# chamadas síncronas ao Postgres dos outros nós do grafo. Uma retentativa curta cobre o
# 429 isolado; passou disso, é melhor falhar rápido e devolver o erro.
RATE_LIMIT_TENTATIVAS_QUERY = 2
RATE_LIMIT_ESPERA_QUERY = 2.0

# Timeout da chamada HTTP. O SDK herda 600s se ninguém escolher, e 10 minutos aqui não é
# só espera: `_pendentes` segura as linhas do lote com FOR UPDATE durante a chamada, e a
# transação fica aberta junto. Num Postgres com `idle_in_transaction_session_timeout`
# ligado, é esse número que decide se a passada morre. 60s é folgado pra um lote de 200
# textos curtos e mantém a janela de trava em algo defensável.
EMBED_TIMEOUT = 60.0


@lru_cache(maxsize=1)
def get_client() -> voyageai.Client:
    """Cliente da Voyage, um por processo.

    A chave é lida **na chamada**, não no import — mesma lição do `DATABASE_URL` em
    `app/db.py`: qualquer leitura de env no nível do módulo faz `import app.rag.embedding`
    morrer num ambiente sem configuração, inclusive durante a *coleta* do pytest, antes
    de qualquer teste rodar (e nenhum teste daqui chama a API de verdade).

    Falha alto e cedo quando a variável não existe, em vez de deixar o SDK descobrir isso
    no meio da passada: aí já haveria lotes pagos e a mensagem seria um 401 genérico.

    O `lru_cache` é o que garante um cliente (e um pool de conexões HTTP) por processo.
    Quem troca a env em teste precisa chamar `get_client.cache_clear()`.
    """
    key = os.environ.get("VOYAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "VOYAGE_API_KEY não está no ambiente — a indexação de vetores precisa da "
            "chave da Voyage. Ponha o valor real no .env (ver .env.example)."
        )
    # `max_retries=0` (o padrão) é deliberado: quem retenta é `_embed_com_retry`, e só
    # rate limit. Deixar o SDK retentar por conta dele empilharia uma política invisível
    # por baixo da nossa — inclusive em erros que não devem ser retentados.
    return voyageai.Client(api_key=key, timeout=EMBED_TIMEOUT, max_retries=0)


def _embed_com_retry(
    client: voyageai.Client,
    texts: list[str],
    input_type: str,
    tentativas: int = RATE_LIMIT_TENTATIVAS,
    espera_inicial: float = RATE_LIMIT_ESPERA_INICIAL,
):
    """`client.embed` com backoff exponencial + jitter, **só** para `RateLimitError`.

    `input_type` é parâmetro (e não constante fixa) porque este é o único ponto que
    chama a API, e os dois lados do par assimétrico passam por aqui: a indexação com
    `"document"`, a busca com `"query"`. Ver o docstring do módulo.

    `tentativas`/`espera_inicial` também são parâmetros porque **o orçamento de espera
    depende de quem está esperando**: o backfill é offline e pode aguardar minutos; a
    busca roda dentro de um request e segura um worker do `to_thread` enquanto dorme.
    Os defaults são os do backfill; a busca passa os valores curtos.

    Rate limit é a única falha que se resolve sozinha esperando — é uma janela que
    reabre. Qualquer outra exceção sobe na hora, sem retry: chave inválida, dimensão
    negociada errada ou texto grande demais não melhoram com espera, e retentar um erro
    real só faz o backfill demorar 7 minutos pra dar a mesma mensagem.

    **`Timeout` fica de fora de propósito, mesmo parecendo transitório.** Um 429 é
    recusado *antes* de processar, então retentar não paga nada duas vezes. Um timeout,
    não: a requisição pode ter sido processada e cobrada do outro lado, e nós só não
    vimos a resposta — reenviar o lote é uma cobrança em dobro que nenhum lado consegue
    detectar depois, dentro do livro-caixa que este projeto existe pra manter confiável.
    Melhor a passada morrer e retomar (o commit por lote garante que nada pago se perde)
    do que gastar duas vezes em silêncio.

    O jitter (metade fixa, metade aleatória) desincroniza as tentativas: sem ele, dois
    processos que baterem no limite ao mesmo tempo voltam ao mesmo tempo, batem de novo,
    e continuam em fase até desistirem juntos.

    Esgotadas as tentativas, a exceção **sobe**. Isso é o desenho, não desistência: o
    commit por lote de `embed_pending` já preservou tudo que foi pago até ali, e a
    re-execução retoma sozinha pelas linhas que continuam com `embedding IS NULL`.
    """
    espera = espera_inicial
    for tentativa in range(1, tentativas + 1):
        try:
            return client.embed(
                texts,
                model=EMBED_MODEL,
                input_type=input_type,
                truncation=False,   # cortar em silêncio é pior que falhar — ver o módulo
            )
        except voyageai.error.RateLimitError as exc:
            if tentativa == tentativas:
                logger.warning(
                    "rate limit da Voyage na tentativa %d/%d — desistindo deste lote: %s",
                    tentativa, tentativas, exc,
                )
                raise
            pausa = espera / 2 + random.uniform(0, espera / 2)
            # WARNING, não DEBUG: um backfill de 20 minutos que fica mudo é
            # indistinguível de um travado, e a reação a cada um é oposta.
            logger.warning(
                "rate limit da Voyage (tentativa %d/%d) — esperando %.1fs antes de "
                "retentar o lote de %d texto(s)",
                tentativa, tentativas, pausa, len(texts),
            )
            time.sleep(pausa)
            espera = min(espera * RATE_LIMIT_FATOR, RATE_LIMIT_ESPERA_MAX)


def _validar(vetores: list[list[float]], texts: list[str]) -> None:
    """Correspondência 1:1 e dimensão certa — as duas checagens que o banco não faz bem.

    Usada pelos dois lados (indexação e busca) de propósito: a mensagem tem que falar do
    MODELO, não da coluna. Ver o porquê de cada uma no docstring de `embed_documents`.
    """
    if len(vetores) != len(texts):
        raise ValueError(
            f"{EMBED_MODEL} devolveu {len(vetores)} vetores para {len(texts)} textos — "
            "sem correspondência 1:1 os vetores seriam gravados no chunk errado."
        )
    for i, v in enumerate(vetores):
        if len(v) != EMBEDDING_DIM:
            raise ValueError(
                f"vetor {i} veio com {len(v)} dimensões, esperado {EMBEDDING_DIM} — "
                f"o modelo configurado ({EMBED_MODEL}) não bate com a coluna "
                "`clause_chunk.embedding`, que é vector(1024) e só muda por migration."
            )


def embed_documents(
    texts: list[str], client: voyageai.Client | None = None
) -> tuple[list[list[float]], int]:
    """Embedda um lote de textos de chunk. Devolve `(vetores, tokens_cobrados)`.

    Síncrona de propósito: o SDK da Voyage é síncrono, e quem precisa de async
    (`embed_pending`) chama isto por `asyncio.to_thread` — mesmo arranjo do
    `get_schema()`/`run_query()` do MCP server.

    A validação de dimensão acontece **aqui**, antes de qualquer escrita: um vetor com
    tamanho errado é rejeitado pelo Postgres de qualquer jeito (`expected 1024
    dimensions`), mas aí o erro chega depois do lote já estar pago e no meio de uma
    transação, com uma mensagem que fala de coluna em vez de falar de modelo. Também
    conferimos a *quantidade* de vetores: um lote que volta curto alinharia vetor com
    chunk errado no `zip` do chamador — o chunk receberia, em silêncio, o vetor do
    vizinho, e nada no banco denunciaria isso.
    """
    if not texts:
        return [], 0

    client = client or get_client()
    # A chamada em si (com input_type="document", OBRIGATÓRIO — a busca consulta com
    # "query" e é o par assimétrico que a Voyage otimiza) vive em `_embed_com_retry`,
    # que reenvia o lote enquanto o erro for rate limit. Ver o docstring do módulo.
    resp = _embed_com_retry(client, texts, INPUT_TYPE_DOCUMENT)

    vetores = list(resp.embeddings)
    _validar(vetores, texts)
    return vetores, resp.total_tokens


def embed_query_with_tokens(
    text: str, client: voyageai.Client | None = None
) -> tuple[list[float], int]:
    """`embed_query` + os tokens cobrados pela chamada. É o que a busca usa.

    Existe separada porque quem embedda uma pergunta dentro de um request precisa gravar
    o `cost_event` (a chamada é paga como qualquer outra), e o número que vai pro
    livro-caixa tem que ser o que a API COBROU — estimar tokens por contagem de
    caracteres gravaria custo errado com cara de certo, exatamente o que o `fail loud` do
    `app/cost.py` existe pra impedir. `embed_query` fica sendo a assinatura simples pra
    quem só quer o vetor.

    Mesma validação de `embed_documents` (dimensão e correspondência 1:1) e mesmo retry
    de rate limit — mas com **orçamento de espera próprio e curto**
    (`RATE_LIMIT_TENTATIVAS_QUERY`): quem espera aqui é um request, não um backfill. Ver
    a constante.
    """
    if not text or not text.strip():
        raise ValueError(
            "pergunta vazia — embeddar espaço em branco custa uma chamada e devolve um "
            "vetor de ruído que casaria com qualquer cláusula do corpus."
        )

    client = client or get_client()
    # `input_type="query"` é o OUTRO lado do par: a cláusula foi indexada como
    # "document", e é essa assimetria que põe a pergunta perto das cláusulas que a
    # respondem. Trocar por "document" aqui não levanta erro — só piora os vizinhos.
    resp = _embed_com_retry(
        client,
        [text],
        INPUT_TYPE_QUERY,
        tentativas=RATE_LIMIT_TENTATIVAS_QUERY,
        espera_inicial=RATE_LIMIT_ESPERA_QUERY,
    )

    vetores = list(resp.embeddings)
    _validar(vetores, [text])
    return vetores[0], resp.total_tokens


def embed_query(text: str, client: voyageai.Client | None = None) -> list[float]:
    """Embedda UMA pergunta, com `input_type="query"`. Devolve só o vetor.

    Síncrona pelo mesmo motivo de `embed_documents`: o SDK da Voyage é síncrono, e quem
    precisa de async chama por `asyncio.to_thread` (ver `app/rag/search.py`).
    """
    vetor, _ = embed_query_with_tokens(text, client)
    return vetor
