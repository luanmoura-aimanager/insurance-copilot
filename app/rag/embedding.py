"""O único ponto que fala com a API de embedding (Voyage). Sem banco, sem ORM.

**Modelo: `voyage-4-lite`, 1024 dimensões.** A escolha já está gravada no schema — a
dimensão vive no *tipo* da coluna (`vector(1024)`), porque o Postgres precisa dela pra
indexar —, então trocar de modelo é migration, não config. O critério foi: multilíngue
(o corpus é pt-BR), barato, e dimensão pequena o bastante pra deixar o índice HNSW leve.

**`input_type` é o detalhe que faz a busca funcionar.** A Voyage treina o modelo de
forma *assimétrica*: documento e pergunta são embeddados com prefixos diferentes, e o
espaço é otimizado pra que uma *pergunta* fique perto dos *documentos* que a respondem
— não perto de outras perguntas. Aqui indexamos cláusulas, então é sempre
`input_type="document"`. A R3 (retrieval) vai consultar com `input_type="query"`, e é o
par que precisa bater: usar o mesmo dos dois lados degrada a recuperação em silêncio,
sem erro nenhum pra denunciar.

**Rate limit é a única falha retentada aqui** (`_embed_com_retry`): é a única que se
resolve sozinha esperando, porque é uma janela que reabre. Todo o resto sobe na hora —
mascarar um erro real atrás de 6 tentativas só faz o backfill levar minutos pra dar a
mesma mensagem.
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

# Prefixo do texto enviado, exigido pelo modelo assimétrico — ver o docstring do módulo.
INPUT_TYPE_DOCUMENT = "document"

# Backoff do rate limit. Os números são grandes de propósito: o limite da Voyage é por
# minuto, então esperar 1s não resolve nada — só queima tentativa. 6 tentativas com base
# 20s dobrando até o teto de 120s cobrem ~7 minutos de janela fechada, que é mais do que
# qualquer reset por minuto precisa.
RATE_LIMIT_TENTATIVAS = 6
RATE_LIMIT_ESPERA_INICIAL = 20.0
RATE_LIMIT_FATOR = 2.0
RATE_LIMIT_ESPERA_MAX = 120.0


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
    return voyageai.Client(api_key=key)


def _embed_com_retry(client: voyageai.Client, texts: list[str]):
    """`client.embed` com backoff exponencial + jitter, **só** para `RateLimitError`.

    Rate limit é a única falha que se resolve sozinha esperando — é uma janela que
    reabre. Qualquer outra exceção sobe na hora, sem retry: chave inválida, dimensão
    negociada errada, texto grande demais ou queda da API não melhoram com espera, e
    retentar um erro real só faz o backfill demorar 7 minutos pra dar a mesma mensagem.

    O jitter (metade fixa, metade aleatória) desincroniza as tentativas: sem ele, dois
    processos que baterem no limite ao mesmo tempo voltam ao mesmo tempo, batem de novo,
    e continuam em fase até desistirem juntos.

    Esgotadas as tentativas, a exceção **sobe**. Isso é o desenho, não desistência: o
    commit por lote de `embed_pending` já preservou tudo que foi pago até ali, e a
    re-execução retoma sozinha pelas linhas que continuam com `embedding IS NULL`.
    """
    espera = RATE_LIMIT_ESPERA_INICIAL
    for tentativa in range(1, RATE_LIMIT_TENTATIVAS + 1):
        try:
            return client.embed(texts, model=EMBED_MODEL, input_type=INPUT_TYPE_DOCUMENT)
        except voyageai.error.RateLimitError as exc:
            if tentativa == RATE_LIMIT_TENTATIVAS:
                logger.warning(
                    "rate limit da Voyage na tentativa %d/%d — desistindo deste lote: %s",
                    tentativa, RATE_LIMIT_TENTATIVAS, exc,
                )
                raise
            pausa = espera / 2 + random.uniform(0, espera / 2)
            # WARNING, não DEBUG: um backfill de 20 minutos que fica mudo é
            # indistinguível de um travado, e a reação a cada um é oposta.
            logger.warning(
                "rate limit da Voyage (tentativa %d/%d) — esperando %.1fs antes de "
                "retentar o lote de %d texto(s)",
                tentativa, RATE_LIMIT_TENTATIVAS, pausa, len(texts),
            )
            time.sleep(pausa)
            espera = min(espera * RATE_LIMIT_FATOR, RATE_LIMIT_ESPERA_MAX)


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
    # A chamada em si (com input_type="document", OBRIGATÓRIO — a R3 consulta com
    # "query" e é o par assimétrico que a Voyage otimiza) vive em `_embed_com_retry`,
    # que reenvia o lote enquanto o erro for rate limit. Ver o docstring do módulo.
    resp = _embed_com_retry(client, texts)

    vetores = list(resp.embeddings)
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
    return vetores, resp.total_tokens
