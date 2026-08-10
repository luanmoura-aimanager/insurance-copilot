"""Worker RAG: do texto já extraído até a busca por similaridade.

Três fatias prontas:

- **R2a — materialização.** `chunking` monta o texto de cada chunk (funções puras) e
  `index` grava as linhas em `clause_chunk`, com `embedding` `NULL`. Não gasta nada.
- **R2b — embedding.** `embedding` fala com a Voyage (`voyage-4-lite`,
  `input_type="document"`) e `embed` preenche as linhas pendentes em lotes, uma linha de
  `cost_event` por lote. **Esta parte custa dinheiro** e commita lote a lote.

- **R3a — busca.** `search` embedda a pergunta (`input_type="query"`, o outro lado do
  par assimétrico) e devolve os chunks mais próximos por distância de cosseno, com a
  origem de cada um. Gasta uma chamada por busca (`cost_event`, `agent_name="rag_search"`).

Falta a **R3b**: o nó de RAG no grafo de agentes — nada em `app/agents/` roteia pra cá
ainda —, e calibrar o limiar de relevância (`scripts/calibrate_search.py`).
"""
