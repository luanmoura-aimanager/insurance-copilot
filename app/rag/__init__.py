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

- **R3b — ligação com o grafo.** Quem chama `search` de dentro de um request é o nó
  `rag_worker` (`app/agents/graph.py`): ele abre a própria session, busca e aplica o
  limiar de relevância. Nada aqui depende de `app/agents/` — a direção é só essa.

Falta calibrar o limiar de relevância (`scripts/calibrate_search.py` existe; o número,
não): `MAX_DISTANCE_PADRAO` segue decidido e não medido.
"""
