"""Worker RAG: do texto já extraído até a busca por similaridade.

Duas fatias prontas:

- **R2a — materialização.** `chunking` monta o texto de cada chunk (funções puras) e
  `index` grava as linhas em `clause_chunk`, com `embedding` `NULL`. Não gasta nada.
- **R2b — embedding.** `embedding` fala com a Voyage (`voyage-4-lite`,
  `input_type="document"`) e `embed` preenche as linhas pendentes em lotes, uma linha de
  `cost_event` por lote. **Esta parte custa dinheiro** e commita lote a lote.

Falta a **R3**: a busca por similaridade (consulta com `input_type="query"`) e o nó de
RAG no grafo de agentes.
"""
