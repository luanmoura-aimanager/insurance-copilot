"""Worker RAG: do texto já extraído até a busca por similaridade.

Fatia atual (R2a): **materialização** dos chunks — `chunking` monta o texto de cada
chunk e `index` grava as linhas em `clause_chunk`. Não há embedding aqui: as colunas
`embedding`/`embedding_model` nascem `NULL` e é a R2b que as preenche.
"""
