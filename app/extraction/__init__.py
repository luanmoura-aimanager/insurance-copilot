"""Pipeline de extraction: PDF de CG → objeto estruturado (schema v1).

O shape aqui (árvore aninhada document→coverages→perils) é o CONTRATO com o LLM,
não o shape de armazenamento. A normalização nas 5 tabelas (app/models.py) acontece
na camada de persistência, não aqui.
"""
