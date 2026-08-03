"""A engine é preguiçosa: importar a app NÃO exige DATABASE_URL.

Esse bug voltou 4 vezes (sempre disfarçado de "ordem de fixture"): `app/db.py` lia
`os.environ["DATABASE_URL"]` no nível do módulo, então qualquer `import app.main` num
ambiente sem banco configurado morria com KeyError no import — inclusive na coleta do
pytest, antes do testcontainers ter subido o Postgres. O teste abaixo é a trava.

Roda em SUBPROCESSO de propósito: o processo do pytest já tem DATABASE_URL no env
(a fixture `db_url` aponta pro container), então dentro dele não dá pra provar nada.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

# Programa do filho. Ele roda com env limpo e fora do repo (ver `_run_child`), mas
# quem decide se o .env do dev local entra na jogada é o `find_dotenv`: hoje, num
# `python -c`, ele procura a partir do CWD (não acha nada) — só que essa heurística é
# detalhe de versão do python-dotenv, e no outro ramo dela a busca sobe a partir do
# diretório de `app/db.py` e acha o .env do repo. Então `load_dotenv` é neutralizado
# ANTES de qualquer import da app: o filho enxerga o mesmo mundo do CI (sem .env, sem
# DATABASE_URL) independente da versão. O assert final é o cinto — se DATABASE_URL
# entrar por outro caminho, o teste falha alto em vez de passar à toa.
_CHILD = """
import os
import dotenv
dotenv.load_dotenv = lambda *a, **k: False
import app.main
assert "DATABASE_URL" not in os.environ, "vazou DATABASE_URL pro subprocesso"
"""


def _run_child(code: str, tmp_path) -> subprocess.CompletedProcess:
    """Roda `code` num Python com env limpo (sem DATABASE_URL) e fora do repo."""
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": str(REPO_ROOT)}
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        cwd=tmp_path,          # fora do repo: nada de .env nem de imports por acaso
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_import_app_main_sem_database_url(tmp_path):
    """`import app.main` tem que funcionar sem banco nenhum configurado.

    É exatamente o que o CI faz: instala as deps e roda `pytest -q` sem setar
    DATABASE_URL — quem cria o Postgres é o testcontainers, já dentro da suíte.
    """
    proc = _run_child(_CHILD, tmp_path)

    assert proc.returncode == 0, (
        "importar app.main sem DATABASE_URL falhou — a leitura do env voltou pro "
        f"nível do módulo?\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


def test_sessionmaker_e_cacheado(db_url):
    """Uma engine por processo: o `lru_cache` é o que impede um pool novo por chamada.

    Pede `db_url` porque aqui a leitura do env acontece de verdade (é o ponto: ela
    acontece em RUNTIME, na primeira chamada, não no import).
    """
    from app.db import SessionLocal, _sessionmaker

    try:
        assert _sessionmaker() is _sessionmaker()
        # O nome público continua sendo um atributo CHAMÁVEL do módulo: 6 scripts
        # fazem `from app.db import SessionLocal` e dois testes trocam esse atributo
        # via monkeypatch.setattr("app.db.SessionLocal", ...).
        assert callable(SessionLocal)
    finally:
        # Não deixa uma engine ligada ao container pendurada no cache do módulo.
        _sessionmaker.cache_clear()


@pytest.mark.parametrize("modulo", ["app.db", "app.cost", "app.agents.graph"])
def test_modulos_da_cadeia_importam_limpos(modulo, tmp_path):
    """Os três degraus da cadeia de import de `app.main`, um a um.

    Se algum dia a leitura do env voltar, isso diz QUAL módulo a trouxe de volta em
    vez de só apontar pro topo da pilha.
    """
    proc = _run_child(_CHILD.replace("import app.main", f"import {modulo}"), tmp_path)

    assert proc.returncode == 0, f"{modulo} exigiu env no import\nstderr:\n{proc.stderr}"
