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
    # Só o mínimo pro Python achar o interpretador e o pacote. HOME entra porque
    # algumas libs expandem `~` no import e um env sem HOME é um jeito bobo de a
    # suíte ficar vermelha por um motivo que não é o do teste — ele não tem nada a
    # ver com a configuração do banco, que é o que precisa estar ausente aqui.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "PYTHONPATH": str(REPO_ROOT),
    }
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


async def test_sessionlocal_abre_session_de_verdade_e_reusa_a_engine(db_url):
    """Chama `SessionLocal()` PRA VALER e roda uma query com o que ela devolve.

    Não é firula: nenhum outro teste chama essa função. Os dois testes de custo
    substituem o atributo (`monkeypatch.setattr("app.db.SessionLocal", ...)`) e a
    fixture `client` sobrescreve a dependência `get_session` — então um bug DENTRO
    de `SessionLocal` (devolver a fábrica em vez da session, p.ex.) passaria pela
    suíte inteira verde e só apareceria no /health/db ou num script.

    Pede `db_url` porque aqui a leitura do env acontece de verdade — que é o ponto
    da fatia: ela acontece em RUNTIME, na primeira chamada, não no import.
    """
    from sqlalchemy import text

    from app.db import SessionLocal, _sessionmaker

    sm = _sessionmaker()
    engine = sm.kw["bind"]
    try:
        # Uma engine por processo: o lru_cache é o que impede um pool novo por chamada.
        assert sm is _sessionmaker()

        async with SessionLocal() as session:
            assert (await session.execute(text("SELECT 1"))).scalar_one() == 1
    finally:
        # Não deixa uma engine ligada ao container pendurada no cache do módulo.
        _sessionmaker.cache_clear()
        await engine.dispose()


@pytest.mark.parametrize("modulo", ["app.db", "app.cost", "app.agents.graph"])
def test_modulos_da_cadeia_importam_limpos(modulo, tmp_path):
    """Os três degraus da cadeia de import de `app.main`, um a um.

    Se algum dia a leitura do env voltar, isso diz QUAL módulo a trouxe de volta em
    vez de só apontar pro topo da pilha.
    """
    proc = _run_child(_CHILD.replace("import app.main", f"import {modulo}"), tmp_path)

    assert proc.returncode == 0, f"{modulo} exigiu env no import\nstderr:\n{proc.stderr}"
