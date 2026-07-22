"""Teste standalone (não-pytest) do postgres-mcp-server.

Carrega o módulo (nome com hífen → importlib), chama as tools direto e imprime
tudo em stdout. Sem supervisor, sem graph — só valida o servidor isolado.

Uso:
    source .venv/bin/activate
    python mcp_servers/test_postgres_mcp.py
"""

import importlib.util
import sys
from pathlib import Path

# Raiz do repo no path só para `import app.db_url` (rodado como script, o cwd não
# entra no sys.path — só o dir do script). Pós-rename mcp/ → mcp_servers/, isso já
# não sombreia o pacote `mcp` (FastMCP), então um insert(0) normal é seguro.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_SERVER_PATH = REPO_ROOT / "mcp_servers" / "postgres-mcp-server.py"
_spec = importlib.util.spec_from_file_location("postgres_mcp_server", _SERVER_PATH)
server = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(server)


def _tool(name):
    """Devolve a função da tool, desembrulhando o wrapper do FastMCP se houver."""
    fn = getattr(server, name)
    return getattr(fn, "fn", fn)


get_schema = _tool("get_schema")
run_query = _tool("run_query")


def main():
    print("=" * 70)
    print("1. get_schema()")
    print("=" * 70)
    print(get_schema())

    print("=" * 70)
    print("2. run_query — SELECT válido: SELECT name FROM peril LIMIT 5")
    print("=" * 70)
    print(run_query("SELECT name FROM peril LIMIT 5"))
    print()

    print("=" * 70)
    print("3. run_query — bloqueada: DROP TABLE coverage")
    print("=" * 70)
    print(run_query("DROP TABLE coverage"))
    print()

    print("=" * 70)
    print("4. run_query — empilhada: SELECT 1; DROP TABLE coverage")
    print("=" * 70)
    print(run_query("SELECT 1; DROP TABLE coverage"))
    print()


if __name__ == "__main__":
    main()
