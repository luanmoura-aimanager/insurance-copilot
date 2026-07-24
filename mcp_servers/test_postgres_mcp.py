"""Teste standalone (não-pytest) do postgres_mcp_server.

Importa as funções core direto (pós-rename o módulo tem nome importável) e chama-as
sem o protocolo MCP, imprimindo tudo em stdout. Sem supervisor, sem graph — só valida
o servidor isolado.

Uso:
    source .venv/bin/activate
    PYTHONPATH=. python mcp_servers/test_postgres_mcp.py
"""

import sys
from pathlib import Path

# Raiz do repo no path só para `from mcp_servers... import` e `import app.db_url`
# (rodado como script, o cwd não entra no sys.path — só o dir do script). Pós-rename
# mcp/ → mcp_servers/, isso já não sombreia o pacote `mcp` (FastMCP), então um
# insert(0) normal é seguro.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mcp_servers.postgres_mcp_server import get_schema, run_query


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
