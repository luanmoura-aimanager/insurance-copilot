"""Prova que a role `insurance_ro` é fisicamente read-only.

Conecta COMO insurance_ro (mesmo host/porta/banco do DATABASE_URL, mas
user=insurance_ro e password=$RO_ROLE_PASSWORD) e checa três coisas:
  - SELECT em `peril` devolve linhas          → leitura funciona
  - INSERT em `peril` levanta permission denied → escrita barrada
  - CREATE TABLE levanta permission denied      → DDL barrado

Uso:
    export RO_ROLE_PASSWORD="dev_ro_pw_local"
    python mcp_servers/test_readonly_role.py
"""

import os
import sys
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv
import psycopg

load_dotenv()

RO_USER = "insurance_ro"


def _ro_conninfo() -> str:
    """Reaproveita host/porta/banco do DATABASE_URL, troca user+senha pela role RO."""
    raw = os.environ["DATABASE_URL"]
    parts = urlsplit(raw)

    # Scheme puro pro libpq (tira qualquer +asyncpg / +psycopg).
    scheme = parts.scheme.split("+", 1)[0]
    if scheme == "postgres":
        scheme = "postgresql"

    password = os.environ.get("RO_ROLE_PASSWORD", "changeme_dev_only")
    host = parts.hostname or "localhost"
    port = f":{parts.port}" if parts.port else ""
    netloc = f"{RO_USER}:{password}@{host}{port}"

    return urlunsplit((scheme, netloc, parts.path, "", ""))


def _report(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    line = f"[{status}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok


def main() -> int:
    conninfo = _ro_conninfo()
    # Mascara a senha ao imprimir.
    print(f"Connecting as role: {RO_USER}\n")

    conn = psycopg.connect(conninfo)
    conn.autocommit = True  # cada statement isolado; erro não trava os seguintes
    results = []

    # 1. Leitura funciona.
    try:
        cur = conn.cursor()
        cur.execute("SELECT name FROM peril LIMIT 5")
        rows = cur.fetchall()
        results.append(
            _report("SELECT peril returns rows", bool(rows), f"{len(rows)} rows")
        )
    except psycopg.Error as exc:
        results.append(_report("SELECT peril returns rows", False, str(exc).strip()))

    # 2. INSERT barrado.
    try:
        conn.cursor().execute("INSERT INTO peril (name) VALUES ('__ro_probe__')")
        results.append(_report("INSERT into peril blocked", False, "INSERT succeeded!"))
    except psycopg.errors.InsufficientPrivilege as exc:
        results.append(
            _report("INSERT into peril blocked", True, str(exc).strip().splitlines()[0])
        )
    except psycopg.Error as exc:
        results.append(
            _report("INSERT into peril blocked", False, f"unexpected: {exc}".strip())
        )

    # 3. DDL (CREATE TABLE) barrado.
    try:
        conn.cursor().execute("CREATE TABLE __ro_probe__ (id int)")
        results.append(_report("CREATE TABLE blocked", False, "CREATE succeeded!"))
    except psycopg.errors.InsufficientPrivilege as exc:
        results.append(
            _report("CREATE TABLE blocked", True, str(exc).strip().splitlines()[0])
        )
    except psycopg.Error as exc:
        results.append(
            _report("CREATE TABLE blocked", False, f"unexpected: {exc}".strip())
        )

    conn.close()

    print()
    all_ok = all(results)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
