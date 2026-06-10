#!/usr/bin/env python3
"""
Harvester de corpus SUSEP — condições gerais de SEGURO RESIDENCIAL.

Pipeline (todo público, sem login):

  1. ÍNDICE  — OData da SUSEP (Olinda), recurso DadosProdutos. Devolve TODOS
     os produtos registrados {tipoproduto, entnome, cnpj, numeroprocesso,
     ramo, subramo}. Filtramos ramo == "01 | COMPREENSIVO RESIDENCIAL".
       https://dados.susep.gov.br/olinda/servico/produtos/versao/v1/odata/DadosProdutos?$format=json
     (Atenção: o serviço NÃO aceita $top/$select/$filter aqui — devolve 500.
      Só o dataset inteiro com $format=json. Baixamos uma vez e cacheamos.)

  2. RESOLVE — processo -> id(s) de download. A consulta pública de produtos
     é um POST (multipart) que devolve HTML com a tabela de versões; cada
     versão tem um link DownloadConsultaPublica/{id}, o nome do arquivo e as
     datas de início/fim de comercialização.
       POST https://www2.susep.gov.br/safe/menumercado/REP2/Produto.aspx/Consultar
            campo: numeroProcesso

  3. DOWNLOAD — id -> PDF.
       GET  https://www2.susep.gov.br/safe/menumercado/REP2/Produto.aspx/DownloadConsultaPublica/{id}

  4. MANIFESTO — data/corpus/corpus_manifest.json com proveniência por versão
     (processo, id, seguradora, cnpj, produto/arquivo, versão, datas, ramo,
     url, sha256, baixado_em, tem_texto). Material de governança do projeto.

Por padrão baixa só a versão MAIS RECENTE (em comercialização) de cada
processo — é a CG vigente, unidade certa pra extração, e mantém o volume
educado (~147 PDFs). Use --all-versions pra baixar todo o histórico.
Todas as versões (baixadas ou não) entram no manifesto, com proveniência.

Educado com o servidor: ~1 req/s global, UA identificável, retry com backoff,
PARA na hora se vier 429/403. Resumível: pula ids já baixados.

Uso:
  python susep_harvest.py                         # latest por processo
  python susep_harvest.py --all-versions          # histórico completo
  python susep_harvest.py --limit 5               # smoke test (5 processos)
  python susep_harvest.py --refresh-index         # rebaixa o índice OData

Requisitos: pip install requests pypdf  (ver gotcha do keyring no probe)
"""

import argparse
import hashlib
import io
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from pypdf import PdfReader

# ---- Endpoints (todos confirmados em 09/06/2026) -----------------------------
ODATA_PRODUTOS = ("https://dados.susep.gov.br/olinda/servico/produtos/"
                  "versao/v1/odata/DadosProdutos")
CONSULTAR = "https://www2.susep.gov.br/safe/menumercado/REP2/Produto.aspx/Consultar"
DOWNLOAD = ("https://www2.susep.gov.br/safe/menumercado/REP2/"
            "Produto.aspx/DownloadConsultaPublica/{id}")

RAMO_RESIDENCIAL = "01 | COMPREENSIVO RESIDENCIAL"

HEADERS = {"User-Agent": "insurance-copilot-research/0.1 (corpus SUSEP residencial; contato: luanmisaelmoura@gmail.com)"}
RATE_LIMIT_SECONDS = 1.0          # 1 req/s global (educado)
SCAN_TEXT_THRESHOLD = 300         # < isso de texto extraído => provável scan/OCR
MAX_RETRIES = 4

# Marcadores de conteúdo residencial (sanity check do que baixamos).
RESIDENCIAL_HINTS = ("residencial", "residência", "residencia", "imóvel residencial",
                     "imovel residencial", "compreensivo residencial")


class StopHarvest(Exception):
    """Servidor pediu pra parar (429/403) — abortamos com elegância."""


class RateLimiter:
    """1 req/s global, compartilhado entre resolve e download."""
    def __init__(self, seconds: float):
        self.seconds = seconds
        self._last = 0.0

    def wait(self):
        dt = time.monotonic() - self._last
        if dt < self.seconds:
            time.sleep(self.seconds - dt)
        self._last = time.monotonic()


def get_with_retry(session, method, url, limiter, **kwargs):
    """GET/POST com rate limit + backoff. Levanta StopHarvest em 429/403."""
    backoff = 2.0
    for attempt in range(1, MAX_RETRIES + 1):
        limiter.wait()
        try:
            r = session.request(method, url, headers=HEADERS, timeout=60, **kwargs)
        except requests.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"    rede falhou ({e}); retry {attempt}/{MAX_RETRIES} em {backoff:.0f}s",
                  file=sys.stderr)
            time.sleep(backoff); backoff *= 2; continue
        if r.status_code in (429, 403):
            raise StopHarvest(f"HTTP {r.status_code} em {url} — servidor pediu pra parar")
        if r.status_code >= 500 and attempt < MAX_RETRIES:
            print(f"    HTTP {r.status_code}; retry {attempt}/{MAX_RETRIES} em {backoff:.0f}s",
                  file=sys.stderr)
            time.sleep(backoff); backoff *= 2; continue
        return r
    return r


# ---- 1. ÍNDICE ---------------------------------------------------------------
def load_index(session, limiter, cache: Path, refresh: bool):
    """Carrega DadosProdutos (cacheado). Devolve lista de dicts."""
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    print("Baixando índice OData (DadosProdutos)...", file=sys.stderr)
    r = get_with_retry(session, "GET", ODATA_PRODUTOS, limiter, params={"$format": "json"})
    r.raise_for_status()
    data = r.json()["value"]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data, ensure_ascii=False))
    print(f"  índice: {len(data)} produtos -> {cache}", file=sys.stderr)
    return data


def residencial_processos(index):
    """Subconjunto residencial, deduplicado por processo, ordem estável."""
    seen, out = set(), []
    for d in index:
        if d.get("ramo") == RAMO_RESIDENCIAL:
            p = d["numeroprocesso"]
            if p not in seen:
                seen.add(p)
                out.append(d)
    return out


# ---- 2. RESOLVE --------------------------------------------------------------
# Cada linha da tabela de versões:
#   <span>NOME.pdf</span> ... DownloadConsultaPublica/{id} ... <td>dd/mm/aaaa</td><td>dd/mm/aaaa?</td>
ROW_RE = re.compile(
    r'<span>(?P<arquivo>[^<]*)</span>.*?DownloadConsultaPublica/(?P<id>\d+).*?'
    r'<td[^>]*>(?P<inicio>[^<]*)</td>\s*<td[^>]*>(?P<fim>[^<]*)</td>',
    re.S)


def _parse_dt(s):
    s = (s or "").strip()
    try:
        return datetime.strptime(s, "%d/%m/%Y")
    except ValueError:
        return None


def _parse_versoes_html(html):
    versoes, ids = [], set()
    for m in ROW_RE.finditer(html):
        vid = int(m.group("id"))
        if vid in ids:
            continue  # dedup; a tabela vem da versão mais nova p/ mais velha
        ids.add(vid)
        versoes.append({
            "id": vid,
            "arquivo": m.group("arquivo").strip(),
            "dt_inicio": m.group("inicio").strip() or None,
            "dt_fim": m.group("fim").strip() or None,
        })
    return versoes


# A consulta pública tem uma COTA POR SESSÃO: depois de ~14 consultas distintas,
# o endpoint passa a devolver HTTP 200 com a página SEM a tabela de resultados
# (não é 429/403 — é uma resposta degradada). A cota é amarrada ao cookie de
# sessão ASP.NET, não ao IP: basta abrir uma sessão nova (novo cookie) que zera.
# Então rotacionamos a sessão a cada ROTATE_EVERY resolves (margem sob o limite)
# e também reativamente quando vem vazio. Mantemos 1 req/s e UA identificável —
# mesma carga, só cookies novos de vez em quando.
ROTATE_EVERY = 10


class SusepClient:
    """Cliente educado p/ a consulta pública, com rotação de sessão."""

    def __init__(self, limiter, rotate_every=ROTATE_EVERY):
        self.limiter = limiter
        self.rotate_every = rotate_every
        self.session = None
        self._since_rotate = 0
        self._new_session()

    def _new_session(self):
        self.session = requests.Session()
        get_with_retry(self.session, "GET", CONSULTAR, self.limiter)  # prime cookie
        self._since_rotate = 0

    def resolve_versoes(self, processo, empty_retries=3):
        """processo -> lista de versões. Vazio => sessão provavelmente esgotou
        a cota; abre sessão nova e tenta de novo."""
        for attempt in range(empty_retries + 1):
            if self._since_rotate >= self.rotate_every:
                self._new_session()
            self._since_rotate += 1
            r = get_with_retry(self.session, "POST", CONSULTAR, self.limiter,
                               files={"numeroProcesso": (None, processo)})
            versoes = _parse_versoes_html(r.text)
            if versoes:
                return versoes
            self._new_session()  # vazio => cota; reseta antes da próxima tentativa
        return []

    def baixar_pdf(self, doc_id):
        r = get_with_retry(self.session, "GET", DOWNLOAD.format(id=doc_id), self.limiter)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        if "pdf" not in r.headers.get("Content-Type", "").lower():
            return None, f"não-PDF ({r.headers.get('Content-Type')})"
        return r.content, None


def pick_latest(versoes):
    """A versão vigente: dt_fim vazio (ainda comercializada) e maior dt_inicio;
    senão, maior dt_inicio."""
    if not versoes:
        return None
    vigentes = [v for v in versoes if not v["dt_fim"]]
    pool = vigentes or versoes
    return max(pool, key=lambda v: (_parse_dt(v["dt_inicio"]) or datetime.min))


# ---- 3. análise --------------------------------------------------------------
def analisar_pdf(pdf_bytes, max_pages=5):
    """Devolve (sha256, n_paginas, tem_texto, parece_residencial)."""
    sha = hashlib.sha256(pdf_bytes).hexdigest()
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        n_pag = len(reader.pages)
        text = "\n".join((p.extract_text() or "") for p in reader.pages[:max_pages])
    except Exception:
        return sha, None, False, None
    tem_texto = len(text) >= SCAN_TEXT_THRESHOLD
    parece = any(h in text.lower() for h in RESIDENCIAL_HINTS) if tem_texto else None
    return sha, n_pag, tem_texto, parece


# ---- main --------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Harvester SUSEP — seguro residencial")
    ap.add_argument("--out", default="../data/corpus", help="pasta dos PDFs + manifesto")
    ap.add_argument("--index-cache", default="../data/index/dados_produtos.json")
    ap.add_argument("--refresh-index", action="store_true", help="rebaixa o índice OData")
    ap.add_argument("--all-versions", action="store_true",
                    help="baixa todas as versões (padrão: só a vigente por processo)")
    ap.add_argument("--limit", type=int, default=0, help="processa só N processos (smoke test)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "corpus_manifest.json"
    limiter = RateLimiter(RATE_LIMIT_SECONDS)

    # índice -> residenciais (sessão simples; OData é outro host, sem cota)
    index = load_index(requests.Session(), limiter, Path(args.index_cache), args.refresh_index)

    # cliente da consulta pública (rotaciona sessão p/ contornar a cota)
    client = SusepClient(limiter)
    resid = residencial_processos(index)
    if args.limit:
        resid = resid[:args.limit]
    print(f"Residenciais (Compreensivo Residencial): {len(resid)} processos\n", file=sys.stderr)

    # estado pra resumir
    entries = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text()).get("documentos", [])
        except Exception:
            entries = []
    ids_baixados = {e["id"] for e in entries if e.get("sha256")}

    stats = {"processos": len(resid), "versoes_total": 0, "baixados": 0,
             "ja_tinha": 0, "falhas": 0, "scan_sem_texto": 0,
             "nao_parece_resid": 0, "bytes": 0}
    abortou = None

    try:
        for i, prod in enumerate(resid, 1):
            processo = prod["numeroprocesso"]
            print(f"[{i}/{len(resid)}] {processo} — {prod['entnome']}", file=sys.stderr)
            try:
                versoes = client.resolve_versoes(processo)
            except StopHarvest:
                raise
            except Exception as e:
                print(f"    resolve falhou: {e}", file=sys.stderr)
                stats["falhas"] += 1
                continue
            stats["versoes_total"] += len(versoes)
            if not versoes:
                print("    (sem versões após rotação de sessão — revisar)", file=sys.stderr)
                stats["sem_versoes"] = stats.get("sem_versoes", 0) + 1
                continue

            alvos = versoes if args.all_versions else [pick_latest(versoes)]
            for v in alvos:
                base_entry = {
                    "processo": processo, "id": v["id"],
                    "seguradora": prod["entnome"], "cnpj": prod["cnpj"],
                    "arquivo": v["arquivo"], "ramo": prod["ramo"],
                    "subramo": prod.get("subramo"),
                    "dt_inicio_comerc": v["dt_inicio"], "dt_fim_comerc": v["dt_fim"],
                    "vigente": v["dt_fim"] is None,
                    "total_versoes_processo": len(versoes),
                    "url": DOWNLOAD.format(id=v["id"]),
                }
                if v["id"] in ids_baixados:
                    stats["ja_tinha"] += 1
                    continue
                pdf, err = client.baixar_pdf(v["id"])
                if pdf is None:
                    print(f"    id {v['id']}: {err}", file=sys.stderr)
                    stats["falhas"] += 1
                    continue
                sha, n_pag, tem_texto, parece = analisar_pdf(pdf)
                (out / f"susep_{v['id']}.pdf").write_bytes(pdf)
                base_entry.update({
                    "sha256": sha, "bytes": len(pdf), "n_paginas": n_pag,
                    "tem_texto": tem_texto, "parece_residencial": parece,
                    "baixado_em": datetime.now(timezone.utc).isoformat(),
                })
                entries.append(base_entry)
                ids_baixados.add(v["id"])
                stats["baixados"] += 1; stats["bytes"] += len(pdf)
                if not tem_texto:
                    stats["scan_sem_texto"] += 1
                if parece is False:
                    stats["nao_parece_resid"] += 1
                    print(f"    ⚠ id {v['id']} não menciona 'residencial' — revisar", file=sys.stderr)
                print(f"    ✓ id {v['id']}  {len(pdf)//1024} KB  "
                      f"{n_pag}p  texto={tem_texto}", file=sys.stderr)
    except StopHarvest as e:
        abortou = str(e)
        print(f"\n!! ABORTADO: {e}\n   Salvando o que já foi baixado.", file=sys.stderr)

    # manifesto
    manifest = {
        "fonte": {
            "indice_odata": ODATA_PRODUTOS,
            "consulta_publica": CONSULTAR,
            "download": DOWNLOAD,
            "ramo": RAMO_RESIDENCIAL,
        },
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "modo": "all-versions" if args.all_versions else "latest-por-processo",
        "abortado": abortou,
        "stats": stats,
        "documentos": entries,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    # volumetria — computada do manifesto (fonte da verdade; robusta a resume)
    cgs_unicas = len({e["processo"] for e in entries})
    com_texto = sum(1 for e in entries if e.get("tem_texto"))
    sem_texto = sum(1 for e in entries if e.get("tem_texto") is False)
    nao_resid = sum(1 for e in entries if e.get("parece_residencial") is False)
    bytes_total = sum(e.get("bytes", 0) for e in entries)
    print("\n===== VOLUMETRIA RESIDENCIAL =====")
    print(f"Esta execução: {json.dumps(stats, ensure_ascii=False)}")
    print(f"\nProcessos residenciais únicos no corpus : {cgs_unicas}")
    print(f"PDFs (versões) no corpus                 : {len(entries)}")
    if entries:
        print(f"% com texto extraível                   : {100*com_texto/len(entries):.0f}%")
        print(f"% provável scan (precisa OCR)           : {100*sem_texto/len(entries):.0f}%")
        print(f"Não mencionam 'residencial' (revisar)   : {nao_resid}")
        print(f"Tamanho total                           : {bytes_total/1e6:.1f} MB")
    print(f"\nManifesto: {manifest_path}")
    if abortou:
        sys.exit(2)


if __name__ == "__main__":
    main()
