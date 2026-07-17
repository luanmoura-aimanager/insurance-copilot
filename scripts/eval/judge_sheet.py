"""Harness de eval F4: ficha por doc comparando coberturas extraídas vs candidatas no texto-fonte.

Custo ZERO. Heurística de candidato = linhas curtas em CAIXA-ALTA (cabeçalhos de
condição especial / garantia) e linhas de enumeração 'Cobertura de X' / 'Garantia de X'.
NÃO é verdade-base: é um radar pro juiz (humano/Claude) ler o trecho e decidir.
Uso: python scripts/eval/judge_sheet.py <out_dir> <slug>
"""

import json
import re
import sys
from pathlib import Path

D = Path(sys.argv[1])
slug = sys.argv[2]

covs = json.loads((D / f"{slug}.covs.json").read_text())
text = (D / f"{slug}.txt").read_text()
lines = text.splitlines()

print(f"===== {slug} | {covs['insurer']} | {covs['susep_process']} =====")
print(f"EXTRAÍDO ({covs['n_coverages']}):")
for c in covs["coverages"]:
    pl = f" [plano={c['plan']}]" if c.get("plan") else ""
    print(f"   • {c['coverage_name']}{pl}  ⟶ {c['kind']} perils={c['perils']}")

# ruído de glossário/exclusão pra filtrar
NOISE = re.compile(r"gloss|defini|exclu|não est|não ser|carência|import[âa]ncia segurada|"
                   r"limite m[áa]ximo|capital segurado|franquia:|participação obrig", re.I)
COVWORD = re.compile(r"vendaval|granizo|alagam|inunda|desmoron|deslizam|inc[êe]nd|explos|"
                     r"danos el[ée]|el[ée]tric|curto|raio|roubo|furto|subtra|quebra|vidro|"
                     r"impacto|ve[íi]culo|aluguel|responsabilidade civil|\bRC\b|tumulto|greve|"
                     r"vazam|[áa]gua|aeronave|fuma[çc]|terremoto|desmorona|"
                     r"assist|equipament|jardi|bicic?|port[õo]|obras de arte|pet\b|animal|"
                     r"despesas|honor|escombro|recomposi|c[ée]u aberto|dano el", re.I)

# Candidatos: CAIXA-ALTA de tamanho de cabeçalho
caps = []
for i, ln in enumerate(lines):
    s = ln.strip()
    if 8 <= len(s) <= 90 and re.match(r"^[A-ZÀ-Ú0-9 ,.\-/º°ª()–&]+$", s):
        if COVWORD.search(s) and not NOISE.search(s):
            caps.append(s)
# uniq preservando ordem
seen = set(); caps_u = [x for x in caps if not (x in seen or seen.add(x))]
print(f"\nCANDIDATOS CAIXA-ALTA c/ termo de cobertura ({len(caps_u)}):")
for s in caps_u:
    print("   |", s)

# Enumerações 'Cobertura(s) de X' / 'Garantia(s) de X'
enum = []
for ln in lines:
    for m in re.finditer(r"(?:coberturas?|garantias?)\s+(?:de\s+|adicional\s+de\s+|b[áa]sica\s+de\s+)?([A-ZÀ-Ú][^.;:\n]{4,60})", ln):
        frag = m.group(0).strip()
        if COVWORD.search(frag) and not NOISE.search(frag):
            enum.append(frag)
seen = set(); enum_u = [x for x in enum if not (x in seen or seen.add(x))]
print(f"\nENUM 'cobertura/garantia de …' ({len(enum_u)}):")
for s in enum_u[:60]:
    print("   »", s)
