#!/usr/bin/env python3
"""
Extração-piloto (manual) de 2 CGs residenciais, pra validar o schema v0 ANTES
de construir o pipeline LLM->Postgres. NÃO é o extrator de produção — os valores
de cobertura aqui foram lidos à mão do texto dos PDFs (Porto Seguro 482868,
Allianz 502775) pra responder as perguntas em aberto do schema:

  1. franquia_tipo cobre os casos reais?
  2. exclusões: tabela própria ou campo-texto?
  3. lista canônica de nome_cobertura (normalização entre seguradoras)?

A proveniência (pdf_url, pdf_hash, processo, versão) vem do corpus_manifest.json
— é o elo entre o corpus baixado e a extração. Quando o pipeline real existir,
ele preenche os mesmos campos via chamada de modelo; este artefato fixa o alvo.

Uso: python pilot_extraction.py   (gera ../data/pilot_extraction.json)
"""
import json
from datetime import datetime, timezone
from pathlib import Path

# Ancorado na raiz do repo (scripts/..), pra funcionar de qualquer CWD.
REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST = REPO_ROOT / "data" / "corpus" / "corpus_manifest.json"
OUT = REPO_ROOT / "data" / "pilot_extraction.json"

# Coberturas lidas do texto das CGs. franquia_tipo/base_calculo refletem o que a
# CG DIZ — o valor numérico em si é "definido na apólice" (não está na CG).
EXTRACAO = {
    482868: {
        "produto": "Porto Seguro Residência (Residencial Fácil)",
        "tipo_imovel": ["habitual", "veraneio"],  # CG cobre residência habitual e de veraneio
        "coberturas": [
            ("Incêndio, explosão, implosão, fumaça e queda de aeronaves", "basica",
             "misto", "POS por cobertura, valor ou percentual definido na apólice", "sobre prejuizo"),
            ("Danos elétricos", "adicional",
             "misto", "POS definida na apólice", "sobre prejuizo"),
            ("Perda ou pagamento de aluguel", "adicional",
             "sem_franquia", "cobertura indenitária por aluguel; sem POS típica", "sobre LMI"),
            ("Quebra de vidros", "adicional",
             "misto", "POS definida na apólice", "sobre prejuizo"),
            ("Responsabilidade civil familiar (danos a terceiros)", "adicional",
             "sem_franquia", "RC sem POS na CG", "sobre LMI"),
            ("Subtração de bens (roubo/furto qualificado)", "adicional",
             "misto", "POS definida na apólice", "sobre prejuizo"),
            ("Vendaval, furacão, ciclone, tornado e queda de granizo", "adicional",
             "misto", "POS definida na apólice", "sobre prejuizo"),
        ],
    },
    502775: {
        "produto": "Allianz Coletivo Patrimonial – Residência",
        "tipo_imovel": ["habitual"],
        "coberturas": [
            ("Incêndio, raio e explosão", "basica",
             "misto", "Franquia e/ou POS: valor ou percentual definido na apólice", "sobre prejuizo"),
            ("Danos elétricos", "adicional",
             "misto", "Franquia/POS definida na apólice", "sobre prejuizo"),
            ("Vendaval, granizo, fumaça e impacto de veículos", "adicional",
             "misto", "Franquia/POS definida na apólice", "sobre prejuizo"),
            ("Roubo e/ou furto qualificado de bens", "adicional",
             "misto", "Franquia/POS definida na apólice", "sobre prejuizo"),
            ("Responsabilidade civil familiar", "adicional",
             "sem_franquia", "RC sem franquia na CG", "sobre LMI"),
            ("Desmoronamento", "adicional",
             "misto", "Franquia/POS definida na apólice", "sobre prejuizo"),
            ("Alagamento / tumulto", "adicional",
             "misto", "Franquia/POS definida na apólice", "sobre prejuizo"),
        ],
    },
}


def main():
    if not MANIFEST.exists():
        raise SystemExit(
            f"manifesto ausente: {MANIFEST}\n"
            "Rode o harvester primeiro: python scripts/susep_harvest.py")
    manifest = {d["id"]: d for d in json.loads(MANIFEST.read_text())["documentos"]}
    faltando = [did for did in EXTRACAO if did not in manifest]
    if faltando:
        raise SystemExit(
            f"ids do piloto ausentes no manifesto: {faltando}\n"
            f"Garanta que o harvester baixou esses processos (sem --limit que os exclua).")
    docs, coverages = [], []
    for did, ext in EXTRACAO.items():
        m = manifest[did]
        docs.append({
            "id": did,
            "seguradora": m["seguradora"],
            "produto": ext["produto"],
            "susep_processo": m["processo"],
            "versao_vigencia": m["dt_inicio_comerc"],
            "tipo_imovel": ext["tipo_imovel"],
            "pdf_url": m["url"],
            "pdf_hash": m["sha256"],
            "extracted_at": datetime.now(timezone.utc).isoformat(),
        })
        for nome, tipo, fr_tipo, fr_txt, base in ext["coberturas"]:
            coverages.append({
                "document_id": did,
                "nome_cobertura": nome,
                "tipo": tipo,
                "franquia_tipo": fr_tipo,
                "franquia_regra_texto": fr_txt,
                "base_calculo": base,
                "exclusoes_principais": None,  # ver achado #2 abaixo
            })
    OUT.write_text(json.dumps(
        {"policy_document": docs, "coverage": coverages,
         "nota": "Piloto manual — valida schema v0; ver scripts/pilot_extraction.py"},
        ensure_ascii=False, indent=2))
    print(f"policy_document: {len(docs)} | coverage: {len(coverages)} -> {OUT}")


if __name__ == "__main__":
    main()
