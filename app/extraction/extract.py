"""Chamada LLM com structured output (Anthropic tool-use).

Padrão canônico de structured output: expõe um único "tool" cujo input_schema é o
JSON Schema do Pydantic e força tool_choice pra esse tool. O modelo é obrigado a
devolver argumentos que batem com o schema (incl. os enums de perigo) → parseamos
o bloco tool_use direto no Pydantic. Sem regex, sem JSON solto no meio do texto.
"""

import os
import sys

import anthropic

from .schema import ExtractedDocument

# Sonnet é o default pra tarefa estruturada (Opus fica pra system design/ADR).
MODEL = os.getenv("EXTRACTION_MODEL", "claude-sonnet-5")
# CG de 122p com exclusões verbatim gera output grande; Sonnet suporta bem mais que 16k.
MAX_TOKENS = int(os.getenv("EXTRACTION_MAX_TOKENS", "32000"))

_TOOL_NAME = "record_extraction"

SYSTEM_PROMPT = """\
Você extrai dados estruturados de Condições Gerais (CG) de seguro residencial \
brasileiro registradas na SUSEP. A CG descreve um PRODUTO, não a apólice de um cliente.

Regras de extração:
- GRÃO: uma entrada por COBERTURA (não uma por documento). Liste todas as coberturas, \
básicas e adicionais.
- PLANOS: se a CG separa as coberturas em planos/tiers comerciais (ex.: "Essencial", "Fácil"), \
ponha o nome do plano no campo `plan` e mantenha `coverage_name` LIMPO, sem o sufixo do plano. \
`plan` deve conter APENAS o(s) nome(s) do plano — sem descrições, parênteses ou notas \
(ex.: "Essencial", "Fácil", "Essencial/Fácil"; NUNCA "Essencial (dentro de Multiproteção)"). \
Se a MESMA cobertura (mesmos perigos, kind, POS e exclusões) aparece idêntica em mais de um \
plano, NÃO duplique: gere UMA entrada com os planos combinados por barra (ex.: plan="Essencial/Fácil"). \
Só gere entradas separadas por plano quando a cobertura DIFERE entre eles. Se não há planos, \
`plan` = null.
- POS ≡ franquia. "Participação Obrigatória do Segurado" é sinônimo de franquia. O padrão \
dominante é "valor ou percentual definido na apólice" → deductible_type = defined_in_policy \
(a CG fixa a estrutura, o número vive na apólice). Use percentage/fixed_amount só quando a \
própria CG traz um número concreto. Use none quando a cobertura não tem POS.
- deductible_rule_text: copie o texto da regra de POS VERBATIM, sem parafrasear.
- PERIGOS: mapeie cada cobertura para os perigos canônicos da lista fixa (enum `perils`). \
Uma cobertura comercial pode cobrir vários perigos. Se um perigo da CG não existe na lista, \
não invente — omita.
- EXCLUSÕES: separe exclusões GERAIS (valem pro documento todo → general_exclusions) das \
exclusões específicas de uma cobertura (→ exclusions daquela coverage). Copie verbatim.
- Não invente dados. Campos ausentes na CG ficam null/vazios.
"""


def extract_document(text: str) -> ExtractedDocument:
    """Roda a extração sobre o texto de UMA CG e devolve o objeto validado."""
    client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente

    tool = {
        "name": _TOOL_NAME,
        "description": "Registra os dados estruturados extraídos de uma CG de seguro residencial.",
        "input_schema": ExtractedDocument.model_json_schema(),
    }

    # Streaming: com max_tokens alto a SDK exige stream (geração pode passar de 10 min).
    # Não usamos os eventos token-a-token — só acumulamos e pegamos a mensagem final.
    with client.messages.stream(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        tools=[tool],
        tool_choice={"type": "tool", "name": _TOOL_NAME},
        messages=[
            {
                "role": "user",
                "content": f"Extraia os dados desta CG:\n\n<cg>\n{text}\n</cg>",
            }
        ],
    ) as stream:
        resp = stream.get_final_message()

    # Diagnóstico: se o output for truncado (max_tokens), o JSON do tool vem parcial/vazio.
    print(
        f"      stop_reason={resp.stop_reason} "
        f"in={resp.usage.input_tokens} out={resp.usage.output_tokens} tok",
        file=sys.stderr,
    )
    if resp.stop_reason == "max_tokens":
        print(
            "      [aviso] output truncado por max_tokens — suba EXTRACTION_MAX_TOKENS "
            "ou use um doc menor.",
            file=sys.stderr,
        )

    # Com tool_choice forçado, o primeiro (e único) bloco é o tool_use.
    tool_use = next(b for b in resp.content if b.type == "tool_use")
    return ExtractedDocument.model_validate(tool_use.input)
