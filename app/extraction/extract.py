"""Chamada LLM com structured output (Anthropic tool-use).

Padrão canônico de structured output: expõe um único "tool" cujo input_schema é o
JSON Schema do Pydantic e força tool_choice pra esse tool. O modelo é obrigado a
devolver argumentos que batem com o schema (incl. os enums de perigo) → parseamos
o bloco tool_use direto no Pydantic. Sem regex, sem JSON solto no meio do texto.
"""

import os
import sys
from dataclasses import dataclass

import anthropic
from pydantic import ValidationError

from .schema import ExtractedDocument

# Sonnet é o default pra tarefa estruturada (Opus fica pra system design/ADR).
MODEL = os.getenv("EXTRACTION_MODEL", "claude-sonnet-5")
# CG de 122p com exclusões verbatim gera output grande; Sonnet suporta bem mais que 16k.
MAX_TOKENS = int(os.getenv("EXTRACTION_MAX_TOKENS", "32000"))

_TOOL_NAME = "record_extraction"


class ExtractionFailed(Exception):
    """A chamada aconteceu (e foi COBRADA), mas a resposta não virou objeto válido.

    Carrega o uso de tokens de propósito: o custo é incorrido na CHAMADA, não no parse.
    Sem isso, uma falha de validação viraria gasto invisível — dinheiro que saiu e não
    aparece em cost_event. Também carrega o payload cru pra debugar sem re-chamar a API.
    """

    def __init__(self, message: str, model: str, input_tokens: int, output_tokens: int, raw):
        super().__init__(message)
        self.model = model
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.raw = raw


_REQUIRED = {"insurer", "product", "susep_process", "coverages"}


def _looks_like_document(value) -> bool:
    """O objeto tem a cara do nosso documento (traz os campos obrigatórios)?"""
    return isinstance(value, dict) and _REQUIRED.issubset(value.keys())


def _unwrap(payload):
    """Acha o documento quando a resposta vem embrulhada.

    Duas variantes já vistas em produção, ambas do modelo (não do nosso código):
      {"$PARAMETER_NAME": {...documento...}}          -> 1 chave
      {"pdf_url": "...", "record": {...documento...}} -> 2 chaves, doc aninhado em uma

    A regra é conservadora: só desce se o objeto aninhado REALMENTE tiver os campos
    obrigatórios do schema. Sem isso, um unwrap esperto demais aceitaria silenciosamente
    qualquer lixo aninhado — e a gente prefere falhar barulhento a gravar dado errado.
    """
    if not isinstance(payload, dict) or _looks_like_document(payload):
        return payload
    for value in payload.values():
        if _looks_like_document(value):
            return value
    return payload


@dataclass(frozen=True)
class ExtractionResult:
    """O documento extraído + o que a chamada consumiu.

    O uso de tokens vem junto de propósito: quem faz a chamada é quem sabe o custo dela.
    Descartar isso aqui obrigaria a adivinhar depois (ver app/cost.py).
    """

    document: ExtractedDocument
    model: str
    input_tokens: int
    output_tokens: int

SYSTEM_PROMPT = """\
Você extrai dados estruturados de Condições Gerais (CG) de seguro residencial \
brasileiro registradas na SUSEP. A CG descreve um PRODUTO, não a apólice de um cliente.

Regras de extração:
- GRÃO: uma entrada por COBERTURA (não uma por documento). Liste todas as coberturas, \
básicas e adicionais.
- CONDIÇÕES ESPECIAIS: muitas CGs descrevem a cobertura BÁSICA no corpo das Condições \
Gerais e as ADICIONAIS numa seção de "Condições Especiais" mais adiante, no MESMO \
documento. Percorra o documento INTEIRO e liste as coberturas das duas partes. Parar na \
cobertura básica é o erro mais comum nesta tarefa: se o documento tem seção de Condições \
Especiais, cada cobertura descrita lá também vira uma entrada.
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


def build_params(text: str, model: str | None = None) -> dict:
    """Monta os parâmetros da chamada — UMA definição só, usada pelo caminho síncrono
    e pelo batch. Se cada caminho montasse o seu, o prompt divergiria com o tempo e a
    eval passaria a comparar coisas diferentes sem ninguém perceber.
    """
    return {
        "model": model or MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "tools": [
            {
                "name": _TOOL_NAME,
                "description": (
                    "Registra os dados estruturados extraídos de uma CG de seguro residencial."
                ),
                "input_schema": ExtractedDocument.model_json_schema(),
            }
        ],
        "tool_choice": {"type": "tool", "name": _TOOL_NAME},
        "messages": [
            {"role": "user", "content": f"Extraia os dados desta CG:\n\n<cg>\n{text}\n</cg>"}
        ],
    }


def parse_response(content, model: str, in_tok: int, out_tok: int) -> ExtractedDocument:
    """Extrai o documento validado dos blocos de resposta. Compartilhado por síncrono e batch.

    Toda falha aqui carrega o uso: neste ponto a chamada JÁ FOI COBRADA.
    """
    tool_use = next((b for b in content if b.type == "tool_use"), None)
    if tool_use is None:
        raise ExtractionFailed(
            "resposta sem bloco tool_use", model, in_tok, out_tok, raw=None
        )
    payload = _unwrap(tool_use.input)
    try:
        return ExtractedDocument.model_validate(payload)
    except ValidationError as exc:
        raise ExtractionFailed(
            f"tool_use.input não bate com o schema: {exc}",
            model, in_tok, out_tok, raw=tool_use.input,
        ) from exc


def extract_document(text: str, model: str | None = None) -> ExtractionResult:
    """Extração SÍNCRONA de UMA CG (1 doc, preço cheio). Para volume, use o batch."""
    model = model or MODEL
    client = anthropic.Anthropic()  # lê ANTHROPIC_API_KEY do ambiente

    # Streaming: com max_tokens alto a SDK exige stream (geração pode passar de 10 min).
    # Não usamos os eventos token-a-token — só acumulamos e pegamos a mensagem final.
    with client.messages.stream(**build_params(text, model)) as stream:
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

    in_tok, out_tok = resp.usage.input_tokens, resp.usage.output_tokens
    return ExtractionResult(
        document=parse_response(resp.content, model, in_tok, out_tok),
        model=model,
        input_tokens=in_tok,
        output_tokens=out_tok,
    )
