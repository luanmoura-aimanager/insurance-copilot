"""Ler a resposta do grafo — o lado de fora do contrato de saída dele.

Mora aqui, e não em `app/main.py` (onde nasceu) nem em `app/inbox.py` (onde passou primeiro),
porque tem DOIS leitores — o `/ask` e a varredura do WhatsApp — e nenhum dos dois é dono da
regra. Uma cópia divergiria no dia em que o nome da mensagem terminal mudasse: mesmo argumento
do `ERRO_PREFIXO` exportado pelo produtor e do `_filtro_pesquisavel` compartilhado entre a
busca e a contagem.

Deixá-la em `app/inbox.py` invertia a camada: o `/ask` passava a importar a borda de SAÍDA do
WhatsApp (httpx, token da Cloud API) só pra ler o próprio resultado, e uma mudança no envio
podia quebrar o caminho do `/ask`.
"""
from langchain_core.messages import AIMessage

from app.agents.graph import NO_ANSWER


def final_answer(state: dict) -> str:
    """A frase do synthesizer, que é sempre o último nó do grafo.

    O preço de uma cópia divergente é maior no WhatsApp do que no `/ask`: lá o sistema
    mandaria o raciocínio interno de um agente para o telefone de alguém, e isso não dá pra
    retirar depois. Ver o docstring do módulo.

    Não há mais o que varrer: todo caminho de saída passa pelo synthesizer (inclusive o que
    não roda worker nenhum), então a resposta é literalmente a última mensagem. O NO_ANSWER
    aqui é só cinto: se o histórico vier com outra coisa no fim, não devolvemos o raciocínio
    interno de um agente como se fosse resposta.

    **Falha de worker também vira frase, não exceção**, e o grafo já cuida disso
    (`FALHA_INTERNA`, via `_falha_de_worker`). A superfície deste sistema é conversa: no
    `/ask` o equivalente do erro seria um 5xx, aqui é o **silêncio** — o usuário fica sem
    resposta E sem saber se vale tentar de novo. Uma frase que diz "falhei do meu lado" é a
    informação que ele pode usar; o operador é servido pelo log, que carrega o traceback.
    """
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.name == "final":
        return last.content
    return NO_ANSWER
