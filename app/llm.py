import anthropic


def get_client() -> anthropic.Anthropic:
    # parameterless; reads ANTHROPIC_API_KEY from env. Single place to later add
    # retry / timeout / cost tracking / model defaults.
    return anthropic.Anthropic()


def get_async_client() -> anthropic.AsyncAnthropic:
    # Versão async do factory acima, usada pelos nós do grafo (rodam dentro do event
    # loop do FastAPI). O síncrono continua existindo: a extração offline usa ele.
    return anthropic.AsyncAnthropic()
