import anthropic


def get_client() -> anthropic.Anthropic:
    # parameterless; reads ANTHROPIC_API_KEY from env. Single place to later add
    # retry / timeout / cost tracking / model defaults.
    return anthropic.Anthropic()
