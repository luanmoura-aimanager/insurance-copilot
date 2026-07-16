"""Testa a seleção da amostra — quem decide ONDE o orçamento é gasto.

Com dinheiro contado, escolher 30 versões do mesmo produto da mesma seguradora seria
volume sem variedade: o SQL não teria o que comparar. Estas regras são o que impede isso.
"""

from pathlib import Path

import pytest

from app.extraction.extract import _unwrap
from app.extraction.sample import eligible_docs, pick_sample


def row(id_, insurer, processo, *, resid=True, texto=True, vigente=True):
    return {
        "id": id_,
        "seguradora": insurer,
        "processo": processo,
        "arquivo": f"{insurer}.pdf",
        "sha256": f"hash{id_}",
        "parece_residencial": resid,
        "tem_texto": texto,
        "vigente": vigente,
    }


@pytest.fixture
def corpus(tmp_path):
    """Diretório com os PDFs 1..9 existindo em disco."""
    for i in range(1, 10):
        (tmp_path / f"susep_{i}.pdf").write_bytes(b"%PDF-")
    return tmp_path


def test_eligible_filters_out_scans_and_old_versions():
    docs = [
        row(1, "A", "p1"),
        row(2, "B", "p2", texto=False),    # scan: sem texto, precisaria de OCR
        row(3, "C", "p3", resid=False),    # não é residencial
        row(4, "D", "p4", vigente=False),  # versão antiga
    ]
    assert [d["id"] for d in eligible_docs(docs)] == [1]


def test_one_doc_per_process(corpus):
    """Várias versões do mesmo produto -> só uma entra."""
    docs = [row(1, "A", "p1"), row(2, "A", "p1"), row(3, "A", "p1")]
    got = pick_sample(docs, 10, corpus)
    assert len(got) == 1


def test_maximizes_insurer_variety(corpus):
    """Com 3 seguradoras e limite 3, sai uma de cada — não 3 da primeira."""
    docs = [
        row(1, "A", "p1"), row(2, "A", "p2"), row(3, "A", "p3"),
        row(4, "B", "p4"), row(5, "B", "p5"),
        row(6, "C", "p6"),
    ]
    got = pick_sample(docs, 3, corpus)
    assert {d["seguradora"] for d in got} == {"A", "B", "C"}


def test_second_round_only_after_covering_everyone(corpus):
    """Passado o limite de seguradoras distintas, aí sim repete."""
    docs = [row(1, "A", "p1"), row(2, "A", "p2"), row(3, "B", "p3")]
    got = pick_sample(docs, 3, corpus)
    assert len(got) == 3
    assert sorted(d["seguradora"] for d in got) == ["A", "A", "B"]


def test_skips_docs_whose_pdf_is_missing(corpus):
    """Doc no manifesto mas sem PDF em disco não pode entrar no batch."""
    docs = [row(1, "A", "p1"), row(999, "B", "p2")]  # 999 não existe em disco
    got = pick_sample(docs, 10, corpus)
    assert [d["id"] for d in got] == [1]


def test_respects_limit(corpus):
    docs = [row(i, f"S{i}", f"p{i}") for i in range(1, 10)]
    assert len(pick_sample(docs, 4, corpus)) == 4


def test_sample_is_reproducible(corpus):
    """Mesma semente = mesma amostra. Sem isso, 'os 30 docs do batch' não é auditável."""
    docs = [row(i, f"S{i}", f"p{i}") for i in range(1, 10)]
    a = pick_sample(docs, 4, corpus, seed=7)
    b = pick_sample(docs, 4, corpus, seed=7)
    assert [d["id"] for d in a] == [d["id"] for d in b]


def test_selection_is_not_alphabetical(corpus):
    """O bug que quase custou US$ 3: ordem alfabética cortada em N vira 'as N primeiras
    do alfabeto', não uma amostra — deixava Porto, Tokio e Zurich sempre de fora."""
    docs = [row(i, f"S{i:02d}", f"p{i}") for i in range(1, 10)]
    got = [d["seguradora"] for d in pick_sample(docs, 9, corpus, seed=1)]
    assert got != sorted(got)  # se fosse alfabético, sairia ordenado


def test_pinned_doc_is_always_included(corpus):
    """O doc do golden precisa entrar mesmo que o sorteio não o escolhesse."""
    docs = [row(i, f"S{i}", f"p{i}") for i in range(1, 10)]
    got = pick_sample(docs, 3, corpus, seed=1, pin_ids=(9,))
    assert 9 in [d["id"] for d in got]
    assert len(got) == 3


def test_pinned_insurer_is_not_duplicated(corpus):
    """Se a seguradora já entrou pelo pin, ela não ganha uma segunda vaga."""
    docs = [row(1, "A", "p1"), row(2, "A", "p2"), row(3, "B", "p3")]
    got = pick_sample(docs, 3, corpus, seed=1, pin_ids=(1,))
    insurers = [d["seguradora"] for d in got]
    assert insurers.count("A") == 1


# --- unwrap: a esquisitice que quebrou a rodada real ---

DOC = {"insurer": "Porto", "product": "X", "susep_process": "1", "coverages": []}


def test_unwrap_leaves_normal_payload_alone():
    assert _unwrap(DOC) == DOC


def test_unwrap_recovers_single_key_wrapper():
    """Variante 1 vista em produção: {"$PARAMETER_NAME": {...doc...}}."""
    assert _unwrap({"$PARAMETER_NAME": DOC}) == DOC


def test_unwrap_recovers_multi_key_wrapper():
    """Variante 2 (VOLTS): duas chaves no topo, doc aninhado em uma delas."""
    assert _unwrap({"pdf_url": "insurer", "record": DOC}) == DOC


def test_unwrap_does_not_descend_into_real_field():
    """Chave única que É campo do schema não pode ser confundida com embrulho."""
    edge = {"coverages": {"nao": "desce"}}
    assert _unwrap(edge) == edge


def test_unwrap_refuses_garbage():
    """Aninhado que NÃO tem os campos obrigatórios não é aceito: melhor falhar
    barulhento no Pydantic do que gravar lixo silenciosamente."""
    garbage = {"algo": {"nada": "a ver"}}
    assert _unwrap(garbage) == garbage
