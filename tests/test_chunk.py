from babyinfoquick.age import AgeBand
from babyinfoquick.config import ChunkingConfig
from babyinfoquick.corpus.chunk import (
    chunk_doc,
    find_running_headers,
    infer_age_bands,
    is_heading,
)
from babyinfoquick.corpus.parse import Page, ParsedDoc
from babyinfoquick.types import SourceDoc

DOC_BANDS = [
    AgeBand.NEWBORN,
    AgeBand.INFANT_1_3,
    AgeBand.INFANT_4_6,
    AgeBand.INFANT_7_12,
]


def test_heading_detection():
    assert is_heading("Your baby at 2 months")
    assert is_heading("Storing Infant Formula")
    assert is_heading("SAFE SLEEP")
    assert not is_heading("- Calms down when spoken to or picked up")
    assert not is_heading("This is an ordinary sentence of body text that runs on.")


def test_mid_sentence_fragment_is_not_a_heading():
    assert not is_heading("the U.S. Food and Drug Administration (FDA). The")


def test_age_headings_survive_running_header_filter():
    pages = [
        Page(page=n, text=f"Your baby at {n} months\nSocial Milestones\nsome body text here")
        for n in (2, 4, 6, 9)
    ]
    headers = find_running_headers(ParsedDoc(doc_id="d", pages=pages))
    assert not any("your baby at" in h for h in headers)
    assert "social milestones" in headers


def test_section_heading_drives_age_band():
    assert infer_age_bands("Your baby at 9 months", "body", DOC_BANDS) == [
        AgeBand.INFANT_7_12
    ]


def test_falls_back_to_doc_bands():
    assert infer_age_bands(None, "generic body text", DOC_BANDS) == DOC_BANDS


def _doc() -> SourceDoc:
    return SourceDoc(
        doc_id="d",
        title="T",
        url="https://example.org/d.pdf",
        publisher="P",
        doc_type="guidance",
        age_bands=DOC_BANDS,
    )


def test_section_chunking_assigns_per_section_bands():
    parsed = ParsedDoc(
        doc_id="d",
        pages=[
            Page(
                page=1,
                text=(
                    "Your baby at 2 months\n"
                    + "Babies this age begin to smile and coo at familiar faces. " * 6
                    + "\nYour baby at 9 months\n"
                    + "Babies this age can sit without support and pick up small items. " * 6
                ),
            )
        ],
    )
    chunks = chunk_doc(parsed, _doc(), ChunkingConfig(strategy="section"))
    by_section = {c.section: c.age_bands for c in chunks}
    assert by_section["Your baby at 2 months"] == [AgeBand.INFANT_1_3]
    assert by_section["Your baby at 9 months"] == [AgeBand.INFANT_7_12]


def test_fixed_chunking_has_no_sections():
    parsed = ParsedDoc(doc_id="d", pages=[Page(page=1, text="word " * 900)])
    chunks = chunk_doc(parsed, _doc(), ChunkingConfig(strategy="fixed"))
    assert len(chunks) > 1
    assert all(c.section is None for c in chunks)
