from pathlib import Path

import config
from document_loader import chunk_document, load_chunks, load_documents, parse_front_matter
from retriever import Retriever

FIXTURES = Path(__file__).parent / "fixtures"


def test_loads_document_with_provenance():
    docs, problems = load_documents(FIXTURES)
    assert problems == []
    d = docs[0]
    assert d.publisher and d.url.startswith("https://") and d.version and d.published


def test_document_without_metadata_is_skipped(tmp_path):
    (tmp_path / "bad.md").write_text("no front matter here", encoding="utf-8")
    (tmp_path / "nodate.md").write_text(
        "---\ntitle: T\npublisher: P\nurl: https://x.org\nversion: 1\n---\nbody text", encoding="utf-8")
    (tmp_path / "badurl.md").write_text(
        "---\ntitle: T\npublisher: P\nurl: ftp://x\nversion: 1\npublished: 2024-01-01\n---\nbody", encoding="utf-8")
    docs, problems = load_documents(tmp_path)
    assert docs == []
    assert len(problems) == 3


def test_missing_folder_is_reported(tmp_path):
    docs, problems = load_documents(tmp_path / "nope")
    assert docs == [] and problems


def test_empty_knowledge_folder_and_readme_are_ignored():
    chunks, docs, problems = load_chunks(config.KNOWLEDGE_DIR)
    assert problems == []  # README.md is ignored, not reported


def test_front_matter_parsing():
    meta, body = parse_front_matter("---\ntitle: A\nurl: https://x.org\n---\nhello")
    assert meta["title"] == "A" and body == "hello"


def test_chunking_splits_long_documents():
    docs, _ = load_documents(FIXTURES)
    chunks = chunk_document(docs[0], max_words=25)
    assert len(chunks) > 1
    assert all(len(c.text.split()) <= 25 for c in chunks)


def test_retrieves_supported_question():
    chunks, _, _ = load_chunks(FIXTURES)
    hits = Retriever(chunks).search("I have a sore throat and want to rest", k=3, min_score=config.MIN_SCORE)
    assert hits and "sore throat" in hits[0].chunk.text.lower()


def test_unsupported_question_returns_nothing():
    chunks, _, _ = load_chunks(FIXTURES)
    assert Retriever(chunks).search("knee replacement surgery recovery", min_score=config.MIN_SCORE) == []


def test_empty_index_returns_nothing():
    assert Retriever([]).search("sore throat") == []


def test_single_generic_word_does_not_count_as_support(tmp_path):
    (tmp_path / "a.md").write_text(
        "---\ntitle: Fever\npublisher: P\nurl: https://x.org\nversion: 1\npublished: 2024-01-01\n---\n"
        "Drink plenty of fluids, with water being best, and rest.", encoding="utf-8")
    chunks, _, _ = load_chunks(tmp_path)
    r = Retriever(chunks)
    assert r.search("best knee replacement surgeon", min_score=config.MIN_SCORE) == []
    assert r.search("best fluids", min_score=config.MIN_SCORE) != []
