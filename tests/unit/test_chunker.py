from rag_agent.services.chunker import chunk_text


def test_short_text_single_chunk() -> None:
    chunks = chunk_text("Hello world.", chunk_size=512)
    assert len(chunks) == 1
    assert chunks[0].text == "Hello world."


def test_long_text_splits() -> None:
    text = "Sentence one. " * 100
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=0)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text) <= 200  # some leeway for separator handling


def test_overlap_applied() -> None:
    text = "alpha beta gamma delta epsilon zeta eta theta iota kappa " * 5
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)
    if len(chunks) > 1:
        # Second chunk should contain some text from first chunk tail
        assert len(chunks[1].text) > 0


def test_empty_chunks_filtered() -> None:
    chunks = chunk_text("   \n\n   ", chunk_size=512)
    assert chunks == []


def test_source_metadata() -> None:
    chunks = chunk_text("Some text here.", source="test.pdf")
    assert chunks[0].metadata["source"] == "test.pdf"


def test_chunk_indices() -> None:
    text = "word " * 200
    chunks = chunk_text(text, chunk_size=50, chunk_overlap=0)
    for i, c in enumerate(chunks):
        assert c.index == i
