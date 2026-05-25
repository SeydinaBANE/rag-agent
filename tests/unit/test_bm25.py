from rag_agent.services.retriever import _bm25_scores, _rrf_fuse


def test_bm25_ranks_relevant_first() -> None:
    docs = [
        "The cat sat on the mat",
        "RAG is a technique combining retrieval and generation",
        "RAG retrieval augmented generation improves LLM answers",
    ]
    scores = _bm25_scores("RAG retrieval", docs)
    assert scores[0] < scores[1]  # cat doc scores lower
    assert scores[2] >= scores[1]  # most relevant doc scores highest


def test_bm25_no_match_returns_zero() -> None:
    docs = ["completely unrelated content here"]
    scores = _bm25_scores("xyz123 nonexistent", docs)
    assert scores[0] == 0.0


def test_rrf_fuse_returns_all_results() -> None:
    dense = [{"text": f"doc{i}", "score": 1.0 - i * 0.1} for i in range(5)]
    bm25_scores = [0.5, 0.8, 0.2, 0.9, 0.4]
    fused = _rrf_fuse(dense, bm25_scores)
    assert len(fused) == 5
    assert all("rrf_score" in r for r in fused)
