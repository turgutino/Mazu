from mazu.memory.bm25 import BM25, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Use PostgreSQL, not SQLite!") == ["use", "postgresql", "not", "sqlite"]


def test_relevant_document_scores_higher():
    documents = [
        "Use PostgreSQL for the database, chosen for concurrency",
        "Use React for the frontend, chosen for component reuse",
        "Run tests with pytest before every commit",
    ]
    bm25 = BM25(documents)
    scores = bm25.score("what database does this project use")

    # The PostgreSQL doc should score highest for a database-related query.
    assert scores[0] > scores[1]
    assert scores[0] > scores[2]


def test_no_query_term_overlap_scores_zero():
    documents = ["Use PostgreSQL for the database"]
    bm25 = BM25(documents)
    scores = bm25.score("completely unrelated frontend react topic")
    # "unrelated" etc share no terms with the doc -- score should be 0 (or very low).
    assert scores[0] == 0.0


def test_empty_documents_list_does_not_crash():
    bm25 = BM25([])
    assert bm25.score("anything") == []


def test_empty_query_scores_zero_for_all():
    documents = ["Use PostgreSQL", "Use Redis"]
    bm25 = BM25(documents)
    scores = bm25.score("")
    assert scores == [0.0, 0.0]
