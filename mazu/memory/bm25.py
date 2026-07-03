import math
import re

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25:
    """Minimal Okapi BM25 ranker. Pure Python, no dependencies — runs entirely
    on the user's machine, so ranking memories against a query costs zero API calls.
    """

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(doc) for doc in documents]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_len = (sum(self.doc_lens) / len(self.doc_lens)) if self.doc_lens else 0.0
        self.n_docs = len(documents)

        doc_freq: dict[str, int] = {}
        for toks in self.doc_tokens:
            for term in set(toks):
                doc_freq[term] = doc_freq.get(term, 0) + 1
        self.idf = {
            term: math.log((self.n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in doc_freq.items()
        }

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        scores = []
        for toks, doc_len in zip(self.doc_tokens, self.doc_lens):
            if not toks or not query_terms:
                scores.append(0.0)
                continue
            term_freq: dict[str, int] = {}
            for t in toks:
                term_freq[t] = term_freq.get(t, 0) + 1
            total = 0.0
            for term in query_terms:
                if term not in term_freq:
                    continue
                idf = self.idf.get(term, 0.0)
                f = term_freq[term]
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / (self.avg_len or 1))
                total += idf * (f * (self.k1 + 1)) / (denom or 1)
            scores.append(total)
        return scores
