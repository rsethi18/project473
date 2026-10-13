"""Approach 1: KNN with TF-IDF, cosine similarity. Neighbors come only from the same question.

Grid: stop-word removal, Porter stemming, n-gram range, K, and how neighbor grades are combined.
For each (preprocessing, fold) the similarity matrix is computed once and every K / combine
setting is scored from it, so the full grid runs in a few minutes.
"""
import re
import itertools
import numpy as np
import pandas as pd
from functools import lru_cache
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from common import load_with_folds, qwk, log_result, N_FOLDS, LABEL, QUESTION, TEXT

_stemmer = PorterStemmer()
_tok = re.compile(r"[a-z0-9]+")


@lru_cache(maxsize=None)
def _stem(w):
    return _stemmer.stem(w)


def preprocess(text, stop, stem):
    toks = _tok.findall(text.lower())
    if stop:
        toks = [t for t in toks if t not in ENGLISH_STOP_WORDS]
    if stem:
        toks = [_stem(t) for t in toks]
    return " ".join(toks)


def neighbors(tr, te, ngram, max_k):
    """For each test row: top max_k same-question neighbor similarities and grades."""
    vec = TfidfVectorizer(ngram_range=ngram, sublinear_tf=True, min_df=1)
    Xtr = vec.fit_transform(tr["clean"])  # fit on training rows only
    Xte = vec.transform(te["clean"])
    sims = np.zeros((len(te), max_k))
    grades = np.zeros((len(te), max_k), dtype=int)
    qmax = np.zeros(len(te), dtype=int)
    tr_q, te_q = tr[QUESTION].values, te[QUESTION].values
    ytr = tr[LABEL].values
    for q in np.unique(te_q):
        ti, si = np.where(te_q == q)[0], np.where(tr_q == q)[0]
        S = (Xte[ti] @ Xtr[si].T).toarray()  # rows are L2-normalised, so this is cosine
        top = np.argsort(-S, axis=1)[:, :max_k]
        sims[ti] = np.take_along_axis(S, top, axis=1)
        grades[ti] = ytr[si][top]
        qmax[ti] = ytr[si].max()
    return sims, grades, qmax


def combine(sims, grades, qmax, k, method):
    s, g = sims[:, :k], grades[:, :k]
    if method == "vote":
        # majority vote, ties broken by summed similarity
        votes = np.zeros((len(g), 4))
        for c in range(4):
            votes[:, c] = (g == c).sum(1) + 1e-6 * ((g == c) * s).sum(1)
        pred = votes.argmax(1)
    else:  # similarity-weighted mean, rounded
        w = s + 1e-9
        pred = np.rint((w * g).sum(1) / w.sum(1)).astype(int)
    return np.clip(pred, 0, qmax)


def predict(tr, te, stop=True, stem=True, ngram=(1, 1), k=10, method="weighted"):
    """Model function for run_cv / experiments: returns integer grades for te."""
    tr, te = tr.copy(), te.copy()
    tr["clean"] = [preprocess(t, stop, stem) for t in tr[TEXT]]
    te["clean"] = [preprocess(t, stop, stem) for t in te[TEXT]]
    sims, grades, qmax = neighbors(tr, te, ngram, k)
    return combine(sims, grades, qmax, k, method)


if __name__ == "__main__":
    df = load_with_folds()
    Ks = [1, 3, 5, 7, 10, 15, 20, 30, 50]
    methods = ["vote", "weighted"]
    rows = []
    for stop, stem, ngram in itertools.product([False, True], [False, True], [(1, 1), (1, 2)]):
        df["clean"] = [preprocess(t, stop, stem) for t in df[TEXT]]
        fold_scores = {(k, m): [] for k in Ks for m in methods}
        for f in range(N_FOLDS):
            tr, te = df[df.fold != f], df[df.fold == f]
            sims, grades, qmax = neighbors(tr, te, ngram, max(Ks))
            for k in Ks:
                for m in methods:
                    fold_scores[(k, m)].append(qwk(te[LABEL], combine(sims, grades, qmax, k, m)))
        for (k, m), sc in fold_scores.items():
            settings = f"stop={stop};stem={stem};ngram={ngram[0]}-{ngram[1]};K={k};combine={m}"
            rows.append(dict(stop=stop, stem=stem, ngram=f"{ngram[0]}-{ngram[1]}", K=k, combine=m,
                             mean_qwk=np.mean(sc), std_qwk=np.std(sc)))
            log_result("tfidf_knn", settings, np.mean(sc), np.std(sc))
        print(f"done stop={stop} stem={stem} ngram={ngram}", flush=True)

    res = pd.DataFrame(rows).sort_values("mean_qwk", ascending=False)
    res.to_csv("tfidf_knn_grid.csv", index=False)
    print("\nTop 10 configurations:\n", res.head(10).round(4).to_string(index=False))