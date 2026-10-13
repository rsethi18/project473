"""Shared setup for CS473 Part 1. Every approach uses these folds, splits and metrics.

Data notes:
- Columns: Id, EssaySet (= question number), Score1, Score2, EssayText.
- Score1 is used as the label ("Manual Score"). Score2 is a second grader, used only
  to report human-human agreement as a ceiling.
"""

import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "training dataset.csv")
FOLDS_PATH = os.path.join(HERE, "folds.csv")
SPLIT_PATH = os.path.join(HERE, "split_90_10.csv")
RESULTS_PATH = os.path.join(HERE, "results.csv")
SEED = 473
N_FOLDS = 5
LABEL = "Score1"
QUESTION = "EssaySet"
TEXT = "EssayText"

def load_data():
    df = pd.read_csv(DATA_PATH)
    df[TEXT] = df[TEXT].fillna("").astype(str)
    return df


def _strata(df):
    # stratify on question and grade together
    return df[QUESTION].astype(str) + "_" + df[LABEL].astype(str)


def make_folds(df):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold = np.zeros(len(df), dtype=int)
    for i, (_, te) in enumerate(skf.split(df, _strata(df))):
        fold[te] = i
    out = pd.DataFrame({"Id": df["Id"], "fold": fold})
    out.to_csv(FOLDS_PATH, index=False)
    return out


def make_split(df):
    tr, te = train_test_split(df["Id"], test_size=0.10, random_state=SEED, stratify=_strata(df))
    out = pd.DataFrame({"Id": df["Id"]})
    out["set"] = np.where(out["Id"].isin(te), "P", "T")
    out.to_csv(SPLIT_PATH, index=False)
    return out


def load_with_folds():
    """Data with 'fold' (0-4) and 'set' (T/P) columns attached."""
    df = load_data()
    if not os.path.exists(FOLDS_PATH):
        make_folds(df)
    if not os.path.exists(SPLIT_PATH):
        make_split(df)
    df = df.merge(pd.read_csv(FOLDS_PATH), on="Id").merge(pd.read_csv(SPLIT_PATH), on="Id")
    return df


def qwk(y_true, y_pred):
    return cohen_kappa_score(np.asarray(y_true, int), np.asarray(y_pred, int), weights="quadratic")


def run_cv(model_fn, df=None):
    """model_fn(train_df, test_df) -> integer grades (0-3) aligned with test_df.
    Returns (mean QWK, std QWK, per-fold list)."""
    if df is None:
        df = load_with_folds()
    scores = []
    for f in range(N_FOLDS):
        tr, te = df[df.fold != f], df[df.fold == f]
        pred = np.asarray(model_fn(tr, te))
        assert len(pred) == len(te), "model returned wrong number of predictions"
        scores.append(qwk(te[LABEL], pred))
    return float(np.mean(scores)), float(np.std(scores)), scores


def log_result(approach, settings, mean, std):
    new = not os.path.exists(RESULTS_PATH)
    with open(RESULTS_PATH, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["approach", "settings", "mean_qwk", "std_qwk"])
        w.writerow([approach, settings, f"{mean:.4f}", f"{std:.4f}"])


if __name__ == "__main__": 
    df = load_data()
    make_folds(df)
    make_split(df)
    d = load_with_folds()
    print("rows per fold:\n", d.fold.value_counts().sort_index().to_string())
    print("grade share per fold:\n", pd.crosstab(d.fold, d[LABEL], normalize="index").round(3).to_string())
    print("P set size:", (d.set == "P").sum(), " T set size:", (d.set == "T").sum())
    print("QWK true vs true:", qwk(d[LABEL], d[LABEL]))
    print("QWK majority-class baseline (CV):", run_cv(lambda tr, te: np.full(len(te), tr[LABEL].mode()[0]))[0])
    print("Human-human QWK (Score1 vs Score2):", round(qwk(d.Score1, d.Score2), 4))