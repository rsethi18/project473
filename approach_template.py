"""Template for a new approach. Copy this file, rename it (e.g. embedding_knn.py),
and fill in predict(). Do not edit common.py.
"""
import numpy as np
from common import run_cv, log_result, LABEL, QUESTION, TEXT


def predict(tr, te):
    """tr = graded training answers, te = answers to grade (both pandas DataFrames).
    Useful columns: tr[TEXT] answer text, tr[QUESTION] question number, tr[LABEL] grade.
    Must return one integer grade (0-3) per row of te, in the same order.
    Only compare an answer with training answers to the SAME question."""
    # placeholder: predict the most common grade for each question
    common_grade = tr.groupby(QUESTION)[LABEL].agg(lambda s: s.mode()[0])
    return te[QUESTION].map(common_grade).values


if __name__ == "__main__":
    mean, std, folds = run_cv(predict)          # tests predict() on the shared 5 folds
    print(f"mean QWK {mean:.4f}  std {std:.4f}")
    log_result("my_approach", "describe settings here", mean, std)  # adds a row to results.csv
