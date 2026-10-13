"""Saves TF-IDF KNN's predicted grade for every answer next to the real grade.
Each answer is predicted while its own fold is hidden, so the model never saw its real grade."""
import numpy as np
from common import load_with_folds, N_FOLDS, LABEL, QUESTION, TEXT
import tfidf_knn as T

df = load_with_folds()
df["predicted"] = -1
for f in range(N_FOLDS):
    tr, te = df[df.fold != f], df[df.fold == f]
    df.loc[te.index, "predicted"] = T.predict(tr, te)
out = df[["Id", QUESTION, TEXT, LABEL, "predicted"]].rename(
    columns={QUESTION: "question", TEXT: "answer", LABEL: "real_grade"})
out.to_csv("tfidf_predictions.csv", index=False)
print("exact match:", round((out.real_grade == out.predicted).mean(), 3))
print("off by one:", round(((out.real_grade - out.predicted).abs() == 1).mean(), 3))
print("off by 2+:", round(((out.real_grade - out.predicted).abs() >= 2).mean(), 3))
