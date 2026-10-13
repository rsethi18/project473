# CS473 Part 1: Short Answer Grade Prediction

## What's in this repo

| File | What it is | Do you edit it? |
|---|---|---|
| training dataset.csv | 17,207 answers to 10 questions, each graded 0-3 by a teacher | No |
| common.py | The shared test every grader takes (piles, scoring, logging) | No |
| folds.csv | Which of the 5 piles (0-4) each answer is in | No |
| split_90_10.csv | T (90% pool) or P (10% test set) for each answer, for Experiments 2 and 3 | No |
| results.csv | Scoreboard. Every test run adds a row here | No, it fills itself |
| approach_template.py | Starter file for a new grader | Copy it, don't edit the original |
| tfidf_knn.py | Approach 1, finished. Use it as a working example | No |

Data columns: `Id`, `EssaySet` (question number 1-10), `Score1` (the grade we predict), `Score2` (second teacher, ignore), `EssayText` (the answer).

## How the test works

1. Answers are split into 5 piles (saved in folds.csv).
2. One pile's grades are hidden. The other 4 piles stay graded.
3. Your grader guesses a grade for every hidden answer, using the graded ones.
4. Guesses are compared to the real grades and scored with QWK (1 = perfect, 0 = random).
5. Repeat with each pile hidden once. Average the 5 scores.

`run_cv` in common.py does all of this. You only write the grader.

## How to add your grader

1. Copy `approach_template.py` and rename it (e.g. `embedding_knn.py`).
2. Fill in the `predict(tr, te)` function:
   - `tr` = graded answers (a pandas table). `te` = answers to grade.
   - Columns you'll use: `EssayText` (answer), `EssaySet` (question), `Score1` (grade, only look at it in `tr`).
   - Return one whole-number grade (0, 1, 2, or 3) for every row of `te`, in the same order.
   - **Only compare an answer with graded answers to the same question.**
3. Run it: `python embedding_knn.py`. It prints your score and adds it to results.csv.

Smallest possible example (guesses the most common grade for each question, scores 0.425):

```python
from common import run_cv, log_result

def predict(tr, te):
    most_common = tr.groupby("EssaySet")["Score1"].agg(lambda s: s.mode()[0])
    return te["EssaySet"].map(most_common).values

mean, std, _ = run_cv(predict)
print(mean, std)
log_result("baseline", "most common grade per question", mean, std)
```

To try several settings (like different K), call `run_cv` once per setting and `log_result` each time.

## Rules

- Use folds.csv and split_90_10.csv from this repo. Don't make your own splits and don't run common.py (it would rewrite them).
- Don't use `Score1` from `te` inside predict. That's the answer key.
- Don't edit common.py. If something's missing, ask.

## Who does what

- **Approach 1, TF-IDF KNN** (done): best QWK 0.646, stop words removed, stemming, single words, K=10, similarity-weighted average.
- **Approach 2, embedding KNN**: same idea but compare answers by meaning. Try at least two models (e.g. sentence-transformers `all-MiniLM-L6-v2` and `all-mpnet-base-v2`). Save embeddings to a file so they're computed once.
- **Approach 3, classifier**: e.g. logistic regression on TF-IDF or embedding features, trained separately per question. Then write the report.
- **Experiments 2 and 3**: use split_90_10.csv. Train on 10%, 20% ... 90% of the T rows, predict the P rows, score with `qwk` from common.py. Use `tfidf_knn.predict` as the stand-in grader until we know the winner. Pick training subsets per question so each question is covered.

Using split_90_10 for the experiments:

```python
from common import load_with_folds, qwk
import tfidf_knn

df = load_with_folds()
T, P = df[df.set == "T"], df[df.set == "P"]
sub = T.groupby("EssaySet").sample(frac=0.2, random_state=0)   # 20% of the pool, per question
print(qwk(P["Score1"], tfidf_knn.predict(sub, P)))
```


## Setup

```
pip install pandas numpy scikit-learn nltk
```
