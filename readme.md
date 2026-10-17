# CS473 Part 1: Short Answer Grade Prediction


## Files at a glance

| File | What it is | Edit it? |
|---|---|---|
| `training dataset.csv` | The data: 17,207 graded answers | No |
| `common.py` | Shared code: folds, splits, scoring, logging | No |
| `folds.csv` | Which of the 5 CV folds each answer is in | No |
| `split_90_10.csv` | T (90%) or P (10%) for each answer, for Experiments 2 and 3 | No |
| `approach_template.py` | Starter file for a new grader | Copy it, don't edit the original |
| `tfidf_knn.py` | Approach 1 (TF-IDF KNN), finished | No |
| `tfidf_knn_grid.csv` | Scores for all 144 TF-IDF settings, best first | No, generated |
| `show_predictions.py` | Runs the best TF-IDF model on every answer, for error analysis | No |
| `tfidf_predictions.csv` | Every answer with its real grade and TF-IDF's guess | No, generated |
| `results.csv` | Team scoreboard, one row per test run | No, fills itself |
| `.gitignore` | Keeps junk files out of git | No |

## The data

### `training dataset.csv`

17,207 student answers to 10 questions, each graded 0 to 3.

| Column | Meaning |
|---|---|
| `Id` | Unique answer ID |
| `EssaySet` | Question number (1 to 10). The handout calls this "Question Number" |
| `Score1` | **The grade we predict.** The handout calls this "Manual Score" |
| `Score2` | A second teacher's grade |
| `EssayText` | The student's answer |

Confirmed by the TA on Ed: `Score1` is the target. `Score2` can be used during training but will **not** exist at test time, so a grader must never need it to make a prediction. We currently use `Score2` only to report human agreement: the two teachers agree at QWK 0.913, which is the realistic ceiling for any model.

Things to know about the data:
- Grades are imbalanced: 6,779 zeros, 5,612 ones, 4,075 twos, 741 threes.
- Questions 3, 4, 7, 8, 9, and 10 have no grade-3 answers at all.
- 204 answers are exact duplicates of another answer to the same question.
- Answers have lots of misspellings ("expirement").

## Shared setup

### `common.py`

The shared toolbox. It does not grade anything itself. Every approach imports from it so all of our numbers are measured the same way.

What's inside:
- **Loading the data.** Reads the CSV and attaches each answer's fold and T/P label. `load_with_folds()` returns the table with a `set` column (`"T"` or `"P"`) and the fold number.
- **`qwk(true, predicted)`.** Quadratic weighted kappa, the handout's metric. 1 = perfect agreement, 0 = no better than chance. Uses scikit-learn's `cohen_kappa_score` with quadratic weights.
- **`run_cv(predict)`.** The main tool. You give it your grader function; it runs 5-fold cross-validation and returns the mean QWK, the standard deviation, and the per-fold scores.
- **`log_result(name, settings, mean, std)`.** Adds one row to `results.csv`.
- **Making the folds and the 90/10 split.** This code already ran once and saved its output to `folds.csv` and `split_90_10.csv`. **Do not run `common.py` directly**: it would regenerate those files and everyone's numbers would stop being comparable.

How `run_cv` works:
1. The answers are already divided into 5 folds (from `folds.csv`).
2. One fold's grades are hidden. The other 4 folds are the training answers.
3. Your grader predicts a grade for every hidden answer, using only the training answers.
4. Predictions are scored against the real grades with QWK.
5. Repeat with each fold hidden once, then average the 5 scores.

This is exactly the procedure in the handout's "Tuning" section.

### `folds.csv`

One row per answer: its `Id` and which fold (0 to 4) it belongs to. Made once with random seed 473. Each fold has the same mix of questions and grades as the full dataset (stratified by question and grade). The handout requires every approach to use the same folds, which is why this file exists.

### `split_90_10.csv`

One row per answer: its `Id` and whether it is in **T** (15,486 answers, the training pool) or **P** (1,721 answers, the held-out set to predict). Stratified by question and grade, seed 473. Used only for Experiments 2 and 3. Approaches 1 to 3 are tuned with `folds.csv`, not with this.

## Building a grader

### `approach_template.py`

Starter file. Copy it, rename it (for example `embedding_knn.py`), and fill in `predict(tr, te)`:

- `tr` is a pandas table of graded training answers. `te` is a table of answers to grade.
- Use `EssayText` (the answer), `EssaySet` (the question), and `Score1` (the grade, **only from `tr`**).
- Return one whole-number grade (0, 1, 2, or 3) for every row of `te`, in the same order.
- Only compare an answer with training answers to the **same question**.

Then run your file. It prints your score and logs it to `results.csv`.

Smallest working example (predicts each question's most common grade):

```python
from common import run_cv, log_result

def predict(tr, te):
    most_common = tr.groupby("EssaySet")["Score1"].agg(lambda s: s.mode()[0])
    return te["EssaySet"].map(most_common).values

mean, std, _ = run_cv(predict)
print(mean, std)
log_result("baseline", "most common grade per question", mean, std)
```

To try several settings (different K, models, etc.), call `run_cv` once per setting and `log_result` each time.

### `tfidf_knn.py` (Approach 1, done)

A finished grader, and the example to copy the structure from.

How it grades an answer:
1. Clean the text: lowercase, remove stop words, Porter-stem each word (NLTK).
2. Turn every answer into a TF-IDF vector. Rare, specific words get high weight; words almost every answer uses get low weight.
3. Compare the new answer with every training answer to the same question using cosine similarity.
4. Take the K = 10 most similar training answers.
5. Average their grades, with closer answers counting more, and round to 0 to 3.

Running it tries all 144 combinations of: stop words on/off, stemming on/off, unigrams vs unigrams+bigrams, K in {1, 3, 5, 7, 10, 15, 20, 30, 50}, and majority vote vs weighted average. It saves them to `tfidf_knn_grid.csv` and `results.csv`.

Best setting: stop words removed, stemming on, unigrams, K = 10, weighted average. **Mean QWK 0.646 (SD 0.007).**

Findings for the report:
- Stemming helps (0.637 to 0.646). Stop-word removal makes almost no difference.
- Bigrams make it slightly worse (0.632).
- Weighted average beats majority vote at every K from 3 to 15.
- QWK peaks at K = 7 to 10, then falls.
- Character n-grams (handle misspellings) reach 0.652, a gain smaller than the fold-to-fold spread.

`predict(tr, te)` in this file uses the best setting by default, so Experiments 2 and 3 can call it directly.

### `tfidf_knn_grid.csv`

All 144 TF-IDF settings and their scores, sorted best first. Columns: stop words, stemming, n-gram range, K, aggregation method, mean QWK, SD. The top row is the best setting above. Source of the K table and tuning numbers in the report.

### `show_predictions.py`

Runs the best TF-IDF setting through the 5 folds and records the prediction for every answer (each answer is predicted once, when its fold is held out). Writes `tfidf_predictions.csv` and prints how often predictions are exactly right or off by one or more. Run it with `python show_predictions.py`.

### `tfidf_predictions.csv`

One row per answer (all 17,207): `Id`, question, answer text, real grade, and TF-IDF's prediction. Use it to find example successes and errors for the report.

Current numbers: 60.2% exactly right, 37.4% off by one, 2.5% off by two or more. Errors pull toward the middle: most true 3s are predicted as 2, and many true 0s as 1.

## Results

### `results.csv`

The team scoreboard. Every call to `log_result` appends one row: approach name, settings, mean QWK, SD. Right now it holds the 144 TF-IDF runs. Every approach's results end up here, so the final comparison table (Experiment 1) comes from this file.

Everyone appends to this one file, so **run `git pull` before `git push`**. If you get a merge conflict here, keep both sets of lines; nothing in it should ever be deleted.

## Experiments 2 and 3

Use `split_90_10.csv`: train on subsets of T, always predict P.

```python
from common import load_with_folds, qwk
import tfidf_knn

df = load_with_folds()
T, P = df[df.set == "T"], df[df.set == "P"]
sub = T.groupby("EssaySet").sample(frac=0.2, random_state=0)   # 20% of T, per question
print(qwk(P["Score1"], tfidf_knn.predict(sub, P)))
```

- Experiment 2: training sizes of 10%, 20%, ..., 90%, predict P each time, plot number of training examples vs QWK. The handout says to use the best of the three approaches, so swap in the winner once Approach 3 is scored.
- Experiment 3: K-Means++ per question, tune the number of clusters N (heatmap of training size vs N), then pick training examples across clusters instead of at random and compare the two curves on one plot.

## Rules

- Use `folds.csv` and `split_90_10.csv`. Don't make your own splits, and don't run `common.py`.
- Never use `Score1` from `te` inside `predict`. That's the answer key.
- Never require `Score2` to make a prediction. It won't exist at test time.
- Only compare answers within the same question.
- Don't edit `common.py`. If something's missing, ask Ranav.

## Setup

```
pip install pandas numpy scikit-learn nltk
```

Embedding approaches also need `pip install sentence-transformers` (or `gensim` for Word2Vec/Doc2Vec).
