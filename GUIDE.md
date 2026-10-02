# From notebook to GitHub repository — guide

> This file is your working guide. **Do not commit it** (move it out of the repo, or delete it once you are done).
> Everything else in this folder is the repository skeleton.

## 0. What you received and what was verified

| Item | Status |
|---|---|
| `README.md`, `data/README.md`, `docs/methodology.md` | Written from the notebook's actual outputs. Placeholders to fill in: `<your-username>`, `[YOUR NAME]`, author line. |
| `src/` (config, data, models, plots, eda, train, evaluate) | Refactor of the notebook's logic. |
| `tests/` | 8 pytest tests, all passing (run locally, Python 3.12). |
| `requirements.txt`, `environment.yml`, `.gitignore`, `LICENSE`, CI workflow | Written; CI workflow **not executed** (only GitHub can run it). |
| `results/figures/` and `results/metrics/` | Pre-generated so the README renders. **Regenerate them yourself** (Step 6). |

**Verification performed.** I rebuilt the Kaggle CSV layout from scikit-learn's bundled copy of the same WDBC data; its `describe()` output matched your notebook's to every printed digit. Running the new `src/` pipeline on it reproduced your notebook's results exactly: CV ROC-AUC 0.9936 / 0.9858 / 0.9880 (LR / KNN / RF), test ROC-AUC 0.9960, and the confusion matrix 71 / 1 / 3 / 39. So the refactor is faithful. What I could **not** do: download from Kaggle (network restricted), run on your Colab, or execute GitHub Actions.

---

## 1. Notebook analysis

### 1.1 Found directly in the notebook

| Topic | What the notebook contains |
|---|---|
| Dataset / source | Breast Cancer Wisconsin (Diagnostic), Kaggle `uciml/breast-cancer-wisconsin-data`, loaded with `kagglehub`. The output line "Using Colab cache…" shows it ran in Google Colab. |
| Size | 569 samples; 30 numeric features (31 columns after cleaning); raw file also has `id` and an empty `Unnamed: 32`. |
| Target | `diagnosis`: M → 1 (malignant), B → 0 (benign). Class balance 62.7 % / 37.3 % (pie chart; mean of target = 0.372583 → 212 malignant, 357 benign). |
| Cleaning / missing values | Drops `id`, `Unnamed: 32`; asserts 0 missing values. |
| EDA | Pie chart + lower-triangle correlation heatmap of the 10 `_mean` features vs diagnosis; 2×2 boxplots (radius, texture, concavity, concave points); `describe().T.head(10)`. |
| Feature selection | **None** (all 30 features used). |
| Scaling | `StandardScaler` inside a `Pipeline` for Logistic Regression and KNN; none for Random Forest. |
| Split | 80/20 stratified, `random_state=42` → 455 train / 114 test. |
| Cross-validation | `RepeatedStratifiedKFold` 5 × 10, on the training set only; accuracy, precision, recall, F1, ROC-AUC + runtime. |
| Models | LogisticRegression(`max_iter=2000`), KNN(`n_neighbors=7`), RandomForest(`n_estimators=300`). |
| Hyperparameter tuning | **None.** |
| CV results (mean) | LR: acc 0.9723, prec 0.9788, rec 0.9471, F1 0.9621, AUC 0.9936 · RF: 0.9556 / 0.9544 / 0.9271 / 0.9394 / 0.9880 · KNN: 0.9659 / 0.9909 / 0.9176 / 0.9521 / 0.9858 |
| Test evaluation | LR selected (highest mean CV AUC), test AUC 0.9960; report: benign P/R/F1 0.96/0.99/0.97 (n = 72), malignant 0.97/0.93/0.95 (n = 42), accuracy 0.96 (rounded). Confusion matrix and ROC plotted (71 / 1 / 3 / 39). |
| Extra sections | AUC-per-second "efficiency"; NumPy-vs-pandas group-mean benchmark (0.693 ms vs 0.082 ms, 8.5×); scaling simulation on bootstrap-resampled data (1k–100k rows); 5 `assert` integrity checks. |
| Libraries | numpy, pandas, seaborn, matplotlib (+`gridspec`), scikit-learn, kagglehub, `time`. |
| Existing plots (6) | Fig 1 dataset overview; Fig 2 boxplots; Fig 4 model comparison; Fig 5 confusion matrix + ROC; efficiency bar chart; scaling line chart. |
| Hard-coded / environment-specific | `kagglehub` download and `path + "/data.csv"` (Colab cache); Colab metadata. No absolute local paths. |

### 1.2 Not available in the notebook — needs to be added or provided

* Package/Python versions (notebook metadata says Python 3.10.0, but that is not proof of the environment). **You:** run `!pip list | grep -Ei "numpy|pandas|scikit|seaborn|matplotlib|kagglehub"` in Colab and keep the output.
* Citation/licence of the original dataset (UCI, CC BY 4.0, doi:10.24432/C5DW2B — now in the README/data README). The licence of the *Kaggle mirror* is **unverified** — check its page.
* Precision-recall curve, feature importance, calibration, threshold analysis, statistical comparison of models, hyperparameter tuning, external validation. These are *not* in the project; the README lists them as future work rather than claiming them.
* Standard deviations for metrics other than accuracy (now computed by `src/train.py`).

### 1.3 Methodology check (leakage and validity)

**Good — keep and highlight:**
* Scaling is inside the `Pipeline`, so it is fitted per training fold. **No preprocessing leakage.**
* CV and model selection use the training set only; the test set is evaluated once. **No selection leakage.**
* Same CV object (fixed seed) for all models → identical folds → fair paired comparison.
* Dataset checked: 0 missing values, 0 duplicate rows, 0 constant columns (I checked the last two; the notebook does not).

**Issues, most important first:**

1. **Over-interpreted model ranking.** LR's CV ROC-AUC is 0.9936 vs 0.9880 (RF) and 0.9858 (KNN); fold-to-fold SDs are 0.007 / 0.010 / 0.015 (not shown in the notebook). Folds overlap, so this is suggestive, not conclusive. Soften "highest predictive performance" wording.
2. **"AUC per second" is not a robust conclusion.** The metric is essentially 1/runtime (AUC only varies 0.986–0.994), and runtimes depend on machine and parallel start-up (`n_jobs=-1` is set both in `cross_validate` and inside the Random Forest → nested parallelism). On my re-run LR and KNN took 1.4 s and 1.3 s, versus 9.7 s and 2.2 s in your Colab run, so "KNN is ~4× more efficient than LR" does not hold across machines. Random Forest being slowest did reproduce. Report runtime plainly, as machine-dependent.
3. **Truncated bar chart** (y from 0.83) and no error bars exaggerate differences. Replaced by a mean ± SD dot plot.
4. **Hard-coded conclusions in Markdown** ("in our case it's the logistic regression", "KNN is the most efficient in this run"). They go stale if anything changes; keep them but phrase as "in this run" and re-check after re-running.
5. **Small test set.** 42 malignant cases; recall 39/42 has a 95 % exact interval of 0.81–0.99 (my computation from the confusion matrix). State it.
6. **Default 0.5 threshold** although the text argues recall matters most. Not wrong; mention as limitation.
7. **Benchmark sections (6–8) are off-topic for the biological question.** A 0.08 ms vs 0.69 ms difference on 569 rows is irrelevant in practice, and the scaling simulation times each size once (no repeats), so it is noisy. → *Decision for you:* if they are a course requirement, keep them but move to a separate notebook `notebooks/02_performance_benchmarks.ipynb`; if not, drop them. The README currently mentions them in one short paragraph — adjust accordingly.

### 1.4 Code and figure cleanup list

| Where | Problem | Fix |
|---|---|---|
| Fig 1 heatmap | Mask includes the diagonal → first row and the `diagnosis` column are blank | `np.triu(..., k=1)` (done in `plots.py`) |
| Fig 1 pie | Pie for two classes hides counts | Bar chart with counts and % |
| Fig 2 | y-label just "Value" | Use the feature name |
| Figure numbering | Figures 1, 2, **4**, 5 (no Figure 3); two figures unnumbered | Renumber or drop numbers from titles and use captions in the README |
| Fig 5 ROC | Legend shows "AUC = 1.00" while the text says 0.9960 | 3 decimals |
| `Acc ± std` column | Confusing; only for accuracy | Mean and SD for every metric |
| `np.random.seed(42)` | Does not affect scikit-learn (explicit `random_state` is used) — misleading | Remove; keep explicit seeds, defined once (`config.SEED`) |
| Imports | `kagglehub`, `time` imported mid-notebook | Move everything to the imports cell |
| Data loading | Colab cache path | Read `data/raw/data.csv` |
| Random Forest | Wrapped in a one-step Pipeline; nested `n_jobs` | Plain estimator; parallelise only in CV |
| Repeated `fig, ax` styling | Each figure styled separately | One `set_style()` (see `plots.py`) |
| Colab metadata | `"colab": {...}` in notebook JSON | Remove when re-saving locally |

### 1.5 What stays in the notebook vs moves to scripts

| Notebook section | Destination |
|---|---|
| Title, motivation, research question, feature explanation, interpretations | **Stays in the notebook** (this is the narrative recruiters read) |
| Loading & cleaning | `src/data.py` (`load_dataset`) — notebook calls it |
| EDA plots | `src/plots.py` + `src/eda.py`; notebook may call the same functions to display them |
| Split, model definitions | `src/data.py`, `src/models.py` |
| CV loop | `src/train.py` |
| Test evaluation | `src/evaluate.py` |
| Interpretation of results | Stays in the notebook |
| Benchmarks / scaling | Separate appendix notebook (or delete) |
| Integrity asserts | `tests/` (pytest) |

Recommended notebook outline (adapted to your content):

```text
1. Project overview & research question     8. Model comparison (CV) — table + dot plot
2. Imports & configuration                    9. Test-set evaluation — report, confusion matrix, ROC
3. Data loading (from data/raw)              10. Interpretation and limitations
4. Data inspection & feature description     11. Conclusions
5. Exploratory analysis (3 figures)          Appendix: runtime benchmarks (separate notebook)
6. Train/test split
7. Models & cross-validation setup
```

Keep saved outputs in the final notebook (GitHub renders them). Re-run **Restart & Run All** before committing so the execution counts are consistent (yours are currently empty).

---

## 2. Repository design

```text
breast-cancer-classification-wdbc/
├── README.md              ✔ required
├── LICENSE                ✔ required
├── .gitignore             ✔ required
├── requirements.txt       ✔ required
├── environment.yml        ○ optional (delete if you don't want to maintain two files)
├── pytest.ini             ✔ (makes `pytest` find `src/`)
├── data/README.md         ✔ required (data itself is NOT committed)
├── notebooks/             ✔ your narrative notebook
├── src/                   ✔ config, data, models, plots, eda, train, evaluate, download_data
├── tests/                 ✔ small but signals engineering discipline
├── results/{figures,metrics}   ✔ committed (small, needed by the README)
├── results/models/        ○ folder kept empty via .gitkeep; binaries ignored
├── docs/methodology.md    ○ optional but cheap and useful
└── .github/workflows/     ○ optional (one CI badge is a credible signal)
```

**Adapted from your template:** no `data/processed` content (cleaning is fast, nothing to cache — the folder exists only as a placeholder; delete it if you prefer); `data_preprocessing.py` became `data.py` (it also loads/validates/splits); `visualization.py` became `plots.py`; added `config.py`, `eda.py`, `download_data.py`; `environment.yml` marked optional.

**Do not commit:** `data/raw/*`, trained `.joblib`/`.pkl` files, `.venv/`, `.ipynb_checkpoints/`, `kaggle.json`/`.env`, `__pycache__/`.
**Do commit:** figures, `cv_results.csv`, `test_metrics.json` (small and citable).

### `.gitignore` decisions
* I did **not** use a blanket `*.csv` / `*.xlsx`: it would silently exclude `results/metrics/cv_results.csv`, which you *want* in the repo. Raw data is excluded by path (`data/raw/*`) instead.
* `!data/raw/.gitkeep` keeps the empty folders in Git so that a fresh clone has the right structure.
* `*.pkl` / `*.joblib` are ignored because pickles are scikit-learn-version-specific and are re-created by `python -m src.train`.

---

## 3. README

Already written (`README.md`). Before publishing:
* Replace `<your-username>`, `[YOUR NAME]`, the author line.
* If you delete/move the benchmark sections, edit the "Computational cost" paragraph.
* Confirm the three figures shown at the top are the ones you regenerated.
* Keep the disclaimer near the top.

## 4. Dataset decision

**Recommendation: download automatically (with a manual fallback), do not commit the CSV.**

| Option | Verdict |
|---|---|
| Commit the CSV | The UCI original is CC BY 4.0 (redistribution allowed with attribution), but the **Kaggle mirror's licence is unverified** and the file is not yours. Avoid unless you check. |
| Automatic download | ✔ `python -m src.download_data` (uses `kagglehub`, as in your notebook). |
| Manual download | ✔ documented fallback in `data/README.md`. |
| Alternative source | `ucimlrepo` (`fetch_ucirepo(id=17)`) fetches the UCI original directly; the column names differ from the Kaggle CSV, so switching would mean adapting `data.py`. Not needed now. |

Note: scikit-learn also ships this dataset (`load_breast_cancer`), but with **inverted labels** (0 = malignant) and different column names. That is why the tests reshape it to the Kaggle layout — do not mix the two encodings in analysis code.

## 5. Figures

Style: white grid, colour-blind-safe (Okabe–Ito) palette, benign = blue, malignant = orange, 200 dpi PNG, tight bounding box, titles that state what is shown, axes labelled. All defined once in `src/plots.py`.

| File | Purpose | Change vs notebook | In README? |
|---|---|---|---|
| `class_distribution.png` | Show class balance with counts | pie → bar with n and % | ✔ |
| `correlation_matrix.png` | Collinearity + link to diagnosis | mask fixed, diverging blue–red map | ✔ |
| `feature_distributions.png` | Class separation for 4 key features | axis labels named | optional (link in docs) |
| `model_comparison.png` | CV comparison with variability | bar → mean ± SD dot plot, no truncated baseline | ✔ **hero figure** |
| `confusion_matrix.png` | Test-set errors as counts | separate file, cleaner title | ✔ |
| `roc_curve.png` | Test-set ranking quality | AUC to 3 decimals, chance line | ✔ |
| `cv_runtime.png` | Runtime (machine-dependent) | replaces "AUC per second" | ✘ (keep in `results/`) |

Not recommended (unsupported by the notebook): feature-importance, precision-recall, calibration plots. Add them only if you actually run those analyses.

## 6. requirements.txt

Provided, grouped: core / visualisation / download / Jupyter / testing. **Versions are lower bounds inferred from the code**, because the notebook records no versions (seaborn ≥ 0.13 for the `legend=` argument; scikit-learn ≥ 1.0 for `RocCurveDisplay.from_estimator`, I set ≥ 1.3 to be safe). To pin exact versions once your clean install works:

```bash
pip freeze > requirements-lock.txt      # commit it; keep requirements.txt readable
```

## 7. GitHub presentation

* **Repository name:** `breast-cancer-classification-wdbc`. "Classification" is more accurate than "detection": the task is benign/malignant classification of already-sampled masses, not screening. (Keep `breast-cancer-detection` if you prefer; adjust the README URLs.)
* **Description:** *Reproducible scikit-learn comparison of Logistic Regression, KNN and Random Forest for benign/malignant classification on the Wisconsin Diagnostic Breast Cancer dataset. Educational — not a clinical tool.*
* **Topics:** `machine-learning`, `breast-cancer`, `classification`, `scikit-learn`, `python`, `jupyter-notebook`, `data-science`, `computational-biology`. (`computational-biology` is a stretch — the data are image-derived tabular features, not omics — so include it only if that framing fits your CV.)
* **Badges:** Python, licence, CI — three is enough.
* **Visible first:** title → disclaimer → *Key results* with the model-comparison figure.
* **Pin** the repository on your profile. Optional: use `model_comparison.png` as the social-preview image (Settings → General → Social preview, 1280×640 works best).

## 8. Commit sequence

| # | Commit message | Content |
|---|---|---|
| 1 | `Initial project structure` | folders, `.gitkeep` files, `.gitignore`, `LICENSE`, `pytest.ini` |
| 2 | `Add dataset documentation` | `data/README.md` |
| 3 | `Add configuration and data-loading module` | `src/__init__.py`, `config.py`, `data.py`, `download_data.py`, `tests/conftest.py`, `tests/test_data.py` |
| 4 | `Add exploratory data analysis` | `plots.py` (EDA functions), `eda.py`, first three figures |
| 5 | `Add model definitions and cross-validated comparison` | `models.py`, `train.py`, `tests/test_pipeline.py` |
| 6 | `Add held-out test-set evaluation` | `evaluate.py` |
| 7 | `Add publication-quality figures and metrics` | `results/figures/*`, `results/metrics/*` |
| 8 | `Add cleaned analysis notebook` | `notebooks/breast_cancer_detection.ipynb` |
| 9 | `Add methodology, README and reproducibility files` | `README.md`, `docs/methodology.md`, `requirements.txt`, `environment.yml` |
| 10 | `Add CI workflow and final cleanup` | `.github/workflows/tests.yml`, fixes found in the audit |

## 9. Step-by-step workflow

**STEP 1 — Inspect and freeze the notebook.**
Open it in Colab, run `!pip list | grep -Ei "numpy|pandas|scikit|seaborn|matplotlib|kagglehub"` and save the output. *Why:* the only source of real versions. *Check:* you have the versions written down.

**STEP 2 — Clean the notebook** (use §1.4 and the outline in §1.5). Decide about the benchmark sections. Load data from `data/raw/data.csv`. **Runtime → Restart and run all.** *Check:* runs top to bottom without errors, numbers unchanged (LR test AUC 0.9960).

**STEP 3 — Create the repository.**
On GitHub: New repository → name from §7 → *no* README/licence/gitignore (you already have them) → create.

**STEP 4 — Create the structure locally.**
```bash
unzip breast-cancer-classification-wdbc.zip && cd breast-cancer-classification-wdbc
rm GUIDE.md        # or move it somewhere outside the repo
cp /path/to/your_cleaned_notebook.ipynb notebooks/breast_cancer_detection.ipynb
```
*Check:* `tree -a -I .git` matches §2.

**STEP 5 — Review the scripts.** Read every file in `src/` once so you can explain it in an interview. Edit `LICENSE` (your name).

**STEP 6 — Install and generate everything on your machine.**
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.download_data && python -m src.eda && python -m src.train && python -m src.evaluate
pytest
```
*Expected:* `Training set: 455 samples`; CV AUC 0.9936 / 0.9858 / 0.9880; test AUC 0.9960; 8 tests pass. *Check:* if the numbers differ, stop and investigate before publishing (different library versions may change the last digits; large differences mean something is wrong).

**STEP 7 — Look at the figures** in `results/figures/` at full size: labels readable, no clipping, titles correct.

**STEP 8 — Freeze versions (optional but recommended):** `pip freeze > requirements-lock.txt`.

**STEP 9 — Finish the README:** placeholders, author line, benchmark paragraph, real timings if you want to quote yours.

**STEP 10 — Test from scratch.** In a *new* folder: `git clone` your local copy, create a fresh venv, follow the README literally. *Check:* it works with no manual fixes. This is the step most people skip and reviewers notice.

**STEP 11 — Initialise Git.**
```bash
git init -b main
git config user.name "Your Name" && git config user.email "you@example.com"
```

**STEP 12 — Commit in the sequence from §8**, e.g.
```bash
git add .gitignore LICENSE pytest.ini data/raw/.gitkeep data/processed/.gitkeep results/*/.gitkeep
git commit -m "Initial project structure"
# ...repeat for each row of the table; check with `git status` that no data or .joblib is staged
```

**STEP 13 — Push.**
```bash
git remote add origin https://github.com/<your-username>/breast-cancer-classification-wdbc.git
git push -u origin main
```
Then add description and topics (§7). *Check:* the Actions tab shows a green run.

**STEP 14 — Final audit.** Open the repo in a private/incognito window as a stranger would, then go through §10.

## 10. Quality-control checklist

**Scientific quality**
- [ ] Dataset documented (source, DOI, licence, citation)
- [ ] Methodology documented (`docs/methodology.md`)
- [ ] Evaluation appropriate (CV on train, test once, recall reported)
- [ ] No unsupported claims (no "best model" without variability; no clinical claims)
- [ ] Limitations documented; disclaimer visible

**Code quality**
- [ ] No hard-coded local paths (grep for `/content`, `C:\`, `/Users`)
- [ ] Preprocessing inside the Pipeline (leakage test passes)
- [ ] Seeds controlled from `config.py`
- [ ] No unused code or commented-out blocks
- [ ] Notebook was run top-to-bottom before committing

**GitHub quality**
- [ ] README complete, placeholders replaced
- [ ] Structure clean; `.gitignore` present
- [ ] `requirements.txt` present; versions checked
- [ ] LICENSE present with your name
- [ ] Figures included and render in the README
- [ ] No large files (`git count-objects -vH`), no data CSV, no `.joblib`
- [ ] No credentials (`kaggle.json`, `.env`)
- [ ] Fresh-clone installation tested

**Portfolio quality**
- [ ] Objective clear in the first 10 seconds
- [ ] Results table visible without scrolling far
- [ ] Figures readable on a laptop screen
- [ ] Skills visible: scikit-learn pipelines, CV, testing, CI
- [ ] Limitations acknowledged
- [ ] Repo looks maintained (green CI, meaningful commits)

## 11. Do this first

1. **Run `!pip list …` in Colab** and save the versions (Step 1).
2. **Decide about notebook sections 6–8** (benchmarks/scaling): course requirement → move to an appendix notebook; otherwise drop.
3. **Check the Kaggle dataset's licence** on its page, and keep the "download, don't commit" approach either way.
4. **Fix the notebook issues** in §1.4 (heatmap mask, truncated bar chart, ROC legend, figure numbering, hard-coded conclusions), then *Restart and run all*.
5. **Unzip the skeleton, run Step 6** and confirm you reproduce the numbers above before writing anything else.
