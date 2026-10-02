"""Central configuration: paths, random seed and experiment settings.

Every value that the notebook hard-coded (seed, split size, CV design)
lives here so scripts and tests use exactly the same settings.
"""
from pathlib import Path

# --- Paths (relative to the repository root, never absolute) -----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_FILE = DATA_RAW / "data.csv"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
METRICS = RESULTS / "metrics"
MODELS = RESULTS / "models"

# --- Dataset ------------------------------------------------------------------
KAGGLE_HANDLE = "uciml/breast-cancer-wisconsin-data"
TARGET = "diagnosis"                       # M -> 1 (malignant), B -> 0 (benign)
DROP_COLUMNS = ["id", "Unnamed: 32"]       # identifier + empty trailing column
N_FEATURES = 30
CLASS_NAMES = {0: "Benign", 1: "Malignant"}

# --- Experiment settings (as used in the notebook) ---------------------------
SEED = 42
TEST_SIZE = 0.2
CV_SPLITS = 5
CV_REPEATS = 10
EDA_BOXPLOT_FEATURES = ["radius_mean", "texture_mean", "concavity_mean", "concave points_mean"]
