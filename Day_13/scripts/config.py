"""
config.py
=========
All constants, hyperparameters, and configuration for MeetMux.
Change values here — the rest of the project picks them up automatically.
"""

import os

# ── Project paths ──────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "../data")
OUTPUTS_DIR = os.path.join(BASE_DIR, "../outputs")
MODELS_DIR  = os.path.join(BASE_DIR, "../models")

for _dir in [DATA_DIR, OUTPUTS_DIR, MODELS_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ── Dataset ────────────────────────────────────────────────────────────────────
RANDOM_SEED      = 42
NUM_USERS        = 300
NUM_PAIRS        = 3000
MATCH_THRESHOLD  = 0.55      # raw score >= this → labelled as Match

# ── Feature categories ─────────────────────────────────────────────────────────
DOMAINS = ["AI/ML", "Data Science", "Backend", "Frontend", "Product", "Design"]
GOALS   = ["Find co-founder", "Get hired", "Hire talent", "Learn & grow", "Build network"]
COMM_STYLES = ["Direct", "Collaborative", "Analytical", "Storyteller"]
LOCATIONS   = ["Mumbai", "Delhi", "Bangalore", "Pune", "Remote"]

# ── Compatibility rules ────────────────────────────────────────────────────────
DOMAIN_GROUPS = {
    "AI/ML": 0, "Data Science": 0,
    "Backend": 1, "Frontend": 1,
    "Product": 2, "Design": 2,
}

GOAL_COMPAT = {
    "Find co-founder": ["Find co-founder", "Learn & grow", "Build network"],
    "Get hired"      : ["Hire talent"],
    "Hire talent"    : ["Get hired"],
    "Learn & grow"   : ["Find co-founder", "Learn & grow", "Build network"],
    "Build network"  : ["Find co-founder", "Learn & grow", "Build network"],
}

COMM_COMPAT = {
    "Direct"       : ["Direct", "Analytical"],
    "Collaborative": ["Collaborative", "Storyteller"],
    "Analytical"   : ["Direct", "Analytical"],
    "Storyteller"  : ["Collaborative", "Storyteller"],
}

# ── Feature weights (used in label generation) ────────────────────────────────
FEATURE_WEIGHTS = {
    "domain_sim": 0.28,
    "goal_sim"  : 0.24,
    "comm_sim"  : 0.18,
    "exp_sim"   : 0.16,
    "loc_sim"   : 0.14,
}
FEATURE_NAMES = list(FEATURE_WEIGHTS.keys())

# ── Model hyperparameters ──────────────────────────────────────────────────────
RF_PARAMS = {
    "n_estimators"    : 200,
    "max_depth"       : 6,
    "min_samples_split": 5,
    "class_weight"    : "balanced",
    "random_state"    : RANDOM_SEED,
    "n_jobs"          : -1,
}

POLY_DEGREE       = 2
TEST_SIZE         = 0.2

# ── Prediction thresholds ──────────────────────────────────────────────────────
SCORE_GOOD   = 70    # score >= 70  → "Good Match"
SCORE_MEDIUM = 45    # score >= 45  → "Possible Match"
                     # score <  45  → "Low Compatibility"

# ── Plot style ─────────────────────────────────────────────────────────────────
PALETTE = {
    "primary"  : "#1D9E75",
    "secondary": "#378ADD",
    "accent"   : "#534AB7",
    "warn"     : "#BA7517",
    "danger"   : "#D85A30",
    "neutral"  : "#888780",
}
