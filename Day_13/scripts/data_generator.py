"""
data_generator.py

Generates synthetic attendee profiles and pairwise training examples.

Usage:
    from data_generator import generate_users, generate_pairs
    users = generate_users()
    pairs = generate_pairs(users)
"""

import numpy as np
import pandas as pd
from itertools import combinations

from config import (
    RANDOM_SEED, NUM_USERS, NUM_PAIRS,
    DOMAINS, GOALS, COMM_STYLES, LOCATIONS,
    DOMAIN_GROUPS, GOAL_COMPAT, COMM_COMPAT,
    FEATURE_WEIGHTS, FEATURE_NAMES, MATCH_THRESHOLD,
    DATA_DIR,
)

# User generation

def generate_users(n: int = NUM_USERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate n synthetic event attendee profiles.

    Returns
    -------
    pd.DataFrame with columns:
        user_id, domain, goal, experience, comm_style, location
    """
    np.random.seed(seed)
    df = pd.DataFrame({
        "user_id"   : range(n),
        "domain"    : np.random.choice(DOMAINS, n),
        "goal"      : np.random.choice(GOALS, n),
        "experience": np.random.randint(0, 16, n),
        "comm_style": np.random.choice(COMM_STYLES, n),
        "location"  : np.random.choice(LOCATIONS, n),
    })
    return df

# Pairwise feature engineering

def _domain_similarity(d1: str, d2: str) -> float:
    return 1.0 if DOMAIN_GROUPS[d1] == DOMAIN_GROUPS[d2] else 0.0

def _goal_similarity(g1: str, g2: str) -> float:
    return 1.0 if g2 in GOAL_COMPAT.get(g1, []) else 0.0

def _experience_similarity(e1: int, e2: int) -> float:
    return max(0.0, 1.0 - abs(e1 - e2) / 15.0)

def _comm_similarity(c1: str, c2: str) -> float:
    return 1.0 if c2 in COMM_COMPAT.get(c1, []) else 0.0

def _location_similarity(l1: str, l2: str) -> float:
    if l1 == l2:
        return 1.0
    if "Remote" in [l1, l2]:
        return 0.7
    return 0.3


def make_pair_features(u1: pd.Series, u2: pd.Series,
                       noise_std: float = 0.08) -> dict:
    """
    Compute a feature vector and compatibility label for a pair.

    Parameters:
    u1, u2      : rows from the users DataFrame
    noise_std   : Gaussian noise added to the raw score before labelling,
                  simulating imperfect human ground-truth annotation

    Returns
    -------
    dict with keys: domain_sim, goal_sim, exp_sim, comm_sim, loc_sim, label
    """
    feats = {
        "domain_sim": _domain_similarity(u1.domain, u2.domain),
        "goal_sim"  : _goal_similarity(u1.goal, u2.goal),
        "exp_sim"   : _experience_similarity(u1.experience, u2.experience),
        "comm_sim"  : _comm_similarity(u1.comm_style, u2.comm_style),
        "loc_sim"   : _location_similarity(u1.location, u2.location),
    }

    raw_score = sum(FEATURE_WEIGHTS[f] * feats[f] for f in FEATURE_NAMES)
    noisy     = raw_score + np.random.normal(0, noise_std)
    feats["label"] = int(noisy >= MATCH_THRESHOLD)
    return feats


def generate_pairs(users: pd.DataFrame,
                   n_pairs: int = NUM_PAIRS,
                   seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Sample n_pairs random user pairs and compute features for each.

    Returns
    -------
    pd.DataFrame with feature columns + 'label'
    """
    np.random.seed(seed)
    all_pairs = list(combinations(range(len(users)), 2))
    np.random.shuffle(all_pairs)
    sampled   = all_pairs[:n_pairs]

    rows = [make_pair_features(users.iloc[i], users.iloc[j])
            for i, j in sampled]

    df = pd.DataFrame(rows)
    return df

# Save / Load helpers

def save_datasets(users: pd.DataFrame, pairs: pd.DataFrame) -> None:
    users_path = f"{DATA_DIR}/users.csv"
    pairs_path = f"{DATA_DIR}/pairs.csv"
    users.to_csv(users_path, index=False)
    pairs.to_csv(pairs_path, index=False)
    print(f"[data_generator] Saved → {users_path}")
    print(f"[data_generator] Saved → {pairs_path}")

def load_datasets():
    users = pd.read_csv(f"{DATA_DIR}/users.csv")
    pairs = pd.read_csv(f"{DATA_DIR}/pairs.csv")
    return users, pairs


# Quick test

if __name__ == "__main__":
    users = generate_users()
    pairs = generate_pairs(users)
    save_datasets(users, pairs)

    print(f"\nUsers : {users.shape}  |  Sample:")
    print(users.head(5).to_string(index=False))
    print(f"\nPairs : {pairs.shape}  |  Match rate: {pairs['label'].mean()*100:.1f}%")
    print(pairs.head(5).to_string(index=False))
