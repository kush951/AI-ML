from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_digits
import numpy as np

# 1. Load Data (Handwritten Digits Dataset)
digits = load_digits()
X, y = digits.data, digits.target

# 2. Define Model
model = RandomForestClassifier(n_estimators=50, random_state=42)

# 3. Apply 5-Fold Cross Validation
scores = cross_val_score(model, X, y, cv=5)

# 4. Output Results
print(f"Scores for each fold: {scores}")
print(f"Mean Accuracy: {np.mean(scores):.4f}")
print(f"Standard Deviation (Stability): {np.std(scores):.4f}")