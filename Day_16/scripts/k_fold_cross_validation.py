from sklearn.model_selection import KFold, cross_val_score
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load dataset
data = load_iris()
X, y = data.data, data.target

# K-Fold Cross Validation
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

# Model
model = RandomForestClassifier(random_state=42)

# Cross Validation
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Results
print(f"Accuracy for each fold: {scores}")

mean_acc = np.mean(scores)
std_dev = np.std(scores)

print(f"Mean Accuracy: {mean_acc:.4f}")
print(f"Standard Deviation: {std_dev:.4f}")