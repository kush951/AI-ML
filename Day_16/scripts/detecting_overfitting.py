import numpy as np


from Day_16.scripts.model_validation import model, X, y, scores

model.fit(X, y)

train_score = model.score(X, y)

print(f"Training Accuracy: {train_score:.4f}")
print(f"Validation Accuracy: {np.mean(scores):.4f}")