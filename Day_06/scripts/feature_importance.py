# SECTION 3: Feature Importance
from Day_06.scripts.classification_task import clf, X

importance = clf.coef_[0]

print("\nSECTION 3 OUTPUT: Feature Importance")

for i, v in enumerate(importance):
    print(f'Feature: {X.columns[i]}, Score: {v:.4f}')

