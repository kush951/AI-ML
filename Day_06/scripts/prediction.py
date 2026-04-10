# Manual Input Prediction
from Day_06.scripts.classification_task import clf

result = clf.predict([[3, 7], [8, 1]])

print("Predictions:", result)

for i, r in enumerate(result):
    if r == 1:
        print(f"Input {i+1}: PASS")
    else:
        print(f"Input {i+1}: FAIL")