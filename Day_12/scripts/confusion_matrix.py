from sklearn.metrics import confusion_matrix

from Day_12.scripts.classification_task import y_pred, y_test

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

