import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from Day_12.scripts.classification_task import y_test, y_pred

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Malignant vs Benign')
plt.show()