import matplotlib.pyplot as plt
import pandas as pd

from Day_11.scripts.random_forest_task import rf_model, data

# Get importance scores
importances = rf_model.feature_importances_
feature_names = data.feature_names
# Create a Series for easy plotting
feat_importances = pd.Series(importances, index=feature_names)
feat_importances.nlargest(5).plot(kind='barh')
plt.title("Top 5 Most Important Features")
plt.show()
