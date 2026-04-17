# Section 5. Feature Importance
import pandas as pd

from Day_11.scripts.random_forest_task import rf_model, data

feature_importances = rf_model.feature_importances_
feature_names = data.feature_names

# Create DataFrame for better visualization
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": feature_importances
})

# Sort by importance
importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\nFeature Importance:\n")
print(importance_df)