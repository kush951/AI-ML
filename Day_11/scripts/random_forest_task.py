# random_forest_task.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 1. Load Data
data = fetch_california_housing()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Initialize the Forest
# n_estimators = number of trees
# max_depth = controls overfitting
rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1   # Use all CPU cores (faster training)
)

# 3. Train
rf_model.fit(X_train, y_train)

# 4. Evaluate
predictions = rf_model.predict(X_test)
r2 = r2_score(y_test, predictions)

print(f"Random Forest R2 Score: {r2:.4f}")