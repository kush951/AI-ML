from sklearn.model_selection import train_test_split
import pandas as pd

# Sample Dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [35, 40, 55, 60, 68, 72, 81, 88, 92, 95]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features (Input)
X = df[['Hours']]

# Target (Output)
y = df['Score']

# Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42     # ensures same split every time
)

print(f"Training items: {len(X_train)}")
print(f"Testing items: {len(X_test)}")