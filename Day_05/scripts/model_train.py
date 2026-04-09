from sklearn.linear_model import LinearRegression

from Day_05.scripts.split_script import X_train, y_train, X_test, y_test

# 1. Initialize the model
model = LinearRegression()

# 2. TRAIN (Learning step)
model.fit(X_train, y_train)

# 3. PREDICT (Testing step)
predictions = model.predict(X_test)

# Output
print("Predictions for Test Set:", predictions)
print("Actual Scores:", y_test.values)