import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# 1. Generate "Curvy" Synthetic Data
np.random.seed(42)  # for reproducibility
X = 6 * np.random.rand(100, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(100, 1)

# 2. Transform X to include polynomial features (X²)
poly_features = PolynomialFeatures(degree=2, include_bias=False)

#NOTE:
#poly_features = PolynomialFeatures(degree=2, include_bias=True)
"""If True (default), then include a bias column, 
the feature in which all polynomial powers are zero 
(i.e. a column of ones - acts as an intercept term in a linear model)."""
X_poly = poly_features.fit_transform(X)


# 3. Fit Linear Regression to transformed data
model = LinearRegression()
model.fit(X_poly, y)

# 4. Predictions
X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = model.predict(X_new_poly)

y_pred = model.predict(X_poly)

mse = mean_squared_error(y, y_pred)
print(f"Mean Squared Error: {mse:.4f}")


# 5. Plot
plt.figure(figsize=(8,5))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_new, y_new, color='red', linewidth=2, label='Polynomial Curve')
plt.title("Polynomial Regression (Curvy Relationship)")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()