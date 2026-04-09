import matplotlib.pyplot as plt

from Day_05.scripts.model_train import model
from Day_05.scripts.split_script import X, y

# Real data points
plt.scatter(X, y, color='blue', label='Actual Data')

# Regression line
plt.plot(X, model.predict(X), color='red', label='Regression Line')

# Labels + Title
plt.title("Hours vs Score: AI Prediction Line")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.legend()

plt.show()