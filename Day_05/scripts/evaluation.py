from sklearn.metrics import mean_squared_error, r2_score

from Day_05.scripts.model_train import predictions
from Day_05.scripts.split_script import y_test

#MSE :- Average squared difference between predicted and actual values
mse = mean_squared_error(y_test, predictions)
#R2 :- indicates how well the model explains the variance in the data.
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R-Squared Score: {r2:.2f}")
