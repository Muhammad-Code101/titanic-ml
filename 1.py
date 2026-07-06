from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

np.random.seed(42)
sizes = np.random.randint(500, 3000, 100).reshape(-1, 1)
prices = sizes * 0.25 + np.random.randint(20, 80, 100).reshape(-1, 1) + 30

X_train, X_test, y_train, y_test = train_test_split(sizes, prices, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict(X_test)

mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

print(f"R² Score: {r2:.4f}")
print(f"MSE:      {mse:.2f}")