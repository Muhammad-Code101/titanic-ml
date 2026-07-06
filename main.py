import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sizes = np.array([500, 700, 800, 1000, 1200, 1500, 1800, 2000]).reshape(-1, 1)
prices = np.array([150, 200, 220, 280, 320, 400, 450, 500])

model = LinearRegression()
model.fit(sizes, prices)

prediction = model.predict([[1100]])
# print(f"Predicted price for 1100 sq ft: ${prediction[0]:.2f}k")

plt.figure(figsize=(8, 5))
plt.scatter(sizes, prices, color="blue", label="Actual data")
plt.plot(sizes, model.predict(sizes), color="red", label="Model prediction")
plt.title("House size vs price")
plt.xlabel("Size (sq ft)")
plt.ylabel("Price ($k)")
plt.legend()
# plt.show()

print(f"Slope: {model.coef_[0]:.2f}")
print(f"Intercept: {model.intercept_:.2f}")