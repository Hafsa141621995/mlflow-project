import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

# Fake dataset simple
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, "model.joblib")
print("Model saved ✅")
