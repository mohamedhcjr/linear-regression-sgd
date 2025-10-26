import numpy as np
import pandas as pd

# ---------------------------------
# 1. Load Dataset
# ---------------------------------
data = pd.read_csv("study_hours_vs_scores.csv")

# Convert columns to float
X = data["Hours"].astype(float).values.reshape(-1, 1)
y = data["Scores"].astype(float).values.reshape(-1, 1)

# Add bias term (x0 = 1)
X_b = np.c_[np.ones((len(X), 1)), X]

# ---------------------------------
# 2. Initialize parameters
# ---------------------------------
theta = np.random.randn(2, 1)  # [bias, weight]
learning_rate = 0.01
n_epochs = 100
m = len(X_b)

# ---------------------------------
# 3. SGD Implementation
# ---------------------------------
for epoch in range(n_epochs):
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]

        # Compute gradient
        gradients = 2 * xi.T.dot(xi.dot(theta) - yi)

        # Update parameters
        theta = theta - learning_rate * gradients

print("Final parameters (theta):", theta.ravel())

# ---------------------------------
# 4. Prediction
# ---------------------------------
X_new = np.array([[2], [9]])  # hours studied
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_pred = X_new_b.dot(theta)

print("Predictions:")
for h, s in zip(X_new.ravel(), y_pred.ravel()):
    print(f"Studied {h} hours → Predicted Score: {s:.2f}")
