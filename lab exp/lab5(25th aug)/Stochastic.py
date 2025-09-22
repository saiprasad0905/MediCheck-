#Implementing the Stochastic Gradient Descent update rule

import numpy as np

# Generate dummy dataset
np.random.seed(42)
X = np.linspace(0, 10, 20)  # features
y = 2.5 * X + 1.0 + np.random.randn(20)  # target with noise


# SGD implementation
def stochastic_gradient_descent(X, y, lr=0.01, epochs=50):
    w, b = 0.0, 0.0  # initialize parameters
    n = len(X)

    for epoch in range(epochs):
        # Shuffle data for each epoch
        indices = np.random.permutation(n)
        X_shuffled, y_shuffled = X[indices], y[indices]

        for i in range(n):
            xi, yi = X_shuffled[i], y_shuffled[i]

            # Predictions
            y_pred = w * xi + b

            # Gradients
            dw = -2 * xi * (yi - y_pred)
            db = -2 * (yi - y_pred)

            # Update rule
            w -= lr * dw
            b -= lr * db

        # Print progress per epoch
        mse = np.mean((y - (w * X + b)) ** 2)
        print(f"Epoch {epoch + 1}: w={w:.3f}, b={b:.3f}, MSE={mse:.3f}")

    return w, b


# Run SGD
w_final, b_final = stochastic_gradient_descent(X, y, lr=0.01, epochs=20)
print(f"\nFinal Parameters: w = {w_final:.3f}, b = {b_final:.3f}")

