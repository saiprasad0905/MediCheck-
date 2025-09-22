#Implemening the gradient descent update rule

import numpy as np

# Function and Gradient
def cost_function(w):
    return (w - 3) ** 2   # J(w)

def gradient(w):
    return 2 * (w - 3)    # dJ/dw

# Gradient Descent
def gradient_descent(starting_w, learning_rate, epochs):
    w = starting_w
    for i in range(epochs):
        grad = gradient(w)       # compute gradient
        w = w - learning_rate * grad   # update rule
        print(f"Epoch {i+1}: w = {w:.4f}, Cost = {cost_function(w):.4f}")
    return w

# Run
final_w = gradient_descent(starting_w=0.0, learning_rate=0.1, epochs=20)
print(f"\nFinal optimized weight: {final_w:.4f}")

