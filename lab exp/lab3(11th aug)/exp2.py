# 1. Import libraries
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 2. Create synthetic regression data
# Let's try to predict y = 3x + noise
torch.manual_seed(42)  # for reproducibilityA
X = torch.rand(100, 1) * 10   # 100 samples, values between 0 and 10
y = 3 * X + torch.randn(100, 1) * 2  # Add some Gaussian noise

# 3. Define the neural network
class RegressionNN(nn.Module):
    def __init__(self):
        super(RegressionNN, self).__init__()
        self.hidden1 = nn.Linear(1,  64)  # Input layer → 64 neurons
        self.hidden2 = nn.Linear(64, 32) # 64 → 32 neurons
        self.output = nn.Linear(32, 1)   # Output layer → 1 neuron (regression)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)  # No activation for regression output
        return x

# 4. Initialize model, loss, optimizer
model = RegressionNN()
criterion = nn.MSELoss()           # Mean Squared Error for regression
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 5. Train the model
epochs = 500
for epoch in range(epochs):
    model.train()

    # Forward pass
    predictions = model(X)
    loss = criterion(predictions, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# 6. Test / visualize results
model.eval()
with torch.no_grad():
    predicted = model(X)

plt.scatter(X, y, label="Actual Data")
plt.plot(X, predicted, color='red', label="Predicted Line")
plt.legend()
plt.show()

