# Import necessary libraries
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical

# 1. Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. Normalize the data (Pixel values: 0-255 → 0-1)
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. One-hot encode the labels (for 10 classes: 0-9)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 4. Build the MLP model
model = Sequential([
    Flatten(input_shape=(28, 28)),        # Input layer (flatten 28x28 to 784)
    Dense(128, activation='relu'),        # Hidden layer 1 with ReLU
    Dense(64, activation='relu'),         # Hidden layer 2 with ReLU
    Dense(10, activation='softmax')       # Output layer with Softmax (10 classes)
])

# 5. Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 6. Train the model
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_split=0.1)

# 7. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

