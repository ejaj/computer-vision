import numpy as np


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exps = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exps / np.sum(exps, axis=1, keepdims=True)


def cross_entropy_loss(y_true, y_pred):
    return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))


# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 4)  # 100 samples, 4 features
y = np.eye(3)[np.random.choice(3, 100)]  # 3 classes, one-hot encoding

# Network parameters
input_dim = 4
hidden_dim = 5
output_dim = 3
learning_rate = 0.01

# Initialize weights and biases
W1 = np.random.randn(input_dim, hidden_dim) * 0.01
b1 = np.zeros((1, hidden_dim))
W2 = np.random.randn(hidden_dim, output_dim) * 0.01
b2 = np.zeros((1, output_dim))

# Training loop
epochs = 1000
for epoch in range(epochs):
    # Forward pass
    z1 = np.dot(X, W1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = softmax(z2)

    # Compute loss
    loss = cross_entropy_loss(y, y_pred)

    # Backward pass
    dz2 = y_pred - y  # Gradient of loss w.r.t. z2
    dW2 = np.dot(a1.T, dz2) / X.shape[0]
    db2 = np.sum(dz2, axis=0, keepdims=True) / X.shape[0]

    da1 = np.dot(dz2, W2.T)
    dz1 = da1 * relu_derivative(z1)  # Gradient of loss w.r.t. z1
    dW1 = np.dot(X.T, dz1) / X.shape[0]
    db1 = np.sum(dz1, axis=0, keepdims=True) / X.shape[0]

    # Weight updates
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Testing the model
test_input = np.random.rand(1, 4)
hidden = relu(np.dot(test_input, W1) + b1)
output = softmax(np.dot(hidden, W2) + b2)
print("Test input:", test_input)
print("Predicted output (probabilities):", output)
