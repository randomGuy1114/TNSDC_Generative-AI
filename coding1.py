 import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

# Define the neural network class
class NeuralNetwork:
    def _init_(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size)
        self.bias_input_hidden = np.zeros((1, self.hidden_size))
        
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size)
        self.bias_hidden_output = np.zeros((1, self.output_size))
    
    # Forward pass
    def forward(self, X):
        self.hidden_output = sigmoid(np.dot(X, self.weights_input_hidden) + self.bias_input_hidden)
        self.predicted_output = sigmoid(np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_hidden_output)
        return self.predicted_output
    
    # Backpropagation
    def backward(self, X, y, learning_rate):
        error = y - self.predicted_output
        delta_output = error * sigmoid_derivative(self.predicted_output)
        
        error_hidden = delta_output.dot(self.weights_hidden_output.T)
        delta_hidden = error_hidden * sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights_hidden_output += self.hidden_output.T.dot(delta_output) * learning_rate
        self.bias_hidden_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
        
        self.weights_input_hidden += X.T.dot(delta_hidden) * learning_rate
        self.bias_input_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate
    
    # Train the neural network
    def train(self, X, y, epochs, learning_rate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Example usage
if _name_ == "_main_":
    # Generate some example data
    np.random.seed(0)
    num_samples = 1000
    X = np.random.rand(num_samples, 3)  # Features: e.g., size, number of bedrooms, location
    y = 100 * X[:, 0] + 200 * X[:, 1] + 300 * X[:, 2] + 400 + np.random.randn(num_samples) * 10  # Prices
    
    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Split data into training and testing sets
    split_ratio = 0.8
    split_index = int(num_samples * split_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Initialize and train the neural network
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = 1
    learning_rate = 0.01
    epochs = 1000
    
    model = NeuralNetwork(input_size, hidden_size, output_size)
    model.train(X_train, y_train.reshape(-1, 1), epochs, learning_rate)
    
    # Evaluate the model
    predicted_prices = model.forward(X_test)
    test_loss = np.mean(np.square(y_test.reshape(-1, 1) - predicted_prices))
    print("Test loss:", test_loss)