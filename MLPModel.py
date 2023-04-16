import numpy as np
import pandas as pd
from tqdm import tqdm
from MNIST_Dataloader import MNIST_Dataloader
from sklearn.metrics import accuracy_score

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -250, 250)))

def sigmoid_derivative(x):
    return x * (1 - x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

# Load Data
dataloader = MNIST_Dataloader()
train_data, train_labels = dataloader.get_train_data()
train_data=train_data.reshape((-1,28*28))
test_data, test_labels = dataloader.get_test_data()
test_data=test_data.reshape((-1,28*28))

# Initialize the weights randomly with mean 0 and small standard deviation
input_weights = np.random.normal(0, 0.1, (784, 32))
output_weights = np.random.normal(0, 0.1, (32, 10))

learning_rate = 0.1
num_epochs = 250
batch_size = 32


for epoch in tqdm(range(num_epochs)):
    # Shuffle the training data
    permutation = np.random.permutation(len(train_data))
    train_data = train_data[permutation]
    train_labels = train_labels[permutation]
    
    # Split the training data into batches
    num_batches = len(train_data) // batch_size
    for i in range(num_batches):
        batch_data = train_data[i*batch_size:(i+1)*batch_size]
        batch_labels = train_labels[i*batch_size:(i+1)*batch_size]
        batch_labels = np.eye(10)[batch_labels]
        
        # Forward Pass
        hidden_layer = sigmoid(np.dot(batch_data, input_weights))
        output_layer = softmax(np.dot(hidden_layer, output_weights))

        # Calculate the error 
        error = batch_labels - output_layer
       
        # Backpropogation
        hidden_derivative = np.dot(error, output_weights.T) * sigmoid_derivative(hidden_layer)
        output_weights += learning_rate * np.dot(hidden_layer.T, error)
        input_weights += learning_rate * np.dot(batch_data.T, hidden_derivative)
    
# Test the model
hidden_layer = sigmoid(np.dot(test_data, input_weights))
output_layer = softmax(np.dot(hidden_layer, output_weights))
predictions = np.argmax(output_layer, axis=1)

val_acc=(predictions==test_labels).mean()
print('\nAccuracy: {:.2f}\n'.format(accuracy_score(predictions, test_labels)))