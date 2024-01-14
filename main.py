import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Sigmoid function and its derivative
def sigmoid(x):
    x = np.clip(x, -500, 500)  # Clip values to avoid overflow
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)

# Neural Network Class
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.randn(self.input.shape[1], 128) / np.sqrt(self.input.shape[1])  # Better weight initialization
        self.weights2 = np.random.randn(128, 64) / np.sqrt(128)  # Additional layer
        self.weights3 = np.random.randn(64, 10) / np.sqrt(64)  # Output layer
        self.y = y
        self.output = np.zeros(self.y.shape)
        self.learning_rate = 0.25  # Smaller learning rate

    def feedforward(self):
        self.layer1 = relu(np.dot(self.input, self.weights1))
        self.layer2 = relu(np.dot(self.layer1, self.weights2))  # Second layer
        self.output = sigmoid(np.dot(self.layer2, self.weights3)) # Sigmoid only at the output layer

    def backprop(self):
        # Cross-entropy loss derivative
        error = -(self.y - self.output) / self.output.shape[0]  # Scaling the error term
        d_weights3 = np.dot(self.layer2.T, error * sigmoid_derivative(self.output))

        error = np.dot(error * sigmoid_derivative(self.output), self.weights3.T)
        d_weights2 = np.dot(self.layer1.T, error * relu_derivative(self.layer2))

        error = np.dot(error * relu_derivative(self.layer2), self.weights2.T)
        d_weights1 = np.dot(self.input.T, error * relu_derivative(self.layer1))

        self.weights1 -= self.learning_rate * d_weights1
        self.weights2 -= self.learning_rate * d_weights2
        self.weights3 -= self.learning_rate * d_weights3

# Load and preprocess the MNIST dataset (Adjust the path to your MNIST CSV file)
mnist_data = pd.read_csv('mnist_train.csv')
print(mnist_data.columns)  # Add this line to check column names

X = mnist_data.drop('label', axis=1).values / 255
y = mnist_data['label'].values

# Convert labels to one-hot encoding
y_one_hot = np.zeros((y.size, y.max()+1))
y_one_hot[np.arange(y.size), y] = 1

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

# Initialize and train the neural network
nn = NeuralNetwork(X_train, y_train)
for i in range(1000):  # Number of epochs for testing
    nn.feedforward()
    nn.backprop()
    if i % 100 == 0:  # Print loss every 100 iterations
        loss = np.mean(np.square(y_train - nn.output))
        print(f'Epoch {i}, Loss: {loss}')

# GUI Class
class DigitRecognizerGUI:
    def __init__(self, master):
        self.master = master
        master.title("Handwritten Digit Recognizer")

        self.label = tk.Label(master, text="Upload a digit image")
        self.label.pack()

        self.upload_button = tk.Button(master, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.predict_button = tk.Button(master, text="Predict", state=tk.DISABLED, command=self.predict_digit)
        self.predict_button.pack()

        self.result_label = tk.Label(master, text="Prediction will appear here")
        self.result_label.pack()

        self.neural_network = nn

    def upload_image(self):
        self.file_path = filedialog.askopenfilename()
        if self.file_path:
            print("Uploaded Image Path:", self.file_path)  # Debugging: Check file path
            self.predict_button["state"] = tk.NORMAL
            img = Image.open(self.file_path).convert('L')
            img.thumbnail((150, 150))
            photo = ImageTk.PhotoImage(img)
            if not hasattr(self, 'image_label'):
                self.image_label = tk.Label(image=photo)
                self.image_label.image = photo
                self.image_label.pack()
            else:
                self.image_label.configure(image=photo)
                self.image_label.image = photo

    def preprocess_image(self, filepath):
        try:
            # Open the image file
            with open(filepath, 'rb') as f:
                img = Image.open(f).convert('L')
                img = img.resize((28, 28))
                img_array = np.array(img)
                img_normalized = img_array / 255.0
                img_flattened = img_normalized.flatten()
                return img_flattened
        except IOError as e:
            print("Error in opening the image file:", e)
            return None

    def predict_digit(self):
        preprocessed_image = self.preprocess_image(self.file_path)
        if preprocessed_image is not None:
            # Debugging: Check the preprocessed image
            print("Preprocessed Image Shape:", preprocessed_image.shape)

            # Ensure the input is set to the new image data
            self.neural_network.input = preprocessed_image.reshape(1, -1)

            # Debugging: Check the neural network input
            print("Neural Network Input Shape:", self.neural_network.input.shape)

            # Proceed with the prediction
            self.neural_network.feedforward()
            prediction = np.argmax(self.neural_network.output)

            # Debugging: Print the prediction
            print("Predicted Digit:", prediction)

            self.result_label.config(text=f"Predicted Digit: {prediction}")
        else:
            self.result_label.config(text="Error in processing the image.")


# Main function to run the GUI
def main():
    root = tk.Tk()
    gui = DigitRecognizerGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
