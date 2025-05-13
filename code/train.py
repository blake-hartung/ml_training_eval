# train.py

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.metrics import classification_report

# Define hyperparameters using argparse
# SageMaker passes hyperparameters as command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
parser.add_argument('--batch-size', type=int, default=64, help='Batch size for training.')
parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate for the optimizer.')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use (e.g., "adam", "sgd").')
parser.add_argument('--filters-conv1', type=int, default=32, help='Number of filters in the first convolutional layer.')
parser.add_argument('--filters-conv2', type=int, default=64, help='Number of filters in the second convolutional layer.')
parser.add_argument('--dense-units', type=int, default=128, help='Number of units in the dense layer.')
parser.add_argument('--dropout-rate', type=float, default=0.5, help='Dropout rate after the dense layer.')

# SageMaker specific parameters
# These are automatically set by the SageMaker environment
parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

args = parser.parse_args()

# --- Data Loading and Preprocessing ---
print("Loading Fashion MNIST data...")
# In a real SageMaker job, you would typically load data from the SM_CHANNEL_TRAIN/TEST paths
# For this example using fashion_mnist built into Keras, we load directly.
# If using your own dataset, you would load from args.train and args.test
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Add a channel dimension for CNNs (grayscale images)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255

# Convert class vectors to binary class matrices (one-hot encoding)
num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# --- Model Definition ---
print("Defining Keras model...")
model = Sequential([
    Conv2D(args.filters_conv1, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(args.filters_conv2, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(args.dense_units, activation='relu'),
    Dropout(args.dropout_rate),
    Dense(num_classes, activation='softmax')
])

# Select optimizer based on hyperparameter
if args.optimizer.lower() == 'adam':
    optimizer = Adam(learning_rate=args.learning_rate)
elif args.optimizer.lower() == 'sgd':
    optimizer = SGD(learning_rate=args.learning_rate)
else:
    print(f"Warning: Unknown optimizer '{args.optimizer}'. Using Adam.")
    optimizer = Adam(learning_rate=args.learning_rate)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

# --- Training ---
print("Starting training...")
model.fit(x_train, y_train,
          batch_size=args.batch_size,
          epochs=args.epochs,
          verbose=2, # Set to 1 for detailed progress, 2 for one line per epoch
          validation_data=(x_test, y_test))

print("Training finished.")

# --- Evaluation ---
print("Starting evaluation...")
# Evaluate the model on the test data
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Generate classification report for more detailed metrics
y_pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=[str(i) for i in range(num_classes)]))


# --- Save Model ---
# Get the local model directory from the environment variable directly
# SageMaker sets this to '/opt/ml/model' inside the container
local_model_dir = os.environ.get('SM_MODEL_DIR')

# Check if SM_MODEL_DIR is set (it should be in a SageMaker job)
if not local_model_dir:
    # Fallback for local testing outside of SageMaker if needed
    local_model_dir = './local_model_output'
    print(f"Warning: SM_MODEL_DIR environment variable not found. Using fallback path: {local_model_dir}")
    # Create the directory if it doesn't exist
    os.makedirs(local_model_dir, exist_ok=True)

print(f"Saving model to local path: {local_model_dir}")

# Define the full local path including the filename and desired extension
# Use the .keras extension as recommended by Keras 3
model_save_path = os.path.join(local_model_dir, 'model.keras')

# Save the model using the updated method for Keras 3
# Remove save_format argument and use the .keras extension
model.save(model_save_path)

print("Model saving complete.")