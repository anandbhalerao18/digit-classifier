# This file trains a small CNN on the MNIST dataset and saves it.

import tensorflow as tf                      
from tensorflow.keras import layers, models  

#load the MNIST dataset
# MNIST comes built-in with keras. It downloads automatically the first time.
# x = images (28x28 pixels), y = labels (the digit 0-9)
# 60,000 images are for training and 10,000 for testing
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Training images shape:", x_train.shape)  # (60000, 28, 28)
print("Test images shape:", x_test.shape)       # (10000, 28, 28)

#preprocess the data
# Pixels are numbers from 0 to 255. We divide by 255 so they become 0 to 1.
# Small numbers help the neural network learn faster and better...
x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN layers expect 4 dimensions (number of images, height, width, channels)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

#build the model
model = models.Sequential([
    # first conv layer: 32 filters of size 3x3 find simple patterns (edges, lines)
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    # max pooling makes the image smaller (keeps the strongest values)
    layers.MaxPooling2D((2, 2)),

    # second conv layer: 64 filters find more complex patterns (curves, loops)
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # flatten turns the 2D feature maps into one long 1D list of numbers
    layers.Flatten(),

    # dense (fully connected) layer to combine all the features
    layers.Dense(64, activation="relu"),

    # output layer: 10 neurons (one for each digit)
    # softmax converts the outputs into probabilities that add up to 1
    layers.Dense(10, activation="softmax"),
])

#compile the model 
# optimizer = how the model updates its weights (adam is a good default)
# loss = how wrong the model is (sparse_categorical_crossentropy is used when labels are integers like 0-9)
# metrics = what we want to see while training
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()  # prints the layers of the model

# train the model
# epochs = how many times the model sees the full training data
# validation_split = 10% of training data is kept aside to check progress
model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.1)

#test the model on data it has never seen
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

#save the trained model to a file
# The API will load this file later, so we don't need to train again
model.save("digit_model.keras")
print("Model saved as digit_model.keras")
