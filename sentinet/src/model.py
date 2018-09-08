#
# src/model.py
# Build and train Sentinet Model
#

from keras import layers, models, regularizers

from dataset import load_dataset
from ml_util import shuffle, split_test_train, stats, normalise
from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess data
features, labels = load_dataset()

# One hot encode labels
labels = to_categorical(labels)

# Randomly split into test train validation sets
features, labels = shuffle(features, labels)
train_ins, train_outs, test_ins, test_outs = split_test_train(features, labels, 0.8)

# Build Sentinet model for the given input shape
def build_model(in_shape):
    model = models.Sequential()
    # Extract features using CNN 
    model.add(layers.Conv2D(16, (3, 3), activation="relu", input_shape=in_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    # Classify using MLP
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(0.1))
    
    # Output layer: 8 possible classes
    model.add(layers.Dense(8, activation="softmax"))

    model.compile(optimizer="adam",
                  loss="categorical_crossentropy",
                  metrics=["acc"])
    return model

print(features.shape[1:])
model = build_model(features.shape[1:])
model.summary()

# Train the model
n_epochs = 100
batch_len = 128
history = model.fit(train_ins, train_outs,
          epochs=n_epochs,
          batch_size=batch_len,
          validation_data=(test_ins, test_outs))

# Save the model to disk
model.save("model.h5")

# Evaluate model
epochs = range(n_epochs)
plt.plot(epochs, history.history["loss"], "b-", label="Loss")
plt.plot(epochs, history.history["val_loss"], "r-", label="Validation Loss")
plt.show()
plt.plot(epochs, history.history["acc"], "b-", label="Accuracy")
plt.plot(epochs, history.history["val_acc"], "r-", label="Accuracy")
plt.show()
