#
# model.py
# Fruit not Fruit Model
#

from keras import layers
from keras import models
from keras import optimizers
from ml_util import *
import pickle
import numpy as np

# Build and return DNN to classify fruit or not fruit
# Input shape given by function
# Output 1 fruit 0 not fruit
def build_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3,3), activation="relu", input_shape=input_shape))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(64, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Conv2D(128, (3,3), activation="relu"))
    model.add(layers.MaxPooling2D((2,2)))

    model.add(layers.Flatten(input_shape=input_shape))
    
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    
    model.compile(optimizer="adam",
                  loss="binary_crossentropy", 
                  metrics=["acc"])
    return model


if __name__ == "__main__":
    print("Preparing data...")
    with open("./data/dataset.pickle", "rb") as f:
        labels, features = pickle.load(f)
    
    # Truncate dataset to specified size
    n_size = len(features)
    labels = labels[:n_size]
    features = features[:n_size]
    
    # Shuffle dataset to evenly distribute data
    features, labels = shuffle(features, labels)

    train_ins, train_outs, test_ins, test_outs = split_test_train(features, labels, 0.8)
    train_ins, train_outs, valid_ins, valid_outs = split_test_train(train_ins, train_outs, 0.8)
    
    # Normalise data to make data more suitable for NN training
    feature_stats = stats(train_ins)
    norm_train_ins = normalise(train_ins, feature_stats)
    norm_test_ins = normalise(test_ins, feature_stats)
    norm_valid_ins = normalise(valid_ins, feature_stats)
    
    # Train model for data
    print("Training model ...")
    model = build_model((32, 32, 3))
    model.summary()
    history = model.fit(norm_train_ins, train_outs,
                batch_size=128,
                epochs=2,
                validation_data=(norm_valid_ins, valid_outs))

    # Test model with test data
    print("Testing model...")
    test_loss, test_acc = model.evaluate(norm_test_ins, test_outs)
    print("Loss:{:.4f} Accruacy:{:.2f}%".format(test_loss, test_acc))
    
    # Save data for model evalutation
    pickle.dump(history.history, open("history.pickle","wb"))
    model.save("model.h5")
