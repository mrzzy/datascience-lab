#
# src/client.py
# Sentinet Model Client
#

from keras import models
from dataset import extract_feature, LABEL_MAP

import numpy as np

# Predict the sentiment of the wav file given by path 
# Returns dict of predicted name of feeling to intensity 
model = models.load_model("model.h5")
def predict_sentiment(path):
    # Extract features
    features = extract_feature(path)
    features = np.reshape(features, (1,) + features.shape)

    # Predict sentiment
    predictions = model.predict(features)[0]

    names = [ l[0] for l in LABEL_MAP ]
    feel_map = dict(zip(names, predictions))
    return feel_map

if __name__ == "__main__":
    print(predict_sentiment("/Users/zzy/Desktop/Untitled.wav"))

