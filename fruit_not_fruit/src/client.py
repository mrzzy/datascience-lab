#
# src/client.py
# Client API
#

import cv2
import numpy as np
import pickle
from keras import models
from multiprocessing import Pool, cpu_count
from ml_util import *

# Label constants
FRUIT = 1
NOT_FRUIT = 0

# Predict whether the given image features vector of shape (32 * 32 * 3)
# Returns a FRUIT if predicted image is fruit else NOT_FRUIT
fruit_model = models.load_model("models/model_1.h5")
stats = pickle.load(open("stats.pickle","rb"))
def predict_features(vector):
    assert vector.shape == (32, 32, 3)
    
    # Predict is fruit probablity with using fruit model 
    features = np.asarray([vector])
    norm_features = normalise(features, stats)
    probablity = fruit_model.predict(norm_features)[0]
    
    return FRUIT if probablity >= 0.5 else NOT_FRUIT

# Predict whether the given image contains a fruit
# Slides a 32 by 32 window accross the given image and computes
# predictions for each of them, if one of them contains a fruit,
# Returns FRUIT, else returns not FRUIT
def predict_image(image):
    height, width, n_channel = image.shape
    assert height >= 32 and width >= 32 and n_channel == 3

    # Slide 32 by 32 window accross image
    n_rows = height // 32
    n_cols = width // 32
    
    def generate_locations():
        for i in range(n_rows):
            for j in range(n_cols):
                yield i, j
    locations = [ loc for loc in generate_locations() ]

    def generate_window(i_row, i_col):
        # Compute begin and end corrdinates for window
        begin_row = i_row * 32
        begin_col = i_col * 32
        end_row = begin_row + 32
        end_col = begin_col + 32

        window = image[begin_row:end_row, begin_col:end_col, :]
        
        return window
    windows = np.asarray(list(map((lambda loc: generate_window(loc[0], loc[1])), locations)))
    norm_features = normalise(windows, stats)
    
    # Compute predictions for windows
    probablities = fruit_model.predict(norm_features)
    predictions = [ 1 if p > 0.5 else 0 for p in probablities ]
    n_predicts = sum(predictions)
    
    print("No. of predictions: ", n_predicts)
    return FRUIT if n_predicts >= 1 else NOT_FRUIT

# Small opencv based client to test the fruit model by using images
# from the device's camera
if __name__ == "__main__":
    print("Press q to quit...")
    cap = cv2.VideoCapture(0)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Convert frame to rgb from bgr
        rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ## downsample features to 32, 32, 3 for prediction
        #features = cv2.resize(rgb_img, (32,32))
        features = cv2.resize(rgb_img, (600,400))
        
        # Make prediction using fruit model
        prediction = predict_image(features)
        
        # Display prediction by rendering prediction over the model
        pred_text = "is that fruit i see.." if prediction == FRUIT else "nothing to see here..."

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, pred_text, (10,500), font, 3, (255,255,255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
