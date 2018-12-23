#
# dataset.py
# Performs features extraction on the RAVDESS dataset
#

from glob import glob

import pickle
import os.path
import librosa
import numpy as np

from multiprocessing import Pool, cpu_count

# mapping of the labels to their indexes
LABEL_MAP = (
    ("neutral", 0),
    ("calm", 1),
    ("happy", 2),
    ("sad", 3),
    ("angry", 4),
    ("fearful", 5),
    ("disgust", 6),
    ("surprised", 7)
)
    
# Extract the sentiment label from the given dataset entry at path
# Return extracted sentiment label as a int
def extract_label(path):
    # chomp off the dirname and the wav extension
    basename = os.path.basename(path)
    identifier = basename[:basename.find(".")]
    
    # Select sentiment label
    labels = identifier.split("-")
    return int(labels[2]) - 1

# Extract the features vector from the given dataset entry at path
# Returns a numpy array representing the input featurs
def extract_feature(path, feature_len=500, sample_rate=16000):
    waveform, sample_rate = librosa.core.load(path)
    features = librosa.feature.melspectrogram(waveform, sr=sample_rate)
    print(features.shape)

    pad_width = feature_len - features.shape[1]
    features = np.pad(features, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Add single channel to features
    features = np.reshape(features, features.shape + (1, ))
    return features

# Loads dataset 
# Returns features, labels
def load_dataset():
    with open("data/dataset.pickle", "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    pool = Pool(cpu_count())
    # Collect labels and features
    print("Collecting data...")
    paths = list(glob("data/*/*.wav"))
    labels = pool.map(extract_label, paths)
    features = pool.map(extract_feature, paths)
    print(max([f.shape[1] for f in features ]))

    # Normalise labels and features
    labels = np.asarray(labels)
    features = np.asarray(features)
    
    
    # Commit to disk
    with open("data/dataset.pickle", "wb") as f:
        pickle.dump((features, labels), f)
    print("Saved datset: {} features, {} labels".format(features.shape, labels.shape))
