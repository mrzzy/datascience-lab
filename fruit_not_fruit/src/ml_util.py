#
# ml_util.py
# Machine Learning Utilities
#

#import matplotlib.pyplot as plt
import numpy as np
import random
import math

# Compute mean and standard deviation for the given set of features
# Returns mean, deviation
def stats(features):
    features = np.asarray(features)
    data_pool = features.ravel()
    mean = sum(data_pool) / len(data_pool)
    variance = sum((data_pool - mean) ** 2.0) / len(data_pool)
    deviation = variance ** 0.5

    return mean, deviation

# Normalise the features by applying 
# (x - mean) / dev where mean and dev are in stats
# Returns the normalised features
def normalise(features, stats):
    features = np.asarray(features)
    mean, deviation = stats 
    norm_features = [ (vector - mean) / deviation for vector in features ]
    return np.asarray(norm_features)

# Test Train Spliter
# Randomly Splits input and output data into test and train data in ratio 7:3
# input and output length must be the same.
# Returns (train_ins, train_outs, test_ins, tests_outs)
def split_test_train(inputs, outputs, ratio=0.7):
    if not len(inputs) == len(outputs): 
        raise ValueError("Input and outs length are different")
    
    border = math.floor(ratio * len(inputs))
    # Demarking a 80% slice of data for training
    train_ins = inputs[:border] # slice from start to (border - 1)
    train_outs = outputs[:border] # slice from start to (border - 1)
    
    test_ins = inputs[border:] # slice from border to end
    test_outs = outputs[border:] # slice from border to end

    return (train_ins, train_outs, test_ins, test_outs)

# Shuffle datasets inputs and outputs.
# Inputs and outputs would be shuffled in the same order, so any ordering relation 
# between coresponding inputs and outputs will be preserved.
# input and output length must be the same.
# Returns the shuffled inputs and outputs
def shuffle(inputs, outputs):
    if not len(inputs) == len(outputs): 
        raise ValueError("Input and outs length are different")

    n_data = len(inputs)

    # Conduct shuffle on copy
    shuffle_ins = inputs[:]
    shuffle_outs = outputs[:]

    # Fisher-Yates shuffle
    for i in range(n_data - 1): # 1 to n - 2
        j = random.randint(i, n_data - 1) # random j that satisfies i <= j <= (n - 1)
        # Swap jth element with ith element to place jth element in randomized
        # subsection of array
        shuffle_ins[i], shuffle_ins[j] = shuffle_ins[j], shuffle_ins[i] 
        shuffle_outs[i], shuffle_outs[j] = shuffle_outs[j], shuffle_outs[i] 
    
    return shuffle_ins, shuffle_outs
