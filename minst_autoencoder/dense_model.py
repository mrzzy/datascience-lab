#
# dense_model.py
# Dense Autoencoder
#

import tensorflow as tf
import numpy as np

from layers import layer_dense, layer_decode_dense

# Build dense minst autoencoder given the minst iemages inputs tensor
# Returns tensors for the encoding and decoded_inputs
def build(inputs):
    with tf.variable_scope("autoencoder"):
        # Flatten inputs
        x = tf.layers.Flatten()(inputs)
        
        # Encode using dense layers
        a = layer_dense(x, 512, tf.nn.relu, "dense_1")
        a = layer_dense(a, 256, tf.nn.relu, "dense_2")
        a = layer_dense(a, 128, tf.nn.relu, "dense_3")
        encoding = layer_dense(a, 64, tf.nn.relu, "dense_4")
        
        # Decode using transposed dense layers
        a = layer_decode_dense(encoding, tf.nn.relu, "dense_4")
        a = layer_decode_dense(a, tf.nn.relu, "dense_3")
        a = layer_decode_dense(a, tf.nn.relu, "dense_2")
        decoded_inputs = layer_decode_dense(a, tf.nn.relu, "dense_1")
    
        return encoding, decoded_inputs

def build_loss(inputs, decoded_inputs):
    inputs = tf.reshape(inputs, (-1, 712))
    return tf.losses.mean_squared_error(inputs, decoded_inputs)
