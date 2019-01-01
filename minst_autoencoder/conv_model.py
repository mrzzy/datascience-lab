#
# autoencoder.py
# MINST autoencoder
#

import tensorflow as tf
import numpy as np

from layers import layer_conv2d, layer_deconv2d

# Build dense minst autoencoder given the minst images inputs tensor
# Returns tensors for the encoding and decoded_inputs
def build(inputs):
    # Reshape inputs for convolution
    inputs = tf.reshape(inputs, (-1, 28, 28, 1))
    
    # Encode using convolutional layers
    shapes = []
    shapes.append(inputs.get_shape())
    a = layer_conv2d(inputs, 16, (3, 3), tf.nn.relu, "conv_1")
    shapes.append(a.get_shape())
    a = layer_conv2d(a, 16, (3, 3), tf.nn.relu, "conv_2")
    shapes.append(a.get_shape())
    encoding = layer_conv2d(a, 16, (3, 3), tf.nn.relu, "conv_3")
    
    # Decode encoding using deconvolutional layers
    a = layer_deconv2d(encoding, shapes[-1], tf.nn.relu, "conv_3")
    a = layer_deconv2d(a, shapes[-2], tf.nn.relu, "conv_2")
    decoding = layer_deconv2d(a, shapes[-3], tf.nn.relu, "conv_1")
    
    # Reshape outputs ensure model compatibility
    return encoding, decoding

def build_loss(inputs, decoded_inputs):
    inputs =  tf.reshape(inputs, (-1, 28, 28, 1))
    return tf.losses.mean_squared_error(inputs, decoded_inputs)
                
if __name__ == "__main__":
    inputs = tf.placeholder(tf.float32, (None, 28, 28))
    build(inputs)
