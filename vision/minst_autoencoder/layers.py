#
# layers.py
# Neural network layers
#

import tensorflow as tf

# Add a dense layer to the given tensor x with the given number of neurons
# using the given activation function
# Return layer activations a of the dense layer
def layer_dense(x, n_neurons, activation, name):
    with tf.variable_scope(name):
        # Setup weights and biases
        input_dim = x.get_shape()[-1].value
        W = tf.get_variable("weight", shape=(input_dim, n_neurons), dtype=tf.float32, 
                            initializer=tf.glorot_normal_initializer())
        b = tf.get_variable("bias", shape=(n_neurons,), dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0))
                            
        # Perform neuron operation to obtain activations
        # z = x * W + b
        z = tf.matmul(x, W) + b
        # a = A(z)
        a = activation(z)
        
        return a

# Add a decoder layer for the dense layer given by name, 
# using the given activation function.
# Returns the deocoder layers actviations
def layer_decode_dense(x, activation, name):
    with tf.variable_scope(name, reuse=True):
        # Retrieve weights and biases for target dense layer
        W = tf.get_variable("weight")
        b = tf.get_variable("bias")
        
        # Inverse dense transformation
        # z = (x - b) * W'
        W_t = tf.transpose(W)
        z = tf.matmul(x - b, W_t)
        # a = A(z)
        a = activation(z)
        
        return a


# Add a convolution 2d layer to the given tensor x using the n_filters filters 
#  the given filter shape,
# using the given activation function on the outputs of convolution
# Returns the activations of the layer
def layer_conv2d(x, n_filter, filter_shape, activation, name, stride=(2, 2)):
    with tf.variable_scope(name):
        # Setup filter maps
        n_input_channels =  x.get_shape()[-1].value
        filters = tf.get_variable("filters",
                                  shape=filter_shape + (n_input_channels, n_filter),
                                  initializer=tf.random_normal_initializer(mean=0.0,
                                                                          stddev=0.02))
        # Perform convolution
        z = tf.nn.conv2d(x, filters, (1,) + stride + (1,), padding="SAME")
        # Apply activation
        a = activation(z)
    
        return a

# Add a deconvolutional layer that inverses the convolution done by the the
# convolution layer specified by name
# using the given activation function on the outputs of deconvolution
# Returns the activations of the layer
def layer_deconv2d(x, output_shape, activation, name, stride=(2,2)):
    with tf.variable_scope(name, reuse=True):
        # Setup filter maps
        n_input_channels =  x.get_shape()[-1].value
        filters = tf.get_variable("filters")

        # Perform deconvolution
        out_shape = tf.stack([tf.shape(x)[0], 
                              output_shape[1], output_shape[2], output_shape[3]])
        z = tf.nn.conv2d_transpose(x, filters, out_shape, (1,) + stride + (1,), 
                                  padding="SAME")
        # Apply activation
        a = activation(z)
    
        return a

