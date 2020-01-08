#
# vae.py
# CADL - Session III
# MINST VAE model
#

import tensorflow as tf

# Build minst varitional autoencoder given the minst iemages inputs tensor
# Returns tensors for the encoding and decoded_inputs, together with the encoded
# mean and log standard deviation
def build(inputs, n_encoding=64):
    # Reshape inputs for convolution
    inputs = tf.reshape(inputs, (-1, 28, 28, 1))
    
    # Encode using convolutional layers
    a = tf.layers.conv2d(inputs, filters=16, kernel_size=(3, 3), strides=(2, 2), 
                         padding='same', activation=tf.nn.relu)
    a = tf.layers.conv2d(a, filters=16, kernel_size=(3, 3), strides=(2, 2), 
                         padding='same', activation=tf.nn.relu)
    a = tf.layers.conv2d(a, filters=16, kernel_size=(3, 3), strides=(2, 2), 
                         padding='same', activation=tf.nn.relu)
    conv_shape = a.get_shape()

    a = tf.layers.flatten(a)
    
    n_flat_units = a.get_shape()[-1].value
    
    # Define varitional encoding by sampling from encoded gaussian distributions
    mean = tf.layers.dense(a, units=n_encoding)
    log_sd = tf.layers.dense(a, units=n_encoding)
    z = tf.random_normal((n_encoding,), mean=0.0, stddev=1.0)
    encoding = mean + tf.exp(log_sd) * z
    
    # Reshape encoding for deconvolution
    a = tf.layers.dense(a, units=n_flat_units)
    a = tf.reshape(a, (-1, conv_shape[1].value, conv_shape[2].value, 
                       conv_shape[3].value))

    # Decode using deconvolution layers
    a = tf.layers.conv2d_transpose(a, filters=16, kernel_size=(3, 3), strides=(2,2), 
                                   padding='same', activation=tf.nn.relu)
    a = tf.layers.conv2d_transpose(a, filters=16, kernel_size=(3, 3), strides=(2,2), 
                                   padding='same', activation=tf.nn.relu)
    a = tf.layers.conv2d_transpose(a, filters=1, kernel_size=(4, 4), strides=(2,2), 
                                   padding='same', activation=tf.nn.relu)

    # Reshape activations for output
    a = tf.layers.flatten(a)
    a = tf.layers.dense(a, units=784)
    decoding = tf.reshape(a, (-1, 28, 28, 1))

    return encoding, decoding, mean, log_sd

def build_loss(inputs, decoded_inputs, mean, log_sd):
    # Reshape args into expected shapes 
    inputs = tf.reshape(inputs, (-1, 784))
    decoded_inputs = tf.reshape(decoded_inputs, (-1, 784))
        
    reconstruction_losses = tf.reduce_sum(tf.squared_difference(inputs, decoded_inputs), axis=1)
    divergence_losses = -0.5 * tf.reduce_sum(1.0 + log_sd - tf.square(mean) - tf.square(tf.exp(log_sd)), axis=1)
    divergence_losses = 0.5 * tf.reduce_sum(tf.square(tf.exp(log_sd)) + tf.square(mean) - log_sd - 1.0, axis=1)

    loss = tf.reduce_mean(reconstruction_losses + divergence_losses)

    return loss


if __name__ == "__main__":
    inputs = tf.placeholder(tf.float32, (None, 784))
    encoding_op, decoding_op, mean_op, log_sd_op = build(inputs)
    loss_op = build_loss(inputs, decoding_op, mean_op, log_sd_op)
