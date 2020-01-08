#
# autoencoder.py
# CADL - Session III
# MINST autoencoder
#

from tensorflow.examples.tutorials.mnist import input_data
from ml_util import stats, normalise, mini_batch

import tensorflow as tf
import numpy as np
import dense_model
import conv_model 
import vae_model as model

# Download and read minst dataset
dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_images = dataset.train.images
test_images = dataset.test.images
train_labels = dataset.train.labels
test_labels = dataset.test.labels

# Define data tensors
input_shape = (None, ) + train_images.shape[1:]
inputs_op = tf.placeholder(tf.float32, input_shape)

# Build autoencoder model
#encoding_op, decoding_op = model.build(inputs_op)
encoding_op, decoding_op, mean_op, log_sd_op = model.build(inputs_op)

# Define loss
loss_op = model.build_loss(inputs_op, decoding_op, mean_op, log_sd_op)

# Define optimisation 
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
train_op = optimizer.minimize(loss_op)

# Train autoencoder
n_epochs = 2
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i_epoch in range(n_epochs):
        print("Epoch {}".format(i_epoch))
        
        # Obtain mini batches of data
        np.random.shuffle(train_images)
        batches = mini_batch(train_images, 128)
        
        # Train using data mini batch
        for batch in batches:
            training_feed = {inputs_op: batch}
            sess.run(train_op, feed_dict=training_feed)
        
            # Display training progress by printing loss
            test_feed = {inputs_op: test_images[:64]}
            loss = sess.run(loss_op, feed_dict=test_feed)
            print("loss: ", loss)
            
