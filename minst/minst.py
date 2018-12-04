#
# minst.py
# Tensorflow Minist Classifier
#

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

import linear_model
import deep_model

# Download and read minst dataset
dataset = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_images = dataset.train.images
test_images = dataset.test.images
train_labels = dataset.train.labels
test_labels = dataset.test.labels

# Setup input(x) and actual output placeholders
x = tf.placeholder(tf.float32, (None, 784)) # input
y = tf.placeholder(tf.float32, (None, 10)) # actucal


# Build model
model = deep_model.build(x, y)
train_op, predict_op = model

# Build and a graph to compute accuracy
def build_accuracy_metric(y, predict_op):
    actual_op = tf.argmax(y, 1)
    correct_map = tf.equal(actual_op, predict_op)
    accuracy_op = tf.reduce_mean(tf.cast(correct_map, tf.float32))
    return accuracy_op

accuracy_op = build_accuracy_metric(y, predict_op)

# Execute training in session
with tf.Session() as sess:
    # init op to init W and b variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # Train the model
    for i in range(1000):
        batch_xs, batch_ys = dataset.train.next_batch(100)
        sess.run(train_op, feed_dict={x: batch_xs, y: batch_ys})

        # Train accuracy 
        accuracy = sess.run(accuracy_op, feed_dict={x: batch_xs, y: batch_ys})
        print("Training... {:.2f}% Accuracy".format(accuracy * 100))
    

    # Test the model
    accuracy = sess.run(accuracy_op, feed_dict={x: test_images, y: test_labels})
    print("Test: {:.2f}% Accuracy".format(accuracy * 100))
