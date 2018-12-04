#
# deep_model.py
# Deep Model with Tensorflow
#

import tensorflow as tf

# Add Densely connected layer with input x
# Returns ouput graph y
def layer_dense(x, n_neurons):
    # Define weight matrix
    # Weight maps input shape to n_neurons output
    W_shape = (x.get_shape()[-1].value, n_neurons)

    # Initalise weights with random values from normal distribution for symmertry
    # breaking
    W = tf.Variable(tf.truncated_normal(shape=W_shape, stddev=0.1))

    # Define biases all set to 0.1
    b = tf.Variable(tf.constant(0.1, shape=[n_neurons]))

    # Forward Propergation
    y = tf.nn.relu(tf.matmul(x, W) + b)

    return y

# Build and return a deep dense NN computation graph given placeholder input x
# and expected output y
# Returns train_op and predict_op for the linear model
def build(x, y):
    # Build NN
    z = layer_dense(x, 64)
    z = layer_dense(z, 10)
    
    # Apply softmax to get probabilities for each category
    # p = softmax(z)
    p = tf.nn.softmax(z)
    # Define model prediction as the class with the highest probabilities
    predict_op = tf.argmax(p, 1)

    # Compute cross entropy loss, which just logistic loss generalised to 
    # multiple classes
    # L = -sum y * log(p) for every class and average it
    cross_entropy = tf.reduce_mean(
        -tf.reduce_sum(y*tf.log(p), 1))

    # Train with gradient desecent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.3)
    train_op = optimizer.minimize(cross_entropy)

    return train_op, predict_op
    
