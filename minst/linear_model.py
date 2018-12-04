#
# linear_model.py
# Linear Model with Tensorflow
#

import tensorflow as tf

# Build and return a linear model computation graph given placeholder input x
# and expected output y
# Returns train_op and predict_op for the linear model
def build(x, y):
    # Perform linear transform
    # z = W * x + b
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    z = tf.matmul(x, W) + b;

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
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
    train_op = optimizer.minimize(cross_entropy)
    
    return train_op, predict_op
        


