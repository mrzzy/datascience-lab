#
# smurge.py
# Image Painting 
# CADL- Session II
#

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
from PIL import Image

## Program Paramaeters
IMAGE_DIM = (512, 512)
HIDDEN_LAYERS = 8
HIDDEN_NEURONS = 256
EPOCHS = 20

## Data Pre Processing
# Crop the given image to a square frame of x by x
# where x is the length of the shorter side of the image
def crop_center(image):
    # Compute new dimentions for image
    # Crop a centered square from the image
    target_dim = min(image.size)
    len_x, len_y = image.size

    begin_y = (len_y // 2) - (target_dim // 2)
    end_y  = (len_y // 2) + (target_dim // 2)
    
    begin_x = (len_x // 2) - (target_dim // 2)
    end_x  = (len_x // 2) + (target_dim // 2)
    
    # Perform crop for computed dimentions
    image = image.crop((begin_x, begin_y, end_x, end_y))
    return image

# Load the image at path
# Reshapes the image for use with the stylenet model by converting it to a 
# the given dimentions dim square.
# Returns the reshaped image as np array
def load_image(path, dim):
    image = Image.open(path)
    # Center crop so we can resize without distortion
    image = crop_center(image)
    image = image.resize(dim)
    return np.array(image, dtype=np.uint8)


# Compute and return the mean and standard deviation for the given x
def compute_stats(x):
    return (np.mean(x), np.std(x))

# Perform standard scaling and mean normalisation on the given x using the given
# statistics stats
def normalise(x, stats):
    mean, std = stats
    return (x - mean) / std

# Extract image painting data from the given image matrix img_mat
def extract_dataset(img_mat):
    # Extract image coordinates as input features
    locations = np.asarray([ (x, y) for x in range(img_mat.shape[0])
                            for y in range(img_mat.shape[1]) ])
    # Extract corresponding image colors as expected outputs 
    colors = np.asarray([ img_mat[x][y] for x, y in locations ])

    return locations, colors

## Neural Net
# Add Densely connected layer with input x and activation function activation
# Returns ouput graph y
def layer_dense(x, n_neurons, activation=None):
    with tf.variable_scope("dense"):
        # Define weight matrix
        # Weight maps input shape to n_neurons output
        W_shape = (x.get_shape()[-1].value, n_neurons)

        # Initalise weights with random values from normal distribution for symmertry
        # breaking
        W = tf.Variable(tf.truncated_normal(shape=W_shape, stddev=0.1), name="weight")

        # Define biases all set to 0.1
        b = tf.Variable(tf.constant(0.1, shape=[n_neurons]), name="bias")

        # Forward Propergation
        z = tf.matmul(x, W) + b
        y = activation(z) if activation else z

        return y

# Construct smurge image painting mode for the given inputs and expected outputs
# Returns the predict, cost, train operations
def build_model(inputs, outputs):
    with tf.variable_scope("smurgenet"):
        # Add hidden layers
        a = inputs
        for i in range(HIDDEN_LAYERS):
            a = layer_dense(a, HIDDEN_NEURONS, tf.nn.relu)
            
        # Add output layer
        # Since we are predicting a three scalar values (RGB)
        # output layer would have three neurons and no activation function
        predict = layer_dense(a, 3)
        
        # Compute cost
        cost = tf.losses.mean_squared_error(outputs, predict)
        # Minimse cost by optimising parameters during training
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        train = optimizer.minimize(cost)

        return (predict, cost, train)

# Permuates random batches of the given batch size batch_size of the given dataset
# Returns a list of the generated batches.
def permute_batches(dataset, batch_size=64):
    dataset = np.asarray(dataset)
    # Randomly permuate indexes of dataset
    indexes = range(len(dataset))
    indexes = np.random.permutation(indexes)

    # Extract batches
    batches = []
    n_batches = len(indexes) // batch_size
    for i_batch in range(n_batches):
        begin, end = i_batch * batch_size, (i_batch + 1) * batch_size
        batch_indexes = indexes[begin:end]
        batch = dataset[batch_indexes]
        batches.append(batch)

    return batches

# Applying image painting for the given reference imag
# Returns the painted image
def paint_image(ref_image_mat):
    locations, colors = extract_dataset(ref_image_mat)
    
    # Normalise data
    locs_stats = compute_stats(locations)
    locations = normalise(locations, locs_stats)
    dataset = list(zip(locations, colors))
    
    # Build model
    inputs = tf.placeholder(tf.float32, (None, 2), name="inputs")
    outputs = tf.placeholder(tf.float32, (None, 3), name="expected_outputs")
    predict, cost, train = build_model(inputs, outputs)
    
    with tf.Session() as sess:
        # Train model
        sess.run(tf.global_variables_initializer())
        for i_epoch in range(EPOCHS):
            print("=" * 80)
            print("Epoch ", i_epoch)
            batches = permute_batches(dataset, batch_size=128)
            for batch in batches:
                # Unzip batch and use for training
                batch_locs, batch_colors = zip(*batch)
                feed = {inputs: batch_locs, outputs: batch_colors}
                sess.run(train, feed_dict=feed)
            
            # Print cost to output training progress
            current_cost = sess.run(cost, feed_dict=feed)
            print("Training: ", current_cost)
        
        # Paint iamge
        feed = {inputs: locations}
        painting_features = sess.run(predict, feed_dict=feed)
        painting = painting_features.reshape(IMAGE_DIM + (3,)).astype(np.uint8)
    
        return painting

if __name__ == "__main__":
    # Load reference image to paint from
    img_paths = glob.glob("data/*")
    for img_path in img_paths:
        reference_img_mat = load_image(img_path, IMAGE_DIM)
        plt.figure(1)
        plt.imshow(reference_img_mat)
        
        painting = paint_image(reference_img_mat)

        plt.figure(2)
        plt.imshow(painting)
        plt.show()
