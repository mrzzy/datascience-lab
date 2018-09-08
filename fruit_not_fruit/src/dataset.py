#
# dataset.py
# Generates and prepares the fruit not fruit dataset
#

import pickle
import random
import glob
from PIL import Image
import numpy as np
from multiprocessing import Pool


# Sample the given no. of samples from the cifar 100 dataset 
# Only includes data for samples that are not fruits 
# Returns bootstrapped sample labels, features where data is a 3072 numpy  Pool
def convert_vector(vector):
    vector = np.reshape(vector, (3, 32, 32))
    converted_vector = np.zeros((32, 32, 3), dtype="uint8")
    for w in range(32):
        for h in range(32):
            for p in range(3):
                converted_vector[w][h][p] = vector[p][w][h]
    return converted_vector

def load_cifar(n_sample):
    if n_sample > 50000: 
        raise ValueError("Request sample larger than dataset")

    # Source data from cifar 100 dataset dataset
    data = pickle.load(open("data/cifar-100-python/train","rb"), encoding="latin1")
    meta = pickle.load(open("data/cifar-100-python/meta", "rb"))
    encoded_labels = data["fine_labels"]
    names = meta["fine_label_names"]
    labels = [names[l] for l in encoded_labels]
    features = data["data"]
    
    # Randomly sample n_sample entries in dataset
    dataset = zip(labels, features)
    dataset = random.sample(list(dataset), n_sample)
    
    # Reshape features from 3 by 32 by 32 to 32 by 32 by 3
    features = [ x[1] for x in dataset ]
    pool = Pool(processes=3)
    converted_features = pool.map(convert_vector, features)
    
    labels = [ x[0] for x in dataset ]
    
    return labels, converted_features


# Load and prepare the image at path to a flattened feature vectoor of size 3072
def prepare_image(path):
    img = Image.open(path)
    img = img.resize((32, 32))
    matrix = np.asarray(img, dtype="uint8" )
    return matrix

# Sample the given no. of samples from the cifar 100 dataset 
# Returns bootstrapped sample labels, features where data is a 3072 numpy array
def load_path(fruit_path):
    # Read label from fruit path
    path_parts = fruit_path.split("/")
    label = path_parts[2]

    # Read image feature vector at fruit path
    vector = prepare_image(fruit_path)

    # Add random background to fruit picture
    augmented_vector = np.zeros((32,32,3), dtype="uint8")
    background_color = [random.randint(180, 250) for i in range(3) ]
    for w in range(32):
        for h in range(32):
                for p in range(3):
                    if vector[w][h][p] == 255:
                        augmented_vector[w][h][p] = background_color[p]
                    else:
                        augmented_vector[w][h][p] = vector[w][h][p]
    return label, augmented_vector

def load_fruit(n_sample):
    if n_sample > 37836: 
        raise ValueError("Request sample larger than dataset")
    
    # Source data from fruit dataset of size of sample
    fruit_paths = glob.glob("data/fruits-360/Training/*/*")
    fruit_paths = random.sample(fruit_paths, n_sample)
    
    pool = Pool(processes=3)
    dataset = pool.map(load_path, fruit_paths)

    labels = [x[0] for x in dataset]
    features  = [x[1] for x in dataset]
    
    return labels, features

# Generates balanced fruit not fruit dataset of given size by aggregating data 
# from cifar 100 and fruit dataset
# Returns dataset as labels, features
def generate_dataset(n_samples):
    # No of samples per class
    n_class = n_samples // 2
    
    # Source data from datsets
    cifar_labels, cifar_features = load_cifar(n_class)
    fruit_labels, fruit_features = load_fruit(n_class)

    # Create dataset
    labels = []
    features = []
    # Collate non fruit data
    fruitlist = ["apple", "orange", "pear"]
    non_fruit_features = [x for l, x in zip(cifar_labels, cifar_features) 
                          if not l in fruitlist]
    labels.extend([0 for i in range(len(non_fruit_features)) ])
    features.extend(non_fruit_features)

    # Collate fruit data as fruit data (1 label)
    fruit_features = [x for l, x in zip(cifar_labels, cifar_features) 
                          if l in fruitlist] + fruit_features
    labels.extend([1 for i in range(len(fruit_features))])
    features.extend(fruit_features)
    
    # Randomly shuffle the dataset
    n_data = len(labels)
    for i in range(n_data - 1):
        j = random.randint(i, n_data -1)
        labels[i], labels[j] = labels[j], labels[i]
        features[i], features[j] = features[j], features[i]

    return labels, features
    
if __name__ == "__main__":
    print("Generating dataset...")
    dataset = generate_dataset(37836 * 2)

    print("Generated {} n examples.".format(len(dataset[0])))

    print("Saving dataset...")
    with open("data/dataset.pickle", "wb") as f:
        pickle.dump(dataset, f)
