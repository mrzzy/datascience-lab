#
# eval.py
# Evaluate a MaskRCNN model losses and prediction time
#

import os
import matplotlib.pyplot as plt
from mrcnn.config import Config
from samples.balloon.balloon import BalloonConfig, BalloonDataset
from samples.coco.coco import CocoConfig
from mrcnn import model as modellib, utils
from time import time

# Path to store trained models
MODELS_DIR = os.path.join(".", "logs")

# Train and return a mo.del for balloon image sUntitledUntitledegmentation
def train_model(config):
    # Create the model
    model = modellib.MaskRCNN(mode="training", config=BalloonConfig(),
                              model_dir=MODELS_DIR)

    # Load pretrained coco weights into the model
    #Exclude the last layers because they require a matching
    # number of classes
    model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=[
        "mrcnn_class_logits", "mrcnn_bbox_fc",
        "mrcnn_bbox", "mrcnn_mask"])

    # Train (Fine tune the head layers) the model for 30 epochs
    model.train(dataset_train, dataset_val,
            learning_rate=model.config.LEARNING_RATE,
            epochs=30,
            layers='heads')

class InferenceConfig(BalloonConfig):
    IMAGES_PER_GPU = 1

# Load a model for the given mode, using the given configuration and 
# weights from the given weights path. The special weight_path 'last'
# specifies the loading weights from the most recent training checkpoint
def load_model(mode, config, weights_path="last"):
    # Create model for inference
    model = modellib.MaskRCNN(mode, config=config,
                              model_dir=".")

    # Load pretrained weights into the model
    weights_path = weights_path if weights_path != "last" else model.find_last()
    model.load_weights(weights_path, by_name=True)
    
    return model
    
# Evalute the model specified by the given model weights_path  using the given 
# evaluation dataset, recording metrics such as loss and evalution time
# Returns evaluation time in seconds and a dictionary of loss names mapping to 
# evaluated to loss values
def evaluate_model(eval_dataset, weights_path="last", verbose=1):
    # Load model for evaluation
    model = load_model("training", InferenceConfig(), weights_path)

    # Build generator for evaluation dataset
    eval_data_generator = modellib.data_generator(eval_dataset, model.config, shuffle=True,
                                                 batch_size=model.config.BATCH_SIZE)
    
    # Evalute model with evaluation images
    n_validation = len(eval_dataset.image_ids)
    if verbose == 1:
        print("Evaluating model with {} evaluation images...".format(n_validation))

    begin_timestamp = time()
    model.compile(model.config.LEARNING_RATE, model.config.LEARNING_MOMENTUM)
    losses = model.keras_model.evaluate_generator(eval_data_generator, n_validation,
                                                  verbose=verbose)
    if verbose == 1: print("Evalution completed")
    eval_time = time() - begin_timestamp 
    loss_map = dict(zip(model.keras_model.metrics_names, losses))
    
    if verbose == 1:
        # Display results of model evaluation
        print("======== Evalution Results =======")
        print("[Losses]")
        for name, value in loss_map.items():
            print(name, ": {:e}".format(value))
        
        print("\nEvaluation time: {:.5f}s".format(eval_time))

    return eval_time, loss_map

# Plot the given list of loss maps as bar graph by exacting losses specified in
# the keys of loss_labels dict. When plotting, will label the loss with the the 
# value of loss_labels dict. Each loss_map will be given the correponding legend
# in map_legends.
# Plotting is done in sequence, so plots from later loss_maps will overlap the
# ones plotted before.
LOSS_LABELS = {
    "loss": "Total Loss",
    "rpn_class_loss": "RPN Classification Loss",
    "rpn_bbox_loss": "RPN Bounding Box Loss",
    "mrcnn_class_loss": "Classification Loss",
    "mrcnn_bbox_loss": "Bounding Box Loss",
    "mrcnn_mask_loss": "Segmentation Mask Loss"
}

def plot_losses(loss_maps, map_legends, loss_labels=LOSS_LABELS):
    # check arguments 
    if len(loss_maps) != len(map_legends):
        raise ValueError("The length of loss_maps and map_legends should be the same")

    # Unpack loss labels into loss keys nad names
    loss_keys = loss_labels.keys()
    loss_names = loss_labels.values()
    loss_indexes = range(len(loss_keys))

    # Plot each loss map as a bar graph
    bar_plots = []
    for loss_map in loss_maps:
        # Extract loss values from loss map
        loss_vals = [ loss_map[k] for k in loss_keys ]
        # Plot loss values as bar graph
        plot = plt.bar(loss_indexes, loss_vals)
        bar_plots.append(plot)
        
    # Decorate graph with labels to make it more readable 
    plt.xlabel("Losses")
    plt.ylabel("Magnitude")
    plt.xticks(loss_indexes, loss_names)
    plt.legend([p[0] for p in bar_plots], map_legends)
    
if __name__ == "__main__":
    # Load dataset
    dataset_path = "datasets/balloon"
    dataset_train = BalloonDataset()
    dataset_train.load_balloon(dataset_path, "train")
    dataset_train.prepare()

    dataset_val = BalloonDataset()
    dataset_val.load_balloon(dataset_path, "val")
    dataset_val.prepare()
    
    # Train model
    train_model(BalloonConfig())
    
    # Evalute trained model
    eval_time, loss_map = evaluate_model(train_dataset)
    plot_losses([loss_map], ["Control Losses"])
    plt.show()
