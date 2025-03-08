import matplotlib.pyplot as plt
from itertools import product
from torchvision import transforms
from PIL import Image
import numpy as np
import random
import glob
import torch
import os

def visualize_predictions(images, masks, outputs, save_path = "", epoch = 0, batch_idx = 0, show = False):
    '''
    Visualizes and saves sample predictions for a given batch of images, masks, and model outputs.

    Args:
    - images (torch.Tensor): Input images (batch of tensors).
    - masks (torch.Tensor): Ground truth segmentation masks (batch of tensors).
    - outputs (torch.Tensor): Model outputs (batch of tensors).
    - save_path (str): Directory path where the visualization will be saved.
    - epoch (int): Current epoch number (for labeling the file).
    - batch_idx (int): Index of the current batch (for labeling the file).
    
    Functionality:
    - Displays and saves the first few samples from the batch, showing the input images, ground truth masks, and predicted masks.
    - Applies a sigmoid function to the outputs and uses a threshold of 0.5 to convert them to binary masks.
    '''
    

    num_img = len(images) if len(images) < 5 else 5
    #Unnormalize images
    images = (images * 0.5) + 0.5 
    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    outputs = (torch.nn.functional.sigmoid(outputs) > 0.5).float()
    outputs = outputs.detach().cpu().numpy()
    plt.figure()
    for id in range(num_img):
        plt.subplot(5, 3, id*3 + 1)
        plt.imshow(images[id].squeeze(0), cmap= "gray")
        plt.title("Image")
        plt.axis('off')

        plt.subplot(5, 3, id*3 + 2)
        plt.imshow(masks[id].squeeze(0), cmap= "gray")
        plt.title("Mask")
        plt.axis('off')

        plt.subplot(5, 3, id*3 + 3)
        plt.imshow(outputs[id].squeeze(0), cmap= "gray")
        plt.title("Output")
        plt.axis('off')
    if save_path != "":
        #For the test predictions
        if epoch == -1:
            plt.savefig(save_path + f"/{batch_idx}.jpg")
        #For the validaiton predictions
        else:    
            plt.savefig(save_path + f"/{epoch+1}/{batch_idx}.jpg")
    plt.tight_layout()
    #Show predictions after batches
    plt.close()


def plot_train_val_history(train_loss_history, val_loss_history, plot_dir, args):
    '''
    Plots and saves the training and validation loss curves.

    Args:
    - train_loss_history (list): List of training loss values over epochs.
    - val_loss_history (list): List of validation loss values over epochs.
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    
    Functionality:
    - Plots the train and validation loss curves.
    - Saves the plot as a JPG file in the specified directory.
    '''
    plt.figure(figsize=(10,6))
    plt.plot(range(len(train_loss_history)), train_loss_history, label = "Training Loss")
    plt.plot(range(len(val_loss_history)), val_loss_history, label = "Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Train/Validation Loss")
    plt.legend(["Train", "Validation"])
    plt.savefig(plot_dir + f"/{args.exp_id}/train_val.jpg")
    plt.show()

def plot_metric(x, ylabel, plot_dir, args, metric, xlabel = "Epochs"):
    '''
    Plots and saves a metric curve over epochs.

    Args:
    - x (list): List of metric values over epochs.
    - label (str): Label for the y-axis (name of the metric).
    - plot_dir (str): Directory path where the plot will be saved.
    - args (argparse.Namespace): Parsed arguments containing experiment details.
    - metric (str): Name of the metric (used for naming the saved file).
    
    Functionality:
    - Plots the given metric curve.
    - Saves the plot as a JPEG file in the specified directory.
    '''
    plt.figure(figsize=(10,6))
    plt.plot(range(len(x)), x)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(metric)
    plt.savefig(plot_dir + f"/{args.exp_id}/{metric}.jpg")
    plt.show()