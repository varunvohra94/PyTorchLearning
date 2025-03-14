"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import argparse

import torch
from torchvision import transforms

import data_detup, engine, model_builder, utils

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(
        description="Python programs that trains TinyVGG model on Steak Sushi Pizza Dataset"
    )

    parser.add_argument(
        "--epochs",
        help="The number of epochs to run the model training for",
        default=5
        )
    parser.add_argument(
        "--batch_size",
        help="The Batch Size for the dataloaders (Number of data points in each batch of training)",
        default=32
    )
    parser.add_argument(
        "--device",
        help="The device to use for the model training",
        default=None
    )
    parser.add_argument(
        "--lr",
        help="The Learning Rate for the model training",
        default=0.001
    )
    parser.add_argument(
        "--hidden_units",
        help="The hidden units in the CNN",
        default=10
    )

    args = parser.parse_args()

    # Setup device agnostic mode
    if args.device:
        assert args.device == "cpu" or args.device == "mps" or args.device == "cuda", "Device must be either 'cpu', 'cuda', or 'mps'"
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    # Setup Hyperparameters
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    HIDDEN_UNITS = args.hidden_units

    
