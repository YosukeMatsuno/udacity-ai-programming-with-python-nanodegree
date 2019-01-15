#!/usr/bin/env python3

import os
import json
import copy
import argparse

import torch
from torchvision import transforms, datasets

def save_checkpoint(model, optimizer, args, classifier):
    
    checkpoint = {"arch": args.arch, 
                  "model": model,
                  "learning_rate": args.learning_rate,
                  "hidden_units": args.hidden_units,
                  "classifier" : classifier,
                  "epochs": args.epochs,
                  "optimizer": optimizer.state_dict(),
                  "state_dict": model.state_dict(),
                  "class_to_idx": model.class_to_idx}

    torch.save(checkpoint, "checkpoint.pth")
    
def load_checkpoint(file_path):

    checkpoint = torch.load(file_path)
    model = checkpoint["model"]
    model.classifier = checkpoint["classifier"]
    learning_rate = checkpoint["learning_rate"]
    epochs = checkpoint["epochs"]
    optimizer = checkpoint["optimizer"]
    model.load_state_dict(checkpoint["state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]
    
    return model

def load_cat_names(filename):
    with open(filename) as file:
        category_names = json.load(file)
    return category_names