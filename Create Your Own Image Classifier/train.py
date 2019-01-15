#!/usr/bin/env python3

import time
import argparse
import numpy as np
from PIL import Image
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms, models
from torchvision.datasets import ImageFolder

from tool import save_checkpoint, load_checkpoint

def parse_args():
    
    parser = argparse.ArgumentParser(description= "Training process")
    parser.add_argument("--data_dir", action= "store")
    parser.add_argument("--arch", dest="arch", default= "vgg19", choices= ["vgg13", "vgg19"])
    parser.add_argument("--learning_rate", dest= "learning_rate", default= "0.01")
    parser.add_argument("--hidden_units", dest= "hidden_units", default= "512")
    parser.add_argument("--epochs", dest= "epochs", default= "8")
    parser.add_argument("--gpu", action= "store_true", default= True)
    return parser.parse_args()

def train(model, criterion, optimizer, dataloaders, epochs, gpu):
      
    cuda = torch.cuda.is_available()
    # GPU
    if cuda:
        model.cuda()
    else:
        model.cpu()

    running_loss = 0
    accuracy = 0

    start_train = time.time()
    print("Training started")

    for e in range(epochs):

        train_mode = 0
        valid_mode = 1

        for mode in [train_mode, valid_mode]:   
            if mode == train_mode:
                model.train()
            else:
                model.eval()

            pass_count = 0

            for data in dataloaders[mode]:
                pass_count += 1
                inputs, labels = data
                if cuda == True:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                optimizer.zero_grad()
                # Forward
                output = model.forward(inputs)
                loss = criterion(output, labels)
                # Backward
                if mode == train_mode:
                    loss.backward()
                    optimizer.step()                

                running_loss += loss.item()
                ps = torch.exp(output).data
                equality = (labels.data == ps.max(1)[1])
                accuracy = equality.type_as(torch.cuda.FloatTensor()).mean()

            if mode == train_mode:
                print("\nEpoch: {}/{} ".format(e+1, epochs),
                      "\nTraining Loss: {:.4f}  ".format(running_loss/pass_count))
            else:
                print("Validation Loss: {:.4f}  ".format(running_loss/pass_count),
                  "Accuracy: {:.4f}".format(accuracy))

            running_loss = 0
            
def main():
    
    args = parse_args()
    
    data_dir = "flowers"
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"
    test_dir = data_dir + "/test"

    training_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                                transforms.RandomRotation(30),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406], 
                                                                     [0.229, 0.224, 0.225])])
    validataion_transforms = transforms.Compose([transforms.Resize(256),
                                                 transforms.CenterCrop(224),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                                                      [0.229, 0.224, 0.225])])
    testing_transforms = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])])
    
    image_datasets = [ImageFolder(train_dir, transform=training_transforms),
                      ImageFolder(valid_dir, transform=validataion_transforms),
                      ImageFolder(test_dir, transform=testing_transforms)]
    
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size= 64, shuffle= True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size= 64),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size= 64)]
   
    model = getattr(models, args.arch)(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False
    
    if args.arch == "vgg13":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ("fc1", nn.Linear(feature_num, 1024)),
                                  ("drop", nn.Dropout(p= 0.5)),
                                  ("relu", nn.ReLU()),
                                  ("fc2", nn.Linear(1024, 102)),
                                  ("output", nn.LogSoftmax(dim= 1))]))
    elif args.arch == "vgg19":
        feature_num = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                                  ("fc1", nn.Linear(feature_num, 1024)),
                                  ("drop", nn.Dropout(p= 0.5)),
                                  ("relu", nn.ReLU()),
                                  ("fc2", nn.Linear(1024, 102)),
                                  ("output", nn.LogSoftmax(dim= 1))]))

    model.classifier = classifier
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr= float(args.learning_rate))
    epochs = int(args.epochs)
    class_index = image_datasets[0].class_to_idx
    gpu = args.gpu
    train(model, criterion, optimizer, dataloaders, epochs, gpu)
    model.class_to_idx = class_index
    save_checkpoint(model, optimizer, args, classifier)

if __name__ == "__main__":
    main()