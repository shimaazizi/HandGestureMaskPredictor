# HandGestureMaskPredictor

This repository contains code for a segmentation project using the UNet architecture. The project involves training a deep learning model to perform image segmentation on a custom dataset of hand gestures.

![GitHub repo size](https://img.shields.io/github/repo-size/shimaazizi/HandGestureMaskPredictor)
![GitHub stars](https://img.shields.io/github/stars/shimaazizi/HandGestureMaskPredictor?style=social)
![GitHub forks](https://img.shields.io/github/forks/shimaazizi/HandGestureMaskPredictor?style=social)

## Table of Contents

- 📚 [Dataset](#dataset)
- 💻 [Installation](#installation)
- 🛠️ [Data Preparation](#data-preparation)
- 🧠 [Model](#model)
- 🔧 [Utils](#utils)
- 🚀 [Training](#training)
- 📈 [Result](#result)

## 📚Dataset

The dataset is organized into the following directories:

- **Dataset**
    - `Fist`
    - `OpenPalm`
    - `PeaceSign`
    - `ThumbsUp`

- **New_Mask**
    - `Fist_Mask`
    - `OpenPalm_Mask`
    - `PeaceSign_Mask`
    - `ThumbsUp_Mask`


## 💻Installation
1. Clone the repository:
   
   git clone https://github.com/shimaazizi/HandGestureMaskPredictor.git
   
   cd HandGestureMaskPredictor

3. Install the required packages:
   
   `pip install -r requirements.txt`

   
The librarie we need:

[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch 2.4.0](https://img.shields.io/badge/PyTorch-2.4.0-red)](https://pytorch.org/get-started/locally/)
[![TorchVision 0.19.0](https://img.shields.io/badge/TorchVision-0.19.0-orange)](https://pytorch.org/vision/stable/index.html)
[![NumPy 2.0.1](https://img.shields.io/badge/NumPy-2.0.1-blue)](https://numpy.org/)
[![Matplotlib 3.9.1](https://img.shields.io/badge/Matplotlib-3.9.1-blueviolet)](https://matplotlib.org/)
[![Scikit-Learn 1.5.1](https://img.shields.io/badge/Scikit--Learn-1.5.1-green)](https://scikit-learn.org/stable/)
[![Pillow 10.4.0](https://img.shields.io/badge/Pillow-10.4.0-yellow)](https://pillow.readthedocs.io/en/stable/)


## 🛠️Data Preparation
This part provides a PyTorch-based pipeline for image segmentation, including data loading and preprocessing. 


## 🧠Model
This part provides a PyTorch implementation of the U-Net architecture for image segmentation.

*Encoder*: Captures image features at multiple scales using a series of convolutional blocks and pooling layers.

*Decoder*: Upsamples feature maps and concatenates them with corresponding encoder outputs to reconstruct the segmented image.

*U-Net*: Combines the encoder and decoder, with a final convolutional layer to produce the segmentation map.


## 🔧utils
This part includes functions to evaluate and visualize image segmentation model performance using PyTorch.

Functions:

* accuracy (pred, target): Computes the accuracy of predictions against the target masks.

* dice_score (pred, target, epsilon=1e-6): Calculates the Dice coefficient for segmentation tasks.

* visualize_prediction (model, test_loader, device, num_classes=4): Visualizes model predictions alongside true masks and original images


## 🚀Training
This module trains and evaluates a U-Net model for image segmentation using PyTorch.

Functions:

* train_model(model, train_loader, val_loader, test_loader, num_classes=4, num_epochs=80, device='cuda'): Trains the model and evaluates it on validation and test sets. Saves the model to unet_model.pth.

* evaluate_model(model, test_loader, device='cuda'): Evaluates the model on the test set


## 📈Result 
in main.py script integrates dataset loading, model training, evaluation, and visualization.

* create_dataloaders: Loads and preprocesses the dataset.
 
* UNet: Defines the U-Net model architecture.
  
* train_model: Trains the model on the training set and evaluates on the validation set.
  
* evaluate_model: Evaluates the trained model on the test set.
  
* visualize_prediction: Visualizes the model predictions compared to the true masks.



