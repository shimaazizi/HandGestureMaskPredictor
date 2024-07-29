# HandGestureMaskPredictor

This repository contains code for a segmentation project using the UNet architecture. The project involves training a deep learning model to perform image segmentation on a custom dataset of hand gestures.


### Table of Contents
* Dataset
* Installation


### Dataset
* Dataset
  
    ├── Fist
  
    ├── OpenPalm
  
    ├── PeaceSign
  
    ├── ThumbsUp
  
* New_Mask
  
    ├── Fist_Mask
  
    ├── OpenPalm_Mask
  
    ├── PeaceSign_Mask
  
    ├── ThumbsUp_Mask


### Installation
1. Clone the repository:
   
   git clone https://github.com/shimaazizi/HandGestureMaskPredictor.git
   
   cd HandGestureMaskPredictor

3. Install the required packages:
   
   pip install -r requirements.txt


### Data Preparation
This part provides a PyTorch-based pipeline for image segmentation, including data loading and preprocessing. 


### Model
This part provides a PyTorch implementation of the U-Net architecture for image segmentation.

*Encoder*: Captures image features at multiple scales using a series of convolutional blocks and pooling layers.
*Decoder*: Upsamples feature maps and concatenates them with corresponding encoder outputs to reconstruct the segmented image.
*U-Net*: Combines the encoder and decoder, with a final convolutional layer to produce the segmentation map.


### utils
This part includes functions to evaluate and visualize image segmentation model performance using PyTorch.

Functions:

* accuracy (pred, target): Computes the accuracy of predictions against the target masks.

* dice_score (pred, target, epsilon=1e-6): Calculates the Dice coefficient for segmentation tasks.

* visualize_prediction (model, test_loader, device, num_classes=4): Visualizes model predictions alongside true masks and original images


