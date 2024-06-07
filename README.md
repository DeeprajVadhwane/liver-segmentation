# Liver Tumor Segmentation Project

This project involves building and training different deep learning models for liver tumor segmentation from CT scan images, using Simple CNN, U-Net, and ResNet architectures.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Preparation](#dataset-preparation)
3. [Preprocessing](#preprocessing)
4. [Model Architectures](#model-architectures)
    - [Simple CNN](#simple-cnn)
    - [U-Net](#u-net)
    - [ResNet](#resnet)
5. [Training the Models](#training-the-models)
6. [Evaluation and Results](#evaluation-and-results)
7. [Conclusion and Next Steps](#conclusion-and-next-steps)

## Introduction

This project aims to develop an automated system for liver tumor segmentation using CT scan images. The segmentation helps in identifying the exact location and size of tumors, assisting in better diagnosis and treatment planning.

## Dataset Preparation

1. **Download Dataset**: The dataset can be downloaded from the Kaggle competition or any other relevant source.
2. **Organize Data**: Structure the data into separate directories for CT scans and corresponding segmentation masks.

## Preprocessing

1. **Import Libraries**: Import necessary libraries such as `numpy`, `pandas`, `nibabel`, `cv2`, `matplotlib`, and deep learning frameworks like TensorFlow and FastAI.
2. **Create Metadata File**: Traverse through the dataset directories to create a metadata file (e.g., a CSV file) that lists all CT scan files and their corresponding segmentation masks.
3. **Read NIfTI Files**: Write a function to read NIfTI (.nii) files and convert them into numpy arrays.
4. **Data Augmentation**: Apply data augmentation techniques like rotations, flips, and scaling to increase the diversity of the training set.

## Model Architectures

### Simple CNN

1. **Define Architecture**: Create a simple Convolutional Neural Network (CNN) with a few convolutional and pooling layers followed by fully connected layers.
2. **Compile Model**: Use the Adam optimizer and binary cross-entropy loss for compiling the model.

### U-Net

1. **Define Architecture**: Implement the U-Net architecture, which includes an encoder (down-sampling path) and a decoder (up-sampling path) with skip connections.
2. **Compile Model**: Use the Adam optimizer and binary cross-entropy loss for compiling the model, including the dice coefficient as a metric.

### ResNet

1. **Define Architecture**: Utilize a pre-trained ResNet model for feature extraction and add a custom segmentation head.
2. **Compile Model**: Use the Adam optimizer and binary cross-entropy loss for compiling the model.

## Training the Models

1. **Split Data**: Split the dataset into training, validation, and test sets.
2. **Set Callbacks**: Define callbacks such as ModelCheckpoint to save the best model, ReduceLROnPlateau to reduce learning rate on plateau, and TensorBoard for logging.
3. **Train Models**: Train the models on the training set and validate on the validation set for a fixed number of epochs.

## Evaluation and Results

### Model Performance Comparison

| Model         | Accuracy     | Precision    | Val Loss  | Val Accuracy | Val Precision | Learning Rate |
|---------------|--------------|--------------|-----------|--------------|---------------|---------------|
| Simple CNN    | 6.6906e-05   | 0.8796       | 0.0910    | 4.2511e-04   | 0.4980        | 0.0010        |
| U-Net         | 0.0015       | 0.9477       | 0.0098    | 0.0020       | 0.9263        | 1.0000e-06    |
| ResNet        | 0.9894       | 0.7841       | 0.0729    | 0.9897       | 0.7749        | 1.0000e-06    |

## Conclusion and Next Steps

### Final Model Selection

We selected the ResNet model as our final model because it demonstrated the highest validation accuracy (0.9897) and the best balance between precision (0.7841) and validation loss (0.0729). While the U-Net model showed excellent precision, its validation loss and accuracy were not as favorable as those of the ResNet model. The Simple CNN model, despite being faster, did not perform well in terms of accuracy and precision. Therefore, the ResNet model was chosen for its superior overall performance in liver tumor segmentation.

