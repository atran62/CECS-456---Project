# CECS-456---Project
A project for classifying images in the Natural Images dataset using CNN.


## Introduction

The goal of this project is to classify images into one of eight categories: `airplane`, `car`, `cat`, `dog`, `flower`, `fruit`, `motorbike`, and `person`. The model is trained on the **Natural Images dataset**, a collection of labeled images representing real-world objects.

The project uses a CNN for feature extraction and classification. The performance of the model is evaluated in terms of accuracy and loss on both training and validation datasets.

## Model Architecture

The CNN model includes the following layers:
1. **Input Layer**: Accepts images of size (128, 128, 3).
2. **Convolutional Layers**: 
   - Two sets of Conv2D layers with ReLU activation followed by MaxPooling2D layers.
3. **Flattening Layer**: Converts the feature maps into a 1D feature vector.
4. **Fully Connected Layers**: 
   - Dense layer with 128 units and ReLU activation.
   - Output layer with 8 units and softmax activation for classification.

The model is compiled using:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

## Usage

### Prerequisites
Ensure you have the following installed:
- Python 3.10 or higher
- TensorFlow 2.0 or higher
- Matplotlib
- NumPy

Run the python script:
    python project.py

## Results
The CNN achieved the following performance:
- Training Accuracy: ~91%
- Validaiton Accuravy: ~89%
- Validation Loss: ~0.31

Training and validation accuracy and loss over 10 epochs.

## Acknowledgements 
- Dataset: The Natural Images dataset was sourced from kaggle (https://www.kaggle.com/datasets/prasunroy/natural-images?resource=download)
- Libraries: TensorFlow, Keras, Matplotlib, NumPy