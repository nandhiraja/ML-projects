# Vegetable Classification using Deep Learning

## Overview
This project implements a Deep Learning model to classify vegetables into one of 13 categories. Using a Convolutional Neural Network (CNN), the model achieves an accuracy of 95% on the test dataset. This project demonstrates the power of deep learning in solving image classification problems with high accuracy.

## Dataset
The dataset contains images of vegetables, classified into the following 13 categories:
1. Broccoli
2. Capsicum
3. Bottle Gourd
4. Radish
5. Tomato
6. Brinjal
7. Pumpkin
8. Carrot
9. Papaya
10. Cabbage
11. Bitter Gourd
12. Cauliflower
13. Bean

Each category contains a sufficient number of labeled images to train the deep learning model effectively.

## Model Architecture
The model is based on a Convolutional Neural Network (CNN) architecture with the following components:
- **Input Layer:** Accepts images of a fixed size (e.g., 128x128 pixels).
- **Convolutional Layers:** Extract features using multiple convolutional and pooling layers.
- **Dense Layers:** Fully connected layers for classification.
- **Output Layer:** A softmax layer with 13 output nodes, one for each vegetable category.

The model is implemented using TensorFlow/Keras.

## Requirements
### Prerequisites
Ensure you have the following installed:
- Python 3.7+
- TensorFlow 2.0+
- NumPy
- OpenCV
- Matplotlib
- Pandas

Install the dependencies using the command:
```bash
pip install -r requirements.txt
