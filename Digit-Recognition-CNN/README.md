# Handwritten Digit Recognition Using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset.

The model is trained on thousands of grayscale images and learns to classify digits from 0 to 9 with high accuracy.

---

## Problem Statement

Handwritten digit recognition is a fundamental computer vision problem with applications in:

- Postal Code Recognition
- Bank Check Processing
- Document Digitization
- Automated Form Reading

The goal is to build a Deep Learning model capable of accurately classifying handwritten digits.

---

## Dataset

### MNIST Dataset

The MNIST dataset contains:

- 60,000 Training Images
- 10,000 Testing Images
- 10 Digit Classes (0–9)

Image Size:

```text
28 × 28 Pixels
```

Image Type:

```text
Grayscale
```

---

## Technologies Used

- Python
- NumPy
- TensorFlow
- Keras
- Matplotlib

---

## Deep Learning Architecture

### CNN Layers

1. Input Layer (28×28×1)

2. Convolution Layer

```text
32 Filters
Kernel Size: 3×3
Activation: ReLU
```

3. Max Pooling Layer

```text
Pool Size: 2×2
```

4. Convolution Layer

```text
64 Filters
Kernel Size: 3×3
Activation: ReLU
```

5. Max Pooling Layer

```text
Pool Size: 2×2
```

6. Flatten Layer

7. Dropout Layer

```text
Dropout Rate: 0.5
```

8. Output Layer

```text
10 Neurons
Activation: Softmax
```

---

## Project Workflow

### Step 1

Load MNIST dataset

### Step 2

Normalize image pixel values

### Step 3

Reshape images for CNN input

### Step 4

One-Hot Encode labels

### Step 5

Build CNN architecture

### Step 6

Train model

### Step 7

Evaluate performance

### Step 8

Predict handwritten digits

---

## Model Training

Training Parameters:

```text
Batch Size: 128
Epochs: 15
Optimizer: Adam
Loss Function: Categorical Crossentropy
```

---

## Installation

Install required packages:

```bash
pip install tensorflow numpy matplotlib
```

---

## Run Project

```bash
python digit_recognition.py
```

---

## Expected Output

The project displays:

- Model Summary
- Training Accuracy
- Validation Accuracy
- Test Accuracy
- Predicted Digit
- Visualization of Sample Image

---

## Real World Applications

- Postal Automation
- Bank Check Verification
- OCR Systems
- Document Processing
- Automated Data Entry

---

## Future Improvements

- Transfer Learning
- Real-Time Webcam Digit Recognition
- Streamlit Deployment
- Mobile Application Integration
- Custom Handwritten Dataset Training

---

## Skills Demonstrated

- Deep Learning
- Computer Vision
- CNN Design
- Image Classification
- TensorFlow/Keras
- Model Evaluation

---

## Author

Tharun Nayak

Aspiring AI Engineer | Machine Learning Enthusiast | Deep Learning Learner
