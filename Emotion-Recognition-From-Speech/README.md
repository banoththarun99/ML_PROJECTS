# Emotion Recognition from Speech using Deep Learning

## Overview

This project identifies human emotions from speech audio using Convolutional Neural Networks (CNN) and MFCC (Mel-Frequency Cepstral Coefficients) feature extraction.

The model is trained on the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset and can classify speech into multiple emotional categories.

---

## Problem Statement

Human emotions play a critical role in communication. Automatically recognizing emotions from speech can improve:

- Virtual Assistants
- Customer Service Systems
- Mental Health Monitoring
- Human-Computer Interaction

The goal of this project is to build a Deep Learning model capable of predicting emotions from audio recordings.

---

## Dataset

### RAVDESS Dataset

The project uses the RAVDESS emotional speech dataset.

Supported emotions:

| Code | Emotion |
|--------|----------|
| 01 | Neutral |
| 02 | Calm |
| 03 | Happy |
| 04 | Sad |
| 05 | Angry |
| 06 | Fearful |
| 07 | Disgust |
| 08 | Surprised |

---

## Technologies Used

- Python
- NumPy
- Librosa
- Scikit-Learn
- TensorFlow / Keras

---

## Deep Learning Architecture

### Feature Extraction

The audio files are converted into:

- MFCC Features (40 coefficients)

Additional preprocessing:

- Normalization
- Padding
- Fixed-length feature generation

### CNN Model

The architecture consists of:

1. Convolution Layer (64 Filters)
2. Max Pooling Layer
3. Dropout Layer
4. Convolution Layer (128 Filters)
5. Max Pooling Layer
6. Dropout Layer
7. Flatten Layer
8. Dense Layer (256 Neurons)
9. Output Layer (Softmax)

---

## Project Workflow

### Step 1

Load RAVDESS audio dataset

### Step 2

Extract MFCC features using Librosa

### Step 3

Encode emotion labels

### Step 4

Split dataset into training and testing sets

### Step 5

Train CNN model

### Step 6

Evaluate model performance

### Step 7

Save trained model

---

## Installation

Install required libraries:

```bash
pip install numpy librosa scikit-learn tensorflow keras
```

## Run Project

```bash
python emotion_recognition.py
```

---

## Output

The model provides:

- Emotion Classification
- Test Accuracy
- Trained Deep Learning Model

Saved Model:

```text
model/emotion_model.h5
```

---

## Real World Applications

- Call Center Analytics
- Voice Assistants
- Emotion-Aware Chatbots
- Mental Health Monitoring
- Customer Experience Analysis
- Smart Healthcare Systems

---

## Future Improvements

- LSTM and CNN-LSTM Hybrid Models
- Transformer-Based Audio Models
- Real-Time Emotion Detection
- Streamlit Deployment
- Flask API Deployment

---

## Skills Demonstrated

- Audio Processing
- Feature Engineering
- MFCC Extraction
- Deep Learning
- CNN Architecture
- Speech Analytics
- TensorFlow/Keras

---

## Author

Tharun Nayak

Aspiring AI Engineer | Machine Learning Enthusiast | Deep Learning Learner
