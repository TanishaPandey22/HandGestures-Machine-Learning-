Task-04: Hand Gesture Recognition
📌 Project Overview

This project develops a Hand Gesture Recognition system using Machine Learning and the LeapGestRecog dataset. The model is trained to identify and classify different hand gestures from images.

The system can be used for touchless human-computer interaction, gesture-based controls, and other applications where hand movements are used as commands.

📂 Dataset

Dataset: LeapGestRecog
Source: Kaggle – LeapGestRecog

The dataset contains approximately 20,000 infrared hand images covering 10 different hand gestures performed by multiple subjects.

🎯 Objectives
Detect and recognize different hand gestures.
Preprocess and prepare hand images for training.
Train a machine learning/deep learning classification model.
Evaluate the model's performance.
Predict gestures from new images.
Enable gesture-based human-computer interaction.
🛠️ Technologies Used
Python
OpenCV
NumPy
TensorFlow / Keras
Matplotlib
Scikit-learn
📁 Project Structure
Task-04-Hand-Gesture-Recognition/
│
├── train_model.py       # Train the gesture recognition model
├── predict.py           # Predict gesture from an image
├── webcam.py            # Real-time gesture recognition using webcam
├── requirements.txt     # Required Python libraries
├── README.md            # Project documentation
│
├── dataset/
│   └── LeapGestRecog/
│
└── gesture_model.h5     # Trained model

📈Dataset
Dataset :-  https://www.kaggle.com/gti-upm/leapgestrecog


⚙️ Installation

Clone/download the project and open the project folder in the terminal.

Create a virtual environment:

python -m venv venv

Activate it on Windows:

venv\Scripts\activate

Install the required libraries:

pip install -r requirements.txt
🚀 Training the Model

Place the LeapGestRecog dataset inside the dataset folder and run:

python train_model.py

The trained model will be saved for making predictions.

🔍 Predicting a Gesture

To classify a new hand gesture image:

python predict.py

The program will process the image and display the predicted gesture.

📷 Real-Time Recognition

To recognize gestures using a webcam:

python webcam.py

The webcam will capture frames and the model will predict the gesture in real time.
