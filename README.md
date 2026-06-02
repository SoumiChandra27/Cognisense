# Cognisense – AI-Powered Attention Monitoring System

## Overview

Cognisense is an AI-powered attention monitoring system designed to analyze user engagement and focus in real time using computer vision and audio analysis. The system combines facial landmark detection, attention classification, speech activity monitoring, and report generation to provide insights into user attentiveness during online sessions, learning activities, and assessments.

## Features

* Real-time attention monitoring using webcam input
* Facial landmark detection using MediaPipe
* Attention classification using a fine-tuned deep learning model
* Speaking activity detection using microphone input
* Drowsiness and distraction detection
* Live monitoring dashboard
* Automated PDF report generation
* Chrome extension integration
* WebSocket-based communication between extension and backend

## Technology Stack

### Frontend

* HTML
* CSS
* JavaScript
* Chrome Extension APIs

### Backend

* Python
* AioHTTP
* WebSockets

### AI & Computer Vision

* OpenCV
* MediaPipe
* TensorFlow/Keras
* NumPy

### Reporting

* FPDF2
* Pillow

## Project Structure

Cognisense/
│
├── backend/
│ ├── server.py
│ ├── Attention_final.py
│ ├── requirements.txt
│ ├── attention_classifier_model_finetuned.h5
│ └── Generated Reports
│
├── attention_extension/
│ ├── manifest.json
│ ├── popup.html
│ ├── popup.js
│ ├── background.js
│ └── dashboard files
│
└── README.md

## Installation

### 1. Clone the Repository

git clone <repository-url>

cd Cognisense

### 2. Create Virtual Environment

Windows:

python -m venv venv

venv\Scripts\activate

### 3. Install Dependencies

pip install -r requirements.txt

### 4. Run Backend Server

cd backend

python server.py

Server will start at:

ws://localhost:8765

## Loading the Chrome Extension

1. Open Chrome.
2. Navigate to chrome://extensions
3. Enable Developer Mode.
4. Click Load Unpacked.
5. Select the attention_extension folder.
6. Open the extension from the Chrome toolbar.

## Usage

1. Start the backend server.
2. Load the Chrome extension.
3. Click Start Monitoring.
4. Allow camera and microphone permissions.
5. Monitor attention and speaking activity in real time.
6. Open the dashboard to view analytics.
7. Save and generate the final PDF report.

## Machine Learning Model

The system uses a fine-tuned attention classification model:

attention_classifier_model_finetuned.h5

The model predicts user attention levels based on facial and behavioral features extracted from webcam input.

## Future Enhancements

* Emotion recognition
* Head pose estimation
* Eye-gaze tracking
* Cloud-based analytics dashboard
* Multi-user monitoring support
* Integration with e-learning platforms

## Applications

* Online Learning
* Virtual Interviews
* Employee Productivity Monitoring
* Examination Proctoring
* Attention Analysis Research

## Author

Developed as an AI-based attention monitoring and behavioral analysis system using Computer Vision, Deep Learning, and Real-Time Analytics.
