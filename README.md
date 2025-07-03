# Poker Card Class Detection

A real-time playing card detection system using YOLOR (You Only Learn One Representation), designed to automatically recognize and classify standard playing cards from images or webcam video streams.

## Overview

This project leverages the YOLOR deep learning model to detect and classify playing cards in real-time. The system is optimized for speed and accuracy, making it suitable for applications such as digital card games, augmented reality, and human-computer interaction.
![image](https://github.com/user-attachments/assets/137ad941-e0a3-4031-a1ed-5ac057c33f6d)


## Features

- **Real-time detection** of playing cards using webcam or uploaded images.
- **High accuracy**: Achieves mAP50 of 0.99 and mAP50-95 of 0.929 on validation data.
- **Robust to variations** in card orientation, position, lighting, and background.
- **Automatic classification** of poker hand combinations (e.g., Royal Flush, Full House).
- **Web-based interface**: Easily accessible via a local Flask server and HTML frontend.
- **Efficient**: Runs on consumer-grade hardware without the need for expensive GPUs.

## How It Works

1. **Data Acquisition**: Uses a dataset of 3,000 annotated playing card images with diverse conditions.
2. **Model Training**: Trains a YOLORv11n model using Ultralytics YOLO and Python.
3. **Detection Pipeline**: Captures frames from webcam or images, detects cards, and classifies poker hands in real-time.
4. **Deployment**: The system is deployed locally using Flask, with a simple web interface for user interaction.

## Installation

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```
4. Run the backend server:
   ```bash
   python backend/main.py
   ```
5. Access the web interface via your browser.

## Usage

- Upload an image or use your webcam to detect and classify playing cards.
- The system will display detected cards and the best poker hand combination in real-time.

## Project Structure

- `backend/`: Python backend, trained model, and detection logic.
- `frontend/`: HTML frontend for user interaction.
- `requirements.txt`: Python dependencies.

## Results

- **Precision**: 0.971
- **Recall**: 0.966
- **mAP50**: 0.99
- **mAP50-95**: 0.929
- **Inference speed**: ~61 ms per image (real-time)

## Future Work

- Support for more card designs and games.
- Performance optimization for low-spec devices.
- Advanced game strategy analysis features.

## References

- [YOLOR](https://github.com/WongKinYiu/yolor)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Roboflow](https://roboflow.com/)
- [Flask](https://flask.palletsprojects.com/)
- [OpenCV](https://opencv.org/)
