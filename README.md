# Autonomous Traffic Sign Detection & Vehicle Control

## Overview

This project uses machine learning, computer vision, and optical character recognition (OCR) to detect and interpret traffic signs using a camera mounted on a Raspberry Pi-powered vehicle.

The system combines a trained YOLO object detection model with Tesseract OCR and real-time vehicle control. Camera footage is processed on a Raspberry Pi, where the trained model identifies traffic signs and the resulting detections are used to make decisions about the vehicle's movement.

The project demonstrates the deployment of a machine learning model onto physical hardware and the integration of ML predictions into a real-time autonomous control system.

## Project Goal

The goal of this project was to develop and deploy a machine learning-based traffic sign detection system capable of identifying traffic signs and using those predictions to control an autonomous vehicle.

The system was designed to:

* Capture live video using an onboard camera
* Process camera frames using OpenCV and Picamera2
* Detect traffic signs using a trained YOLO model
* Run the trained model using TensorFlow Lite
* Use Tesseract OCR to recognize text from detected signs
* Evaluate detection confidence and bounding-box size
* Determine when a detected sign is sufficiently close to the vehicle
* Stop the vehicle when an approaching Stop Sign is detected
* Resume movement after the programmed stop period

## Machine Learning & Computer Vision

The project uses a YOLO-based object detection model trained to identify traffic signs.

The trained model was converted to TensorFlow Lite (`New_model.tflite`) for deployment on the Raspberry Pi. This allowed the model to perform inference directly on the vehicle rather than requiring a separate computer to process the camera feed.

The system also incorporates Tesseract OCR to extract text from detected traffic signs. Combining object detection with OCR allowed the system to use both visual classification and text recognition as part of its traffic-sign processing pipeline.

OpenCV is used for image processing and displaying the camera feed, while Picamera2 provides the Raspberry Pi camera interface.

## System Architecture

The overall workflow is:

```text
Camera Input
     ↓
Picamera2
     ↓
OpenCV Image Processing
     ↓
YOLO Object Detection Model
     ↓
Traffic Sign Detection
     ↓
Confidence & Bounding Box Evaluation
     ↓
Tesseract OCR
     ↓
Vehicle Control Logic
     ↓
Stop / Resume
```

The vehicle does not simply react to every detection. The control logic evaluates the model's prediction confidence and the detected object's bounding-box size to determine when the sign is close enough to trigger a response.

## Technologies Used

* Python
* Machine Learning
* YOLO Object Detection
* TensorFlow Lite
* Tesseract OCR
* OpenCV
* NumPy
* Picamera2
* Raspberry Pi 5
* Yahboom Raspbot
* Git/GitHub

## Repository Structure

```text
Capstone/
├── New_model.tflite
├── RaspbotAIModelDemo.py
├── Requirements.txt
├── Yolo_model.ipynb
└── README.md
```

### `RaspbotAIModelDemo.py`

The primary vehicle-control and inference script. It initializes the Raspberry Pi camera, loads the trained TensorFlow Lite model, processes camera frames, performs traffic-sign detection and OCR processing, and uses the results to control the Raspbot.

### `New_model.tflite`

The trained TensorFlow Lite machine learning model used for inference on the Raspberry Pi.

### `Yolo_model.ipynb`

The Jupyter Notebook containing the model development and training workflow.

### `Requirements.txt`

Contains the Python packages required by the project.

## Running the Project

### Hardware

This project was developed using:

* Raspberry Pi 5
* Yahboom Raspbot
* Raspberry Pi-compatible camera
* Vehicle-mounted camera/servo system

### Software Setup

Clone the repository:

```bash
git clone https://github.com/dssantii/Capstone.git
cd Capstone
```

Install the required Python dependencies:

```bash
pip install -r Requirements.txt
```

Tesseract OCR must also be installed and configured on the Raspberry Pi.

The repository already contains the trained model (`New_model.tflite`), so the training dataset is not required to run the inference and vehicle-control system.

The vehicle-control program can then be started with:

```bash
python RaspbotAIModelDemo.py
```

The program will initialize the camera, load the trained model, and begin processing camera frames.

> **Note:** The vehicle-control portion of this project requires the appropriate Raspberry Pi, camera, and Yahboom Raspbot hardware. Additional Yahboom hardware software/drivers may also be required for motor control.

## Dataset

The traffic sign detection model was trained using a traffic sign image dataset.

The original dataset is not required to run the trained model. The repository contains the trained TensorFlow Lite model, allowing the inference program to load the model and perform predictions without retraining.

The dataset would be required to reproduce the model-training process, retrain the model, or fine-tune it with additional data.

## Model Deployment

A primary objective of this project was deploying a machine learning model to a resource-constrained physical device.

The trained YOLO model was converted to TensorFlow Lite and deployed directly to a Raspberry Pi 5. The Raspberry Pi then performed inference using live camera input from the vehicle.

The project also incorporated Tesseract OCR into the deployed computer vision pipeline, requiring the system to coordinate machine learning inference, image processing, text recognition, and vehicle-control decisions on the Raspberry Pi.

This required considering:

* Model size and computational requirements
* Inference speed
* Camera processing
* OCR processing
* Detection confidence
* Bounding-box size
* Detection frequency
* Real-time vehicle control
* Raspberry Pi hardware limitations

The project therefore extended beyond model training into the deployment and integration of an ML system with physical hardware.

## Results

The completed system was able to process live camera footage, identify trained traffic-sign classes, and use the resulting detections as inputs to the vehicle-control system. The system also incorporated OCR to extract text from relevant traffic sign imagery.

The project demonstrated an end-to-end machine learning workflow:

```text
Data
 ↓
Model Training
 ↓
YOLO Object Detection
 ↓
TensorFlow Lite Conversion
 ↓
Raspberry Pi Deployment
 ↓
Real-Time Camera Inference
 ↓
OCR / Sign Interpretation
 ↓
Automated Vehicle Decision
```

## Future Improvements

Potential improvements include:

* Increasing the size and diversity of the training dataset
* Improving detection accuracy under different lighting and environmental conditions
* Improving OCR accuracy for different viewing angles and distances
* Optimizing inference and OCR processing speed
* Adding additional traffic-sign classes
* Improving distance estimation
* Implementing smoother vehicle control
* Adding more sophisticated autonomous navigation
* Evaluating model performance using additional metrics and test data

## Author

**Danny Santiago**

This project was developed as part of my academic work in Applied Computing and demonstrates experience with machine learning, computer vision, Python programming, OCR, model deployment, and integrating ML predictions with physical hardware.

## Repository

https://github.com/dssantii/Capstone

## Main Vehicle & ML Script

https://github.com/dssantii/Capstone/blob/main/RaspbotAIModelDemo.py
