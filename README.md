# Pumpkin Detection using OpenCV and YOLO v3
![image](https://github.com/user-attachments/assets/cf98c500-fddb-418d-a053-2d1335586f60)

This project demonstrates the use of YOLO v3 for object detection, specifically for detecting pumpkins in a video stream. The implementation uses OpenCV for video processing and deep learning inference.

## Table of Contents
I'll spare you the images collection and model training on google colab details, to jump directly on the detection step. 
- [Installation](#installation)
- [Usage](#usage)
- [Files](#files)


## Installation

To run this project, you will need to install the following dependencies:

- Python 3.11.9
- OpenCV 4.8.1.78
- NumPy 1.26.0

You can install the required Python packages using pip:

    pip install opencv-python numpy


## Usage

  git clone https://github.com/Gabriel-la-flamboyance/Pumpkin-Detection-OpenCV-YOLO-v3.git

  cd Pumpkin-Detection-OpenCV-YOLO-v3

  python pumpkinDetection.py


## Files

- detect_pumpkins.py: Main script for detecting pumpkins in a video stream.
- yolov3_testing.cfg: YOLO v3 configuration file.
- yolov3_training_final.weights: Trained weights file for YOLO v3. # COULDN'T PUSH IT HERE, PLEASE DOWNLOAD IT HERE --> https://drive.google.com/file/d/1P6PJe-5K9yO1YX9X7XR_78d29Mdmun7K/view?usp=sharing
- classes.txt: File containing the class names (e.g., "pumpkin").
- test5.mp4: Sample video file for detection.

