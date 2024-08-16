# Person and PPE Detection using YOLOv8

This project implements a two-stage detection system using YOLOv8 models for person detection in full images and Personal Protective Equipment (PPE) detection on cropped person images.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)

## Features

- Convert Pascal VOC annotations to YOLOv8 format
- Person detection using YOLOv8
- PPE detection on cropped person images
- Support for multiple PPE classes (hard-hat, gloves, mask, glasses, boots, vest, etc.)
- Visualization of detection results
- Flexible data processing pipeline for model training

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster training and inference)
- Basic understanding of object detection and YOLOv8

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/ayaankhan28/SyookAI.git
   cd person-ppe-detection

1. Install the required packages:
   ```sh
   pip install -r requirements.txt

## Project Structure 
    ```
    Syook/
      │
      ├── cropped_yolodataset/
      │   ├── test/
      │   ├── train/
      │   ├── valid/
      │   └── data.yaml
      │
      ├── datasets/
      │   ├── images/
      │   ├── labels/
      │   ├── classes.txt
      │
      ├── finaloutput/
      │
      └── yolodataset/
      ├──    test/
      │         ├── images/
      │         └── labels/
      ├──    train/
      │         ├── images/
      │         └── labels/
      ├──    valid/
      │         ├── images/
      │         └── labels/
      ├──    data.yaml
      └──    yolov8n.pt
    ├── best.pt            # Configuration file
    ├── bestppe.pt       # Project dependencies
    ├── pascaslVOC_to_yolo.py            # Configuration file
    ├── syook.ipynb  
    ├── syookcrop.ipynb  
    ├── ppe_dataset_generation.ipynb  
    └── inference.py               # This file
