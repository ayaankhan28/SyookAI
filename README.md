# Person and PPE Detection using YOLOv8

This project implements a two-stage detection system using YOLOv8 models for person detection in full images and Personal Protective Equipment (PPE) detection on cropped person images.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

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
    person-ppe-detection/
    ├── data/
    │   ├── raw/                # Raw dataset
    │   ├── processed/          # Processed dataset for training
    │   └── annotations/        # Converted YOLO annotations
    ├── models/
    │   ├── person_detection/   # Person detection model weights
    │   └── ppe_detection/      # PPE detection model weights
    ├── scripts/
    │   ├── pascal_to_yolo.py   # Script to convert Pascal VOC to YOLO format
    │   ├── process_data.py     # Script to process data for PPE model training
    │   └── train.py            # Training script
    ├── src/
    │   ├── data_utils.py       # Utilities for data processing
    │   ├── inference.py        # Inference pipeline
    │   └── visualization.py    # Utilities for result visualization
    ├── config.yaml             # Configuration file
    ├── requirements.txt        # Project dependencies
    └── README.md               # This file
