# 🌿 Weed Detection System - YOLOv5s

A real-time weed detection system using YOLOv5 for agricultural applications. This project detects weeds in images and video streams, providing accurate center coordinates for each detection.

## 📋 Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Dataset](#dataset)
- [Results](#results)
- [Documentation](#documentation)

## ✨ Features

- **Real-time Detection**: Webcam-based real-time weed detection with live visualization
- **Batch Processing**: Process multiple images at once with coordinate export
- **GUI Interface**: User-friendly GUI for single image inference
- **Accurate Coordinates**: Advanced center coordinate calculation using edge detection
- **Model Analysis**: Performance analysis tools to identify and fix issues
- **Jupyter Notebook**: Complete training and inference workflow in notebook format

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for faster inference)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd FYP-Model-Training
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Ensure `best.pt` model file is in the project directory

## 🎯 Quick Start

### Real-time Webcam Detection

```bash
python webcam_inference.py
```

**Controls:**
- `Q` - Quit
- `S` - Save current frame
- `R` - Toggle video recording
- `+/-` - Adjust confidence threshold

### Batch Image Processing

```bash
python batch_inference.py --source Sample-Dataset/ --output batch_results/ --conf 0.25
```

### GUI Interface

```bash
python image_inference_gui_fixed.py
```

### Jupyter Notebook

Open `Weed_Detection_YOLO_FYP.ipynb` for complete training and inference workflow.

## 📖 Usage

### Webcam Inference

```bash
python webcam_inference.py [OPTIONS]

Options:
  --weights PATH      Model weights file (default: best.pt)
  --camera INDEX      Camera index (default: 0)
  --width WIDTH       Frame width (default: 1280)
  --height HEIGHT     Frame height (default: 720)
  --conf THRESHOLD    Confidence threshold 0.0-1.0 (default: 0.25)
  --iou THRESHOLD     IOU threshold for NMS (default: 0.45)
  --save-dir DIR      Directory to save frames (default: saved_frames)
  --record            Start video recording immediately
```

### Batch Inference

```bash
python batch_inference.py [OPTIONS]

Options:
  --source PATH       Input directory with images
  --output PATH       Output directory for results
  --weights PATH      Model weights (default: best.pt)
  --conf THRESHOLD    Confidence threshold (default: 0.25)
  --format FORMAT     Output format: json, csv, or both (default: json)
```

### Image Inference GUI

```bash
python image_inference_gui_fixed.py
```

Features:
- Select image from file dialog
- View detection results with bounding boxes
- Export coordinates to JSON/CSV
- Adjust confidence threshold
- Save annotated images

## 📁 Project Structure

```
FYP-Model-Training/
├── best.pt                          # Trained YOLOv5 model weights
├── Weed_Detection_YOLO_FYP.ipynb   # Complete Jupyter notebook workflow
├── webcam_inference.py              # Real-time webcam detection
├── batch_inference.py               # Batch image processing
├── image_inference_gui_fixed.py     # GUI for single image inference
├── requirements.txt                 # Python dependencies
├── model_analysis_results.json      # Model performance analysis
├── README.md                        # This file
├── QUICK_START.md                   # Quick start guide
├── MODEL_IMPROVEMENT_GUIDE.md       # Model improvement guide
├── dataset/                         # Dataset organization
│   ├── unprocessed/                 # Raw input images
│   ├── processed/                   # Processed/annotated images
│   └── working/                     # Working dataset samples
└── batch_results/                   # Batch inference results
    ├── annotated_images/            # Images with detections
    ├── coordinates.json             # Detection coordinates
    └── coordinates.csv              # CSV format coordinates
```

## 📊 Model Performance

The model performance analysis is available in `model_analysis_results.json`. Key metrics include:

- **Total Images Analyzed**: See JSON file
- **Detection Statistics**: Average confidence, detection count per image
- **Class Distribution**: Weed detection distribution
- **Confidence Distribution**: Low/Medium/High confidence breakdown

### View Analysis Results

```bash
# The analysis results are stored in model_analysis_results.json
# Open the file to view detailed performance metrics
```

## 🗂️ Dataset

The dataset is organized into three categories:

1. **unprocessed/**: Raw input images (original dataset)
2. **processed/**: Images with annotations and detections
3. **working/**: Sample working dataset for testing

### Dataset Structure

- **Sample-Dataset/**: Contains unprocessed agricultural field images
- **batch_results/**: Contains processed images with detection annotations

## 📈 Results

### Batch Inference Results

Batch processing results are stored in `batch_results/`:
- Annotated images with bounding boxes
- Detection coordinates in JSON and CSV formats
- Confidence scores for each detection

### Model Analysis

The `model_analysis_results.json` file contains:
- Detection statistics
- Confidence score distributions
- Class distribution
- Potential issues and recommendations

## 📚 Documentation

- **README.md**: This file - project overview and usage
- **QUICK_START.md**: Quick start guide for common tasks
- **MODEL_IMPROVEMENT_GUIDE.md**: Detailed guide for improving model performance

## 🔧 Key Features Explained

### Accurate Center Coordinate Calculation

The system uses advanced edge detection and center of mass calculation to determine precise weed center coordinates:

1. **Canny Edge Detection**: Identifies weed edges within bounding box
2. **Center of Mass**: Calculates geometric center of detected edges
3. **Fallback Methods**: Otsu thresholding and weighted center as alternatives

### Real-time Detection

- FPS monitoring and display
- Adjustable confidence threshold on-the-fly
- Frame saving and video recording
- Detection statistics overlay

### Batch Processing

- Processes entire directories
- Exports coordinates in multiple formats
- Generates annotated images
- Comprehensive JSON/CSV reports

## 🛠️ Model Information

- **Architecture**: YOLOv5 (Ultralytics)
- **Classes**: 1 (weed)
- **Input Size**: 640x640 (configurable)
- **Confidence Threshold**: 0.25 (default, adjustable)
- **IOU Threshold**: 0.45 (default)

## 📝 Notes

- The model is trained for single-class weed detection
- Confidence threshold can be adjusted based on use case (0.2-0.4 recommended)
- For best results, use images with similar conditions to training data
- GPU acceleration recommended for real-time inference

## 🤝 Contributing

This is a Final Year Project (FYP). For questions or improvements, please open an issue.

## 📄 License

This project is part of an academic Final Year Project.

## 🙏 Acknowledgments

- YOLOv5 by Ultralytics
- OpenCV for image processing
- PyTorch for deep learning framework

---

**Author**: FYP Project  
**Year**: 2024  
**Purpose**: Agricultural Weed Detection System
