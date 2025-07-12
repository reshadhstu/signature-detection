# 🎯 Signature Detection with YOLO11

A high-performance signature detection system using YOLO11 architecture, achieving **99%+ accuracy** on signature detection tasks.

## � Performance Highlights
- **mAP@0.5**: 99.13%
- **mAP@0.5:0.95**: 91.14%
- **Precision**: 99.8%
- **Recall**: 94.3%
- **Model Size**: ~6MB (YOLO11n)
- **Inference Speed**: ~33ms per image

## 📁 Repository Structure

```
signature-detection/
├── 📂 data/signature/                # Local dataset
│   ├── train/images/ (143 files)
│   ├── train/labels/ (143 files)  
│   ├── val/images/ (35 files)
│   ├── val/labels/ (35 files)
│   └── data.yaml
├── 📂 signature_detection/           # Training output folder
│   └── train/weights/
│       ├── best.pt                  # Trained model
│       └── last.pt
├── 📂 signature-detection-dataset/   # Kaggle dataset folder
├── 📄 train.py                      # Main training script
├── 📄 signature_detection_kaggle.ipynb # Kaggle notebook
├── 📄 test_model.py                 # Model testing
└── 📄 requirements.txt
```

## 🚀 Quick Start

### Local Training
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py
```

### Kaggle Training
1. Upload `signature-detection-dataset/` to Kaggle as dataset
2. Run `signature_detection_kaggle.ipynb` notebook

## 🔧 Key Features

- **Optimized Training**: Fixed parameters for consistent convergence
- **Dual Environment**: Works locally and on Kaggle
- **Production Ready**: Exported models in multiple formats
- **Comprehensive Testing**: Built-in validation and testing scripts

## 📊 Training Configuration

The model uses optimized parameters that ensure consistent training success:

- **Epochs**: 50 (increased for better convergence)
- **Batch Size**: 8 (smaller for stability)
- **Learning Rate**: 0.001 (lower for fine-tuning)
- **Optimizer**: AdamW with cosine scheduling
- **Loss Weights**: Optimized for single-class detection

## 🎯 Model Performance

| Metric | Value | Status |
|--------|-------|--------|
| mAP@0.5 | 99.13% | ✅ Excellent |
| mAP@0.5:0.95 | 91.14% | ✅ Excellent |
| Precision | 99.8% | ✅ Excellent |
| Recall | 94.3% | ✅ Excellent |

## 📝 Dataset Information

### Local Dataset (`data/signature/`)
- **Purpose**: Local development and training
- **Images**: 178 total (143 train + 35 val)
- **Format**: YOLO format with normalized coordinates
- **Classes**: 1 (signature)

### Kaggle Dataset (`signature-detection-dataset/`)
- **Purpose**: Kaggle platform upload
- **Structure**: Same as local but with Kaggle-specific paths
- **Upload**: Ready for Kaggle dataset creation

## 🔍 Testing the Model

```python
from ultralytics import YOLO

# Load trained model
model = YOLO('signature_detection/train/weights/best.pt')

# Run inference
results = model('path/to/image.jpg')

# Display results
results[0].show()
```

## 📚 Documentation

- `FOLDER_STRUCTURE.md` - Detailed folder structure guide
- `TRAINING_SUCCESS_REPORT.md` - Complete training results
- See notebook for step-by-step Kaggle implementation

## 🎉 Success Story

This project successfully resolved the common **0.0000 mAP** issue in YOLO training by:

1. **Optimizing Training Parameters**: Proper learning rate, batch size, and epochs
2. **Loss Function Tuning**: Balanced box, classification, and DFL losses  
3. **Data Configuration**: Fixed path issues and encoding problems
4. **Training Strategy**: Single-class detection with proper convergence

**Result**: From 0% to 99%+ accuracy! 🚀
│   ├── train/               # Training results
│   ├── val/                 # Validation results
│   └── predict/             # Prediction results
│
└── myassignmentenv/         # Python virtual environment
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 5GB+ free disk space

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/reshadhstu/signature-detection.git
cd signature-detection
```

2. **Create and activate virtual environment:**
```bash
python -m venv myassignmentenv
# On Windows:
myassignmentenv\Scripts\activate
# On Linux/Mac:
source myassignmentenv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Dataset Setup

The project uses a signature detection dataset with YOLO format annotations:

1. **Download the dataset** (if not already present):
   - Images should be in `data/signature/train/images/` and `data/signature/val/images/`
   - Labels should be in `data/signature/train/labels/` and `data/signature/val/labels/`

2. **Verify dataset structure:**
```bash
python -c "from dataset import SignatureDataset; ds = SignatureDataset.create_from_config('data/signature/data.yaml', 'train'); print(f'Dataset loaded: {len(ds)} samples')"
```

## 🔧 Configuration

The project uses a comprehensive YAML configuration file (`config.yaml`) that controls all aspects of training, evaluation, and inference. Key sections include:

- **Data**: Dataset paths and augmentation settings
- **Model**: Architecture and model parameters
- **Training**: Training hyperparameters and strategies
- **Evaluation**: Evaluation metrics and thresholds
- **Inference**: Prediction settings and output options

## 💻 Usage

### Training

Train the signature detection model:

```bash
# Basic training
python train.py --config config.yaml

# Custom output directory
python train.py --config config.yaml --output-dir runs/custom_training

# Resume training
python train.py --config config.yaml --resume path/to/checkpoint.pt

# Specify device
python train.py --config config.yaml --device cuda:0
```

### Evaluation

Evaluate the trained model:

```bash
# Evaluate on validation set
python evaluate.py --model runs/train/best_model.pt --data data/signature/data.yaml

# Custom evaluation parameters
python evaluate.py --model path/to/model.pt --data data/signature/data.yaml \
                  --conf-threshold 0.3 --iou-threshold 0.5 \
                  --output-dir runs/custom_eval
```

### Inference

Make predictions on new images:

```bash
# Single image prediction
python predict.py --model runs/train/best_model.pt --source path/to/image.jpg

# Directory prediction
python predict.py --model runs/train/best_model.pt --source path/to/images/ \
                  --save-results

# Video prediction
python predict.py --model runs/train/best_model.pt --source path/to/video.mp4 \
                  --save-results --frame-skip 2

# Custom parameters
python predict.py --model path/to/model.pt --source path/to/images/ \
                  --conf-threshold 0.3 --iou-threshold 0.5 \
                  --output-dir runs/custom_predict
```

## 📊 Model Performance

### Training Results

The model is trained on signature detection dataset with the following performance:

| Metric | Value |
|--------|-------|
| mAP@0.5 | TBD after training |
| mAP@0.5:0.95 | TBD after training |
| Precision | TBD after training |
| Recall | TBD after training |
| F1-Score | TBD after training |

### Model Architecture

- **Base Model**: YOLO11n (nano variant for efficiency)
- **Input Size**: 640x640 pixels
- **Classes**: 1 (signature)
- **Parameters**: ~2.6M parameters
- **Inference Speed**: ~1.5ms per image (GPU)

## 🛠️ Advanced Features

### Custom Dataset

To use your own signature dataset:

1. **Prepare your data** in YOLO format:
   - Images in JPG/PNG format
   - Labels in TXT format with normalized coordinates
   - One label file per image

2. **Update data configuration**:
   ```yaml
   # data/your_dataset/data.yaml
   path: ../data/your_dataset
   train: train/images
   val: val/images
   names:
     0: signature
   ```

3. **Update main configuration**:
   ```yaml
   # config.yaml
   data:
     config_path: "data/your_dataset/data.yaml"
   ```

### Model Customization

To use different YOLO11 variants:

```yaml
# config.yaml
model:
  size: "s"  # Options: n, s, m, l, x
  architecture: "yolo11s.yaml"
  pretrained: "yolo11s.pt"
```

### Hyperparameter Tuning

Key hyperparameters to tune:

```yaml
# config.yaml
train:
  learning_rate: 0.01      # Learning rate
  batch_size: 16           # Batch size
  epochs: 100              # Training epochs
  patience: 50             # Early stopping patience
  
  # Data augmentation
  data:
    augmentation:
      hsv_h: 0.015          # HSV-Hue augmentation
      hsv_s: 0.7            # HSV-Saturation augmentation
      hsv_v: 0.4            # HSV-Value augmentation
      degrees: 0.0          # Rotation augmentation
      translate: 0.1        # Translation augmentation
      scale: 0.5            # Scale augmentation
      fliplr: 0.5           # Horizontal flip probability
```

## 📈 Results and Visualization

The project provides comprehensive visualization capabilities:

### Training Curves
- Loss curves (box, objectness, classification)
- Metric curves (mAP, precision, recall)
- Learning rate schedule

### Evaluation Results
- Confusion matrix
- Precision-Recall curves
- Sample predictions with ground truth
- Error analysis

### Prediction Outputs
- Annotated images with bounding boxes
- Confidence scores and class labels
- Batch processing results
- Video annotations

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce batch size in config.yaml
   - Use smaller model variant (yolo11n instead of yolo11s)
   - Enable gradient accumulation

2. **Poor Training Performance**:
   - Check dataset quality and annotations
   - Increase training epochs
   - Adjust learning rate and augmentation parameters

3. **Low Detection Accuracy**:
   - Collect more training data
   - Improve annotation quality
   - Tune confidence and IoU thresholds

### Performance Optimization

1. **Training Speed**:
   - Use multiple GPUs with distributed training
   - Enable mixed precision training (AMP)
   - Use larger batch sizes if memory allows

2. **Inference Speed**:
   - Use TensorRT for deployment
   - Export to ONNX format
   - Use smaller model variants

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLO11 implementation
- [PyTorch](https://pytorch.org/) for deep learning framework
- [OpenCV](https://opencv.org/) for computer vision utilities
- Community contributors and dataset providers

## 📞 Contact

For questions, issues, or collaboration opportunities:

- **Email**: reshad.hstu@gmail.com
- **GitHub Issues**: [Create an issue](https://github.com/reshadhstu/signature-detection/issues)
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/reshadhstu/)

---

**Note**: This is a educational/research project. For production use, please ensure proper testing and validation with your specific use case and data.

## 🔄 Updates and Roadmap

### Recent Updates
- ✅ Modular architecture implementation
- ✅ Comprehensive configuration system
- ✅ Advanced evaluation metrics
- ✅ Video inference support
- ✅ Batch processing capabilities

### Upcoming Features
- 🔄 Multi-class signature detection
- 🔄 Real-time inference optimization
- 🔄 Web interface for easy usage
- 🔄 Docker containerization
- 🔄 Cloud deployment guides
