#!/usr/bin/env python3
"""
Simple training script for signature detection using YOLO11.
This script provides a direct way to train the model with the existing dataset.
"""

import os
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO

def main():
    """Main training function using direct YOLO training."""
    
    # Set up paths
    project_root = Path(__file__).parent
    data_config = project_root / "data" / "signature" / "data.yaml"
    config_file = project_root / "config.yaml"
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract training parameters
    train_config = config.get('train', {})
    epochs = train_config.get('epochs', 100)
    batch_size = train_config.get('batch_size', 16)
    img_size = train_config.get('img_size', 640)
    lr = train_config.get('learning_rate', 0.01)
    
    model_config = config.get('model', {})
    model_size = model_config.get('size', 'n')
    
    print(f"🚀 Starting signature detection training...")
    print(f"📊 Training parameters:")
    print(f"   - Model: YOLO11{model_size}")
    print(f"   - Epochs: {epochs}")
    print(f"   - Batch size: {batch_size}")
    print(f"   - Image size: {img_size}")
    print(f"   - Learning rate: {lr}")
    print(f"   - Data config: {data_config}")
    
    # Initialize YOLO model
    model = YOLO(f"yolo11{model_size}.pt")
    
    # Start training
    try:
        results = model.train(
            data=str(data_config),
            epochs=epochs,
            batch=batch_size,
            imgsz=img_size,
            lr0=lr,
            project="runs/train",
            name="signature_detection",
            exist_ok=True,
            pretrained=True,
            optimizer="AdamW",
            verbose=True,
            seed=42,
            deterministic=True,
            single_cls=True,
            cos_lr=True,
            close_mosaic=10,
            resume=False,
            amp=True,
            fraction=1.0,
            patience=50,
            save_period=-1,
            cache=False,
            device=None,
            workers=8,
            plots=True,
            val=True
        )
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Results saved to: runs/train/signature_detection/")
        print(f"🎯 Best model saved as: runs/train/signature_detection/weights/best.pt")
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
