#!/usr/bin/env python3
"""
Simple evaluation script for signature detection.
This script evaluates the trained model on the validation set.
"""
import os
import sys
from pathlib import Path
import yaml
from ultralytics import YOLO

def main():
    """Main evaluation function."""
    
    # Set up paths
    project_root = Path(__file__).parent
    data_config = project_root / "data" / "signature" / "data.yaml"
    config_file = project_root / "config.yaml"
    
    # Default model path (can be changed)
    model_path = project_root / "runs" / "train" / "signature_detection" / "weights" / "best.pt"
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using train_simple.py")
        sys.exit(1)
    
    # Load configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract evaluation parameters
    eval_config = config.get('eval', {})
    conf_thresh = eval_config.get('conf_thresh', 0.25)
    iou_thresh = eval_config.get('iou_thresh', 0.45)
    
    print(f"🔍 Starting signature detection evaluation...")
    print(f"📊 Evaluation parameters:")
    print(f"   - Model: {model_path}")
    print(f"   - Data config: {data_config}")
    print(f"   - Confidence threshold: {conf_thresh}")
    print(f"   - IoU threshold: {iou_thresh}")
    
    # Load trained model
    model = YOLO(str(model_path))
    
    # Start evaluation
    try:
        results = model.val(
            data=str(data_config),
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=True,
            save_json=True,
            save_hybrid=False,
            half=False,
            dnn=False,
            plots=True,
            project="runs/val",
            name="signature_evaluation",
            exist_ok=True
        )
        
        print("\n✅ Evaluation completed successfully!")
        print(f"📊 Results:")
        
        # Print key metrics
        if hasattr(results, 'box'):
            print(f"   - mAP@0.5: {results.box.map50:.4f}")
            print(f"   - mAP@0.5:0.95: {results.box.map:.4f}")
            print(f"   - Precision: {results.box.mp:.4f}")
            print(f"   - Recall: {results.box.mr:.4f}")
        
        print(f"📁 Detailed results saved to: runs/val/signature_evaluation/")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
