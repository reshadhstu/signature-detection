#!/usr/bin/env python3
"""
Simple inference script for signature detection.
This script makes predictions on new images using the trained model.
"""

import os
import sys
from pathlib import Path
import argparse
from ultralytics import YOLO

def main():
    """Main inference function."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Signature detection inference')
    parser.add_argument('--model', type=str, default=None,
                        help='Path to trained model (default: runs/train/signature_detection/weights/best.pt)')
    parser.add_argument('--source', type=str, required=True,
                        help='Source path (image, directory, or video)')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--save', action='store_true',
                        help='Save results')
    parser.add_argument('--show', action='store_true',
                        help='Show results')
    
    args = parser.parse_args()
    
    # Set up paths
    project_root = Path(__file__).parent
    
    # Default model path if not provided
    if args.model is None:
        model_path = project_root / "runs" / "train" / "signature_detection" / "weights" / "best.pt"
    else:
        model_path = Path(args.model)
    
    # Check if model exists
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print("Please train the model first using train_simple.py")
        sys.exit(1)
    
    # Check if source exists
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"❌ Source not found: {source_path}")
        sys.exit(1)
    
    print(f"🔍 Starting signature detection inference...")
    print(f"📊 Parameters:")
    print(f"   - Model: {model_path}")
    print(f"   - Source: {source_path}")
    print(f"   - Confidence threshold: {args.conf}")
    print(f"   - IoU threshold: {args.iou}")
    
    # Load trained model
    model = YOLO(str(model_path))
    
    # Make predictions
    try:
        results = model.predict(
            source=str(source_path),
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            show=args.show,
            project="runs/predict",
            name="signature_predictions",
            exist_ok=True,
            show_labels=True,
            show_conf=True,
            line_thickness=2
        )
        
        print(f"\n✅ Inference completed successfully!")
        print(f"📊 Results:")
        
        total_detections = 0
        images_with_detections = 0
        
        for i, result in enumerate(results):
            num_detections = len(result.boxes) if result.boxes is not None else 0
            total_detections += num_detections
            
            if num_detections > 0:
                images_with_detections += 1
                print(f"   - Image {i+1}: {num_detections} signature(s) detected")
                
                # Print detection details
                if result.boxes is not None:
                    for j, (conf, cls) in enumerate(zip(result.boxes.conf, result.boxes.cls)):
                        print(f"     • Detection {j+1}: confidence={conf:.3f}")
        
        print(f"\n📈 Summary:")
        print(f"   - Total images processed: {len(results)}")
        print(f"   - Images with detections: {images_with_detections}")
        print(f"   - Total detections: {total_detections}")
        
        if args.save:
            print(f"📁 Results saved to: runs/predict/signature_predictions/")
            
    except Exception as e:
        print(f"❌ Inference failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
