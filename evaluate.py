#!/usr/bin/env python3
"""
Evaluation script for signature detection model.
This script evaluates the trained model and provides detailed metrics.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model import SignatureDetectionModel
from dataset import SignatureDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

def evaluate_model(model_path: str, config: dict, save_plots: bool = True):
    """
    Evaluate the trained model.
    
    Args:
        model_path: Path to the trained model
        config: Configuration dictionary
        save_plots: Whether to save evaluation plots
    """
    try:
        # Load model
        logger.info(f"Loading model from: {model_path}")
        model = SignatureDetectionModel()
        model.load_model(model_path)
        
        # Get evaluation configuration
        eval_config = config['evaluation']
        data_config = config['data']
        
        # Run evaluation
        logger.info("Running model evaluation...")
        results = model.evaluate(
            data_yaml=data_config['config_path'],
            conf_threshold=eval_config['conf_threshold'],
            iou_threshold=eval_config['iou_threshold'],
            save_json=eval_config['save_json'],
            plots=save_plots
        )
        
        # Print detailed results
        print_evaluation_results(results)
        
        # Save evaluation report
        if eval_config['save_report']:
            save_evaluation_report(results, config)
        
        # Test on individual images
        if eval_config['test_individual_images']:
            test_individual_images(model, data_config, eval_config)
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

def print_evaluation_results(results: dict):
    """Print detailed evaluation results."""
    metrics = results['metrics']
    
    print("\n" + "="*60)
    print("📊 EVALUATION RESULTS")
    print("="*60)
    
    print(f"🎯 Key Metrics:")
    print(f"   mAP@0.5:     {metrics['mAP@0.5']:.4f} ({metrics['mAP@0.5']*100:.2f}%)")
    print(f"   mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f} ({metrics['mAP@0.5:0.95']*100:.2f}%)")
    print(f"   Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"   Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    
    # Performance assessment
    print(f"\n🏆 Performance Assessment:")
    if metrics['mAP@0.5'] >= 0.9:
        print("   ✅ EXCELLENT - Model performance is outstanding!")
    elif metrics['mAP@0.5'] >= 0.7:
        print("   ✅ GOOD - Model performance is satisfactory")
    elif metrics['mAP@0.5'] >= 0.5:
        print("   ⚠️  FAIR - Model performance needs improvement")
    else:
        print("   ❌ POOR - Model performance is inadequate")
    
    print("="*60)

def test_individual_images(model: SignatureDetectionModel, data_config: dict, eval_config: dict):
    """Test model on individual validation images."""
    try:
        # Load data configuration
        with open(data_config['config_path'], 'r', encoding='utf-8') as f:
            data_yaml = yaml.safe_load(f)
        
        # Get validation images
        val_images_dir = Path(data_yaml['path']) / "val/images"
        val_images = list(val_images_dir.glob("*.jpg"))[:eval_config['num_test_images']]
        
        if not val_images:
            logger.warning("No validation images found")
            return
        
        print(f"\n🔍 Testing on {len(val_images)} individual images:")
        print("-" * 50)
        
        total_detections = 0
        images_with_detections = 0
        
        for i, img_path in enumerate(val_images, 1):
            print(f"{i:2d}. {img_path.name}")
            
            # Run prediction
            predictions = model.predict(
                source=str(img_path),
                conf_threshold=eval_config['conf_threshold'],
                iou_threshold=eval_config['iou_threshold']
            )
            
            if predictions and len(predictions) > 0:
                pred = predictions[0]
                num_detections = len(pred['detections'])
                
                if num_detections > 0:
                    total_detections += num_detections
                    images_with_detections += 1
                    
                    print(f"     ✅ {num_detections} signature(s) detected")
                    for j, conf in enumerate(pred['confidence_scores']):
                        print(f"        - Signature {j+1}: {conf:.3f} confidence")
                else:
                    print(f"     ❌ No signatures detected")
            else:
                print(f"     ❌ No signatures detected")
        
        print("-" * 50)
        print(f"📈 Summary:")
        print(f"   Images tested: {len(val_images)}")
        print(f"   Images with detections: {images_with_detections}")
        print(f"   Total detections: {total_detections}")
        print(f"   Detection rate: {images_with_detections/len(val_images)*100:.1f}%")
        
    except Exception as e:
        logger.error(f"Individual image testing failed: {str(e)}")

def save_evaluation_report(results: dict, config: dict):
    """Save evaluation report to file."""
    try:
        from datetime import datetime
        
        report = {
            'model_architecture': config['model']['architecture'],
            'evaluation_timestamp': datetime.now().isoformat(),
            'metrics': results['metrics'],
            'configuration': {
                'conf_threshold': config['evaluation']['conf_threshold'],
                'iou_threshold': config['evaluation']['iou_threshold'],
                'dataset': config['data']['config_path']
            }
        }
        
        # Save report
        report_path = Path("evaluation_report.yaml")
        with open(report_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False)
        
        logger.info(f"Evaluation report saved to: {report_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save evaluation report: {str(e)}")

def compare_models(model_paths: list, config: dict):
    """Compare multiple models."""
    print("\n📊 MODEL COMPARISON")
    print("=" * 60)
    
    results = {}
    
    for model_path in model_paths:
        if not Path(model_path).exists():
            logger.warning(f"Model not found: {model_path}")
            continue
        
        print(f"\nEvaluating: {model_path}")
        model = SignatureDetectionModel()
        model.load_model(model_path)
        
        eval_results = model.evaluate(
            data_yaml=config['data']['config_path'],
            conf_threshold=config['evaluation']['conf_threshold'],
            iou_threshold=config['evaluation']['iou_threshold']
        )
        
        results[model_path] = eval_results['metrics']
    
    # Print comparison
    if results:
        print("\n🏆 COMPARISON RESULTS:")
        print("-" * 60)
        print(f"{'Model':<30} {'mAP@0.5':<12} {'mAP@0.5:0.95':<15} {'Precision':<12} {'Recall':<12}")
        print("-" * 60)
        
        for model_path, metrics in results.items():
            model_name = Path(model_path).name
            print(f"{model_name:<30} {metrics['mAP@0.5']:<12.4f} {metrics['mAP@0.5:0.95']:<15.4f} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f}")

def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate signature detection model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file')
    parser.add_argument('--compare', nargs='+', type=str,
                       help='Compare multiple models')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip saving plots')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Check if model exists
    if not Path(args.model).exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)
    
    try:
        if args.compare:
            # Compare multiple models
            compare_models([args.model] + args.compare, config)
        else:
            # Evaluate single model
            results = evaluate_model(args.model, config, save_plots=not args.no_plots)
            
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
