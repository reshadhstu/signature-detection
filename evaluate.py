#!/usr/bin/env python3
"""
Evaluation script for signature detection model.

This script provides comprehensive evaluation capabilities including:
- Model performance metrics
- Visualization of results
- Error analysis
- Detailed reporting
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List

import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model import SignatureDetectionModel
from dataset import SignatureDataset
from utils import setup_logging, save_results, visualize_predictions, create_submission_format

# Configure logging
logger = logging.getLogger(__name__)

class SignatureEvaluator:
    """
    Evaluator class for signature detection model.
    
    This class provides comprehensive evaluation capabilities including
    metrics calculation, visualization, and detailed analysis.
    """
    
    def __init__(self, model_path: str, data_config: str, output_dir: str = "runs/eval"):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the trained model
            data_config: Path to data configuration file
            output_dir: Output directory for evaluation results
        """
        self.model_path = model_path
        self.data_config = data_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(self.output_dir / 'evaluation.log')
        
        # Initialize model and dataset
        self.model = None
        self.dataset = None
        self.results = {}
        
        self._load_model()
        self._load_dataset()
    
    def _load_model(self) -> None:
        """Load the trained model."""
        try:
            self.model = SignatureDetectionModel()
            self.model.load_model(self.model_path)
            logger.info(f"Model loaded from: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def _load_dataset(self) -> None:
        """Load the evaluation dataset."""
        try:
            self.dataset = SignatureDataset.create_from_config(
                config_path=self.data_config,
                split='val'
            )
            logger.info(f"Dataset loaded: {len(self.dataset)} samples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {str(e)}")
            raise
    
    def evaluate_model(self, 
                      conf_threshold: float = 0.25,
                      iou_threshold: float = 0.45) -> Dict[str, Any]:
        """
        Evaluate the model on the validation dataset.
        
        Args:
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Evaluation results dictionary
        """
        try:
            logger.info("Starting model evaluation...")
            
            # Run evaluation using YOLO's built-in evaluation
            results = self.model.evaluate(
                data_config=self.data_config,
                split='val',
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
            # Extract key metrics
            metrics = {
                'map50': results.box.map50 if hasattr(results, 'box') else 0.0,
                'map50_95': results.box.map if hasattr(results, 'box') else 0.0,
                'precision': results.box.mp if hasattr(results, 'box') else 0.0,
                'recall': results.box.mr if hasattr(results, 'box') else 0.0,
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold
            }
            
            self.results.update(metrics)
            
            logger.info(f"Evaluation completed!")
            logger.info(f"mAP@0.5: {metrics['map50']:.4f}")
            logger.info(f"mAP@0.5:0.95: {metrics['map50_95']:.4f}")
            logger.info(f"Precision: {metrics['precision']:.4f}")
            logger.info(f"Recall: {metrics['recall']:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def analyze_predictions(self, 
                          num_samples: int = 10,
                          conf_threshold: float = 0.25) -> None:
        """
        Analyze predictions on sample images.
        
        Args:
            num_samples: Number of samples to analyze
            conf_threshold: Confidence threshold for predictions
        """
        try:
            logger.info(f"Analyzing predictions on {num_samples} samples...")
            
            # Get sample images
            sample_indices = np.random.choice(len(self.dataset), 
                                            min(num_samples, len(self.dataset)), 
                                            replace=False)
            
            predictions_dir = self.output_dir / 'predictions'
            predictions_dir.mkdir(exist_ok=True)
            
            for i, idx in enumerate(sample_indices):
                # Get image and ground truth
                img_path = self.dataset.img_paths[idx]
                
                # Make prediction
                pred_results = self.model.predict(
                    source=img_path,
                    conf_threshold=conf_threshold,
                    save_results=False
                )
                
                if pred_results:
                    # Visualize prediction
                    save_path = predictions_dir / f'prediction_{i:03d}.png'
                    visualize_predictions(
                        image_path=img_path,
                        predictions=pred_results[0],
                        save_path=str(save_path)
                    )
            
            logger.info(f"Prediction analysis completed. Results saved to: {predictions_dir}")
            
        except Exception as e:
            logger.error(f"Prediction analysis failed: {str(e)}")
            raise
    
    def generate_confusion_matrix(self, 
                                conf_threshold: float = 0.25,
                                iou_threshold: float = 0.45) -> None:
        """
        Generate and save confusion matrix.
        
        Args:
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
        """
        try:
            logger.info("Generating confusion matrix...")
            
            # For signature detection (single class), we can create a simple TP/FP/FN analysis
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            
            for idx in range(len(self.dataset)):
                img_path = self.dataset.img_paths[idx]
                
                # Get ground truth
                _, target = self.dataset[idx]
                gt_boxes = target['boxes'].numpy()
                
                # Get predictions
                pred_results = self.model.predict(
                    source=img_path,
                    conf_threshold=conf_threshold,
                    save_results=False
                )
                
                if pred_results:
                    pred_boxes = pred_results[0]['boxes']
                else:
                    pred_boxes = []
                
                # Simple matching based on IoU
                matched_gt = set()
                for pred_box in pred_boxes:
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, gt_box in enumerate(gt_boxes):
                        if gt_idx in matched_gt:
                            continue
                        
                        iou = self._calculate_iou(pred_box, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou > iou_threshold:
                        true_positives += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        false_positives += 1
                
                # Count unmatched ground truth as false negatives
                false_negatives += len(gt_boxes) - len(matched_gt)
            
            # Calculate metrics
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            # Create confusion matrix visualization
            confusion_data = np.array([[true_positives, false_positives],
                                     [false_negatives, 0]])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(confusion_data, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Predicted Positive', 'Predicted Negative'],
                       yticklabels=['Actual Positive', 'Actual Negative'])
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Add metrics text
            metrics_text = f'Precision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1_score:.3f}'
            plt.figtext(0.02, 0.02, metrics_text, fontsize=10)
            
            # Save confusion matrix
            save_path = self.output_dir / 'confusion_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            # Update results
            self.results.update({
                'true_positives': true_positives,
                'false_positives': false_positives,
                'false_negatives': false_negatives,
                'precision_manual': precision,
                'recall_manual': recall,
                'f1_score': f1_score
            })
            
            logger.info(f"Confusion matrix saved to: {save_path}")
            logger.info(f"Manual metrics - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1_score:.3f}")
            
        except Exception as e:
            logger.error(f"Confusion matrix generation failed: {str(e)}")
            raise
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def generate_report(self) -> None:
        """Generate comprehensive evaluation report."""
        try:
            logger.info("Generating evaluation report...")
            
            # Create report
            report = {
                'model_info': {
                    'model_path': str(self.model_path),
                    'model_type': 'YOLO11',
                    'task': 'signature_detection'
                },
                'dataset_info': {
                    'data_config': str(self.data_config),
                    'num_samples': len(self.dataset),
                    'class_distribution': self.dataset.get_class_distribution()
                },
                'evaluation_results': self.results,
                'model_performance': self.model.get_model_info()
            }
            
            # Save report
            save_results(report, self.output_dir / 'evaluation_report.json', 'json')
            save_results(report, self.output_dir / 'evaluation_report.yaml', 'yaml')
            
            logger.info(f"Evaluation report saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Report generation failed: {str(e)}")
            raise
    
    def run_complete_evaluation(self, 
                              conf_threshold: float = 0.25,
                              iou_threshold: float = 0.45,
                              num_samples: int = 10) -> None:
        """
        Run complete evaluation pipeline.
        
        Args:
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold
            num_samples: Number of samples for analysis
        """
        try:
            logger.info("=== Starting Complete Evaluation ===")
            
            # Model evaluation
            self.evaluate_model(conf_threshold, iou_threshold)
            
            # Prediction analysis
            self.analyze_predictions(num_samples, conf_threshold)
            
            # Confusion matrix
            self.generate_confusion_matrix(conf_threshold, iou_threshold)
            
            # Generate report
            self.generate_report()
            
            logger.info("=== Complete Evaluation Finished ===")
            
        except Exception as e:
            logger.error(f"Complete evaluation failed: {str(e)}")
            raise


def main():
    """Main function to run the evaluation script."""
    parser = argparse.ArgumentParser(description='Evaluate YOLO11 signature detection model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to the trained model')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data configuration file')
    parser.add_argument('--output-dir', type=str, default='runs/eval',
                        help='Output directory for evaluation results')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold for predictions')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples for prediction analysis')
    
    args = parser.parse_args()
    
    try:
        # Create evaluator
        evaluator = SignatureEvaluator(
            model_path=args.model,
            data_config=args.data,
            output_dir=args.output_dir
        )
        
        # Run complete evaluation
        evaluator.run_complete_evaluation(
            conf_threshold=args.conf_threshold,
            iou_threshold=args.iou_threshold,
            num_samples=args.num_samples
        )
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()