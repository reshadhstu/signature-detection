import logging
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from ultralytics import YOLO
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignatureDetectionModel:
    """
    YOLO model wrapper for signature detection.
    
    Provides methods for training, evaluation, and inference.
    """
    
    def __init__(self, model_name: str = 'yolo11n', config: Optional[Dict[str, Any]] = None):
        """
        Initialize the signature detection model.
        
        Args:
            model_name: Name of the YOLO model (e.g., 'yolo11n', 'yolo11s', 'yolo11m')
            config: Optional configuration dictionary
        """
        self.model_name = model_name
        self.config = config or {}
        self.model = None
        self.is_trained = False
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"SignatureDetectionModel initialized with {model_name}")
    
    def _initialize_model(self):
        """Initialize the YOLO model."""
        try:
            # Load pre-trained model
            model_path = f"{self.model_name}.pt"
            self.model = YOLO(model_path)
            logger.info(f"Model loaded: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def train(self, 
              data_yaml: str, 
              epochs: int = 50, 
              batch_size: int = 8,
              img_size: int = 640,
              lr0: float = 0.001,
              **kwargs) -> Dict[str, Any]:
        """
        Train the signature detection model.
        
        Args:
            data_yaml: Path to YOLO data configuration file
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Input image size
            lr0: Initial learning rate
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Default training parameters optimized for signature detection
        train_params = {
            'data': data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': img_size,
            'lr0': lr0,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,  # Box loss gain
            'cls': 0.5,  # Classification loss gain
            'dfl': 1.5,  # Distribution focal loss gain
            'optimizer': 'AdamW',
            'cos_lr': True,
            'close_mosaic': 10,
            'amp': True,
            'single_cls': True,  # Single class detection
            'patience': 15,
            'save_period': 10,
            'val': True,
            'plots': True,
            'save': True,
            'verbose': True,
            'seed': 42,
            'deterministic': True,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'workers': 4,
            'cache': False,
            'project': 'runs/train',
            'name': 'signature_detection',
            'exist_ok': True,
        }
        
        # Update with any additional parameters
        train_params.update(kwargs)
        
        logger.info("Starting training with optimized parameters...")
        logger.info(f"Training parameters: {train_params}")
        
        try:
            # Train the model
            results = self.model.train(**train_params)
            self.is_trained = True
            
            logger.info("Training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate(self, 
                 data_yaml: str = None, 
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model on validation data.
        
        Args:
            data_yaml: Path to YOLO data configuration file
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation results dictionary
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Default evaluation parameters
        eval_params = {
            'conf': conf_threshold,
            'iou': iou_threshold,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'plots': True,
            'save_json': True,
            'verbose': True,
        }
        
        # Add data path if provided
        if data_yaml:
            eval_params['data'] = data_yaml
        
        # Update with additional parameters
        eval_params.update(kwargs)
        
        logger.info("Starting evaluation...")
        
        try:
            # Run evaluation
            results = self.model.val(**eval_params)
            
            # Extract key metrics
            metrics = {
                'mAP@0.5': float(results.box.map50),
                'mAP@0.5:0.95': float(results.box.map),
                'precision': float(results.box.mp),
                'recall': float(results.box.mr),
            }
            
            logger.info("Evaluation completed successfully!")
            logger.info(f"Metrics: {metrics}")
            
            return {
                'results': results,
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def predict(self, 
                source: str, 
                conf_threshold: float = 0.25,
                iou_threshold: float = 0.45,
                save_results: bool = False,
                **kwargs) -> List[Dict[str, Any]]:
        """
        Run inference on images.
        
        Args:
            source: Path to image(s) or directory
            conf_threshold: Confidence threshold for predictions
            iou_threshold: IoU threshold for NMS
            save_results: Whether to save prediction results
            **kwargs: Additional prediction parameters
            
        Returns:
            List of prediction results
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        # Default prediction parameters
        pred_params = {
            'conf': conf_threshold,
            'iou': iou_threshold,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'save': save_results,
            'verbose': True,
        }
        
        # Update with additional parameters
        pred_params.update(kwargs)
        
        logger.info(f"Running inference on: {source}")
        
        try:
            # Run prediction
            results = self.model(source, **pred_params)
            
            # Process results
            predictions = []
            for result in results:
                pred_data = {
                    'image_path': result.path,
                    'detections': [],
                    'confidence_scores': [],
                    'boxes': []
                }
                
                if len(result.boxes) > 0:
                    for box in result.boxes:
                        pred_data['detections'].append('signature')
                        pred_data['confidence_scores'].append(float(box.conf))
                        pred_data['boxes'].append(box.xyxy.cpu().numpy().tolist())
                
                predictions.append(pred_data)
            
            logger.info(f"Inference completed. Found {len(predictions)} results.")
            return predictions
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def save_model(self, save_path: str):
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        try:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save model
            self.model.save(save_path)
            logger.info(f"Model saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, model_path: str):
        """
        Load a pre-trained model.
        
        Args:
            model_path: Path to the model file
        """
        try:
            self.model = YOLO(model_path)
            self.is_trained = True
            logger.info(f"Model loaded from: {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def export_model(self, 
                     format: str = 'onnx',
                     **kwargs) -> str:
        """
        Export model to different formats.
        
        Args:
            format: Export format ('onnx', 'torchscript', 'tflite', etc.)
            **kwargs: Additional export parameters
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise ValueError("Model not initialized")
        
        logger.info(f"Exporting model to {format} format...")
        
        try:
            exported_path = self.model.export(format=format, **kwargs)
            logger.info(f"Model exported to: {exported_path}")
            return exported_path
            
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary containing model information
        """
        if self.model is None:
            return {"error": "Model not initialized"}
        
        try:
            # Get model info
            info = {
                'model_name': self.model_name,
                'is_trained': self.is_trained,
                'device': next(self.model.model.parameters()).device if hasattr(self.model, 'model') else 'unknown',
                'num_classes': 1,  # Signature detection is single class
                'input_size': 640,
                'model_size': f"~{self._get_model_size()}MB"
            }
            
            return info
            
        except Exception as e:
            logger.error(f"Failed to get model info: {str(e)}")
            return {"error": str(e)}
    
    def _get_model_size(self) -> float:
        """Estimate model size in MB."""
        if self.model_name == 'yolo11n':
            return 6.0
        elif self.model_name == 'yolo11s':
            return 22.0
        elif self.model_name == 'yolo11m':
            return 50.0
        elif self.model_name == 'yolo11l':
            return 87.0
        elif self.model_name == 'yolo11x':
            return 143.0
        else:
            return 25.0  # Default estimate