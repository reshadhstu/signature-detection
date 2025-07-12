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
    Wrapper class for YOLO model specifically designed for signature detection.
    
    This class provides a clean interface for training, evaluation, and inference
    with YOLO models for signature detection tasks.
    """
    
    def __init__(self, model_size: str = 'n', num_classes: int = 1, class_names: Optional[List[str]] = None):
        """
        Initialize the SignatureDetectionModel.
        
        Args:
            model_size: Size of the YOLO model ('n', 's', 'm', 'l', 'x')
            num_classes: Number of classes for detection
            class_names: List of class names
        """
        self.model_size = model_size
        self.num_classes = num_classes
        self.class_names = class_names or ['signature']
        self.model = None
        self.is_trained = False
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the YOLO model."""
        try:
            model_name = f"yolo11{self.model_size}.pt"
            self.model = YOLO(model_name)
            
            # Configure model for signature detection
            if hasattr(self.model.model, 'nc'):
                self.model.model.nc = self.num_classes
            if hasattr(self.model.model, 'names'):
                self.model.model.names = {i: name for i, name in enumerate(self.class_names)}
            
            logger.info(f"Model initialized: {model_name}")
            logger.info(f"Number of classes: {self.num_classes}")
            logger.info(f"Class names: {self.class_names}")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def train(self, 
              data_config: str, 
              epochs: int = 100, 
              batch_size: int = 16,
              img_size: int = 640,
              learning_rate: float = 0.01,
              patience: int = 50,
              save_dir: str = "runs/train",
              **kwargs) -> Dict[str, Any]:
        """
        Train the signature detection model.
        
        Args:
            data_config: Path to YOLO data configuration file
            epochs: Number of training epochs
            batch_size: Batch size for training
            img_size: Image size for training
            learning_rate: Learning rate
            patience: Early stopping patience
            save_dir: Directory to save training results
            **kwargs: Additional training parameters
            
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Starting model training...")
            logger.info(f"Data config: {data_config}")
            logger.info(f"Epochs: {epochs}, Batch size: {batch_size}")
            logger.info(f"Image size: {img_size}, Learning rate: {learning_rate}")
            
            # Training parameters
            train_params = {
                'data': data_config,
                'epochs': epochs,
                'batch': batch_size,
                'imgsz': img_size,
                'lr0': learning_rate,
                'patience': patience,
                'project': save_dir,
                'name': f'signature_detection_{self.model_size}',
                'exist_ok': True,
                'pretrained': True,
                'optimizer': 'AdamW',
                'verbose': True,
                'seed': 42,
                'deterministic': True,
                'single_cls': self.num_classes == 1,
                'rect': False,
                'cos_lr': True,
                'close_mosaic': 10,
                'resume': False,
                'amp': True,
                'fraction': 1.0,
                'profile': False,
                'freeze': None,
                'multi_scale': True,
                'overlap_mask': True,
                'mask_ratio': 4,
                'dropout': 0.0,
                'val': True,
                'split': 'val',
                'save_json': False,
                'save_hybrid': False,
                'conf': None,
                'iou': 0.7,
                'max_det': 300,
                'half': False,
                'dnn': False,
                'plots': True,
                'source': None,
                'show': False,
                'save_txt': False,
                'save_conf': False,
                'save_crop': False,
                'show_labels': True,
                'show_conf': True,
                'vid_stride': 1,
                'stream_buffer': False,
                'line_width': None,
                'visualize': False,
                'augment': False,
                'agnostic_nms': False,
                'classes': None,
                'retina_masks': False,
                'boxes': True,
                'format': 'torchscript',
                'keras': False,
                'optimize': False,
                'int8': False,
                'dynamic': False,
                'simplify': False,
                'opset': None,
                'workspace': 4,
                'nms': False,
                'lr_scheduler': 'auto',
                'save_period': -1,
                'cache': False,
                'device': None,
                'workers': 8,
                'image_weights': False,
                'rect': False,
                'multi_scale': False,
                'label_smoothing': 0.0,
                'box': 7.5,
                'cls': 0.5,
                'dfl': 1.5,
                'pose': 12.0,
                'kobj': 1.0,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'degrees': 0.0,
                'translate': 0.1,
                'scale': 0.5,
                'shear': 0.0,
                'perspective': 0.0,
                'flipud': 0.0,
                'fliplr': 0.5,
                'bgr': 0.0,
                'mosaic': 1.0,
                'mixup': 0.0,
                'copy_paste': 0.0,
                'auto_augment': 'randaugment',
                'erasing': 0.4,
                'crop_fraction': 1.0,
                **kwargs
            }
            
            # Start training
            results = self.model.train(**train_params)
            
            self.is_trained = True
            logger.info("Training completed successfully!")
            
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def evaluate(self, 
                 data_config: str, 
                 split: str = 'val',
                 conf_threshold: float = 0.25,
                 iou_threshold: float = 0.45,
                 **kwargs) -> Dict[str, Any]:
        """
        Evaluate the model on validation/test data.
        
        Args:
            data_config: Path to YOLO data configuration file
            split: Dataset split to evaluate on
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            **kwargs: Additional evaluation parameters
            
        Returns:
            Evaluation results dictionary
        """
        try:
            logger.info(f"Evaluating model on {split} split...")
            
            eval_params = {
                'data': data_config,
                'split': split,
                'conf': conf_threshold,
                'iou': iou_threshold,
                'verbose': True,
                'save_json': True,
                'save_hybrid': False,
                'half': False,
                'dnn': False,
                'plots': True,
                **kwargs
            }
            
            results = self.model.val(**eval_params)
            
            logger.info("Evaluation completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}")
            raise
    
    def predict(self, 
                source: str, 
                conf_threshold: float = 0.25,
                iou_threshold: float = 0.45,
                save_results: bool = False,
                save_dir: str = "runs/predict",
                **kwargs) -> List[Dict[str, Any]]:
        """
        Make predictions on new data.
        
        Args:
            source: Path to image/video/directory
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            save_results: Whether to save prediction results
            save_dir: Directory to save results
            **kwargs: Additional prediction parameters
            
        Returns:
            List of prediction results
        """
        try:
            logger.info(f"Making predictions on: {source}")
            
            predict_params = {
                'source': source,
                'conf': conf_threshold,
                'iou': iou_threshold,
                'save': save_results,
                'project': save_dir,
                'name': 'signature_predictions',
                'exist_ok': True,
                'show_labels': True,
                'show_conf': True,
                'visualize': False,
                'augment': False,
                'agnostic_nms': False,
                'classes': None,
                'retina_masks': False,
                'boxes': True,
                'line_width': None,
                'half': False,
                'dnn': False,
                'vid_stride': 1,
                'stream_buffer': False,
                **kwargs
            }
            
            results = self.model.predict(**predict_params)
            
            # Process results
            processed_results = []
            for result in results:
                processed_result = {
                    'image_path': result.path,
                    'image_shape': result.orig_shape,
                    'boxes': result.boxes.xyxy.cpu().numpy() if result.boxes is not None else [],
                    'confidences': result.boxes.conf.cpu().numpy() if result.boxes is not None else [],
                    'classes': result.boxes.cls.cpu().numpy() if result.boxes is not None else [],
                    'class_names': [self.class_names[int(cls)] for cls in result.boxes.cls.cpu().numpy()] if result.boxes is not None else []
                }
                processed_results.append(processed_result)
            
            logger.info(f"Predictions completed for {len(processed_results)} images")
            return processed_results
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def save_model(self, save_path: str) -> None:
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
        """
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            self.model.save(save_path)
            logger.info(f"Model saved to: {save_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def load_model(self, model_path: str) -> None:
        """
        Load a trained model.
        
        Args:
            model_path: Path to the saved model
        """
        try:
            self.model = YOLO(model_path)
            self.is_trained = True
            logger.info(f"Model loaded from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_size': self.model_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'is_trained': self.is_trained,
            'model_type': 'YOLOv11' if self.model else None
        }
    
    @classmethod
    def from_config(cls, config_path: str) -> 'SignatureDetectionModel':
        """
        Create model from configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            SignatureDetectionModel instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        model_config = config.get('model', {})
        model_size = model_config.get('architecture', 'yolo11n.pt').replace('yolo11', '').replace('.pt', '').replace('.yaml', '')
        
        return cls(
            model_size=model_size,
            num_classes=1,  # For signature detection
            class_names=['signature']
        )


def build_model(config_path: str) -> SignatureDetectionModel:
    """
    Build model from configuration file (legacy function for backward compatibility).
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SignatureDetectionModel instance
    """
    return SignatureDetectionModel.from_config(config_path)