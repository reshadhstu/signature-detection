#!/usr/bin/env python3
"""
Utility functions for signature detection project.

This module contains helper functions for logging, configuration management,
data visualization, and other common tasks.
"""

import logging
import os
import yaml
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import cv2
from datetime import datetime

def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Setup file handler if log_file is provided
    handlers = [console_handler]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the configuration
    """
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        logging.info(f"Configuration saved to: {save_path}")
    except Exception as e:
        logging.error(f"Failed to save configuration: {str(e)}")
        raise

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {str(e)}")
        raise

def create_directories(dirs: List[str]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        dirs: List of directory paths to create
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        logging.info(f"Directory created/verified: {dir_path}")

def visualize_dataset_stats(dataset, save_path: Optional[str] = None) -> None:
    """
    Visualize dataset statistics.
    
    Args:
        dataset: Dataset object
        save_path: Path to save the visualization
    """
    try:
        # Get class distribution
        class_dist = dataset.get_class_distribution()
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Plot class distribution
        plt.subplot(2, 2, 1)
        classes = list(class_dist.keys())
        counts = list(class_dist.values())
        
        plt.bar(classes, counts, color='skyblue', alpha=0.8)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        
        # Plot dataset size
        plt.subplot(2, 2, 2)
        plt.bar(['Total Samples'], [len(dataset)], color='lightgreen', alpha=0.8)
        plt.title('Dataset Size')
        plt.ylabel('Count')
        
        # Additional statistics
        plt.subplot(2, 2, 3)
        total_annotations = sum(counts)
        avg_annotations = total_annotations / len(dataset) if len(dataset) > 0 else 0
        
        stats = ['Total Annotations', 'Avg per Image']
        values = [total_annotations, avg_annotations]
        
        plt.bar(stats, values, color='coral', alpha=0.8)
        plt.title('Annotation Statistics')
        plt.ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Dataset statistics saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        logging.error(f"Failed to visualize dataset stats: {str(e)}")

def visualize_predictions(image_path: str, 
                         predictions: Dict[str, Any], 
                         save_path: Optional[str] = None,
                         show_labels: bool = True,
                         show_conf: bool = True) -> None:
    """
    Visualize predictions on an image.
    
    Args:
        image_path: Path to the image
        predictions: Prediction results dictionary
        save_path: Path to save the visualization
        show_labels: Whether to show class labels
        show_conf: Whether to show confidence scores
    """
    try:
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        
        # Draw bounding boxes
        boxes = predictions.get('boxes', [])
        confidences = predictions.get('confidences', [])
        class_names = predictions.get('class_names', [])
        
        for i, (box, conf, class_name) in enumerate(zip(boxes, confidences, class_names)):
            x1, y1, x2, y2 = box
            
            # Draw rectangle
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                               fill=False, edgecolor='red', linewidth=2)
            plt.gca().add_patch(rect)
            
            # Add label
            label = ""
            if show_labels:
                label += f"{class_name}"
            if show_conf:
                label += f" {conf:.2f}"
            
            if label:
                plt.text(x1, y1-5, label, fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                        color='white')
        
        plt.title(f'Predictions: {Path(image_path).name}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Prediction visualization saved to: {save_path}")
        
        plt.show()
        
    except Exception as e:
        logging.error(f"Failed to visualize predictions: {str(e)}")

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
        
    Returns:
        IoU value
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def apply_nms(boxes: List[List[float]], 
              scores: List[float], 
              iou_threshold: float = 0.5) -> List[int]:
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes.
    
    Args:
        boxes: List of bounding boxes
        scores: List of confidence scores
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of indices of boxes to keep
    """
    if not boxes:
        return []
    
    # Sort by score
    indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    keep = []
    while indices:
        # Take the box with highest score
        current = indices.pop(0)
        keep.append(current)
        
        # Remove boxes with high IoU
        indices = [i for i in indices 
                  if calculate_iou(boxes[current], boxes[i]) < iou_threshold]
    
    return keep

def create_submission_format(predictions: List[Dict[str, Any]], 
                           format_type: str = 'coco') -> Dict[str, Any]:
    """
    Create submission format for evaluation.
    
    Args:
        predictions: List of prediction dictionaries
        format_type: Format type ('coco', 'yolo', etc.)
        
    Returns:
        Formatted submission dictionary
    """
    if format_type == 'coco':
        submission = {
            'images': [],
            'annotations': [],
            'categories': [{'id': 0, 'name': 'signature'}]
        }
        
        ann_id = 1
        for img_id, pred in enumerate(predictions):
            submission['images'].append({
                'id': img_id,
                'file_name': Path(pred['image_path']).name,
                'width': pred['image_shape'][1],
                'height': pred['image_shape'][0]
            })
            
            boxes = pred.get('boxes', [])
            confidences = pred.get('confidences', [])
            classes = pred.get('classes', [])
            
            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box
                submission['annotations'].append({
                    'id': ann_id,
                    'image_id': img_id,
                    'category_id': int(cls),
                    'bbox': [x1, y1, x2-x1, y2-y1],
                    'area': (x2-x1) * (y2-y1),
                    'score': float(conf),
                    'iscrowd': 0
                })
                ann_id += 1
        
        return submission
    
    else:
        raise ValueError(f"Unsupported format type: {format_type}")

def save_results(results: Dict[str, Any], 
                save_path: str, 
                format_type: str = 'json') -> None:
    """
    Save results to file.
    
    Args:
        results: Results dictionary
        save_path: Path to save results
        format_type: Format type ('json', 'yaml')
    """
    try:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        if format_type == 'json':
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
        elif format_type == 'yaml':
            with open(save_path, 'w') as f:
                yaml.dump(results, f, default_flow_style=False, indent=2)
        else:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        logging.info(f"Results saved to: {save_path}")
        
    except Exception as e:
        logging.error(f"Failed to save results: {str(e)}")
        raise

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for reproducibility.
    
    Returns:
        Dictionary containing system information
    """
    import platform
    import torch
    
    info = {
        'timestamp': datetime.now().isoformat(),
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['gpu_names'] = [torch.cuda.get_device_name(i) 
                           for i in range(torch.cuda.device_count())]
    
    return info

def create_readme(project_info: Dict[str, Any], save_path: str) -> None:
    """
    Create README.md file for the project.
    
    Args:
        project_info: Project information dictionary
        save_path: Path to save README.md
    """
    readme_content = f"""# {project_info.get('title', 'Signature Detection Project')}

## Overview
{project_info.get('description', 'YOLO-based signature detection project')}

## Project Structure
```
{project_info.get('structure', 'Project structure will be updated')}
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python train.py --config config.yaml
```

### Evaluation
```bash
python evaluate.py --model path/to/model.pt --data path/to/data.yaml
```

### Inference
```bash
python predict.py --model path/to/model.pt --source path/to/images
```

## Results
{project_info.get('results', 'Results will be updated after training')}

## Model Performance
{project_info.get('performance', 'Performance metrics will be updated')}

## Citation
```bibtex
@article{{signature_detection_{datetime.now().year},
  title={{Signature Detection using YOLO}},
  author={{{project_info.get('author', 'Author')}}},
  year={{{datetime.now().year}}}
}}
```

## License
{project_info.get('license', 'GNU License')}
"""
    
    try:
        with open(save_path, 'w') as f:
            f.write(readme_content)
        logging.info(f"README.md created at: {save_path}")
    except Exception as e:
        logging.error(f"Failed to create README.md: {str(e)}")
        raise
