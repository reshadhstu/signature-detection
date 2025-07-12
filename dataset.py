import os
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np
import yaml
from torch.utils.data import Dataset
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignatureDataset(Dataset):
    """
    Custom dataset class for YOLO signature detection.
    
    This class handles loading images and their corresponding YOLO format labels,
    converting them to the appropriate format for training/evaluation.
    """
    
    def __init__(self, 
                 img_dir: str, 
                 label_dir: str, 
                 img_size: int = 640,
                 transforms: Optional[Any] = None,
                 validate_data: bool = True):
        """
        Initialize the SignatureDataset.
        
        Args:
            img_dir: Directory containing images
            label_dir: Directory containing YOLO format labels
            img_size: Target image size for resizing
            transforms: Optional transformations to apply
            validate_data: Whether to validate data integrity
        """
        self.img_dir = Path(img_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.transforms = transforms
        
        # Get all image files
        self.img_paths = self._get_image_paths()
        self.label_paths = self._get_label_paths()
        
        if validate_data:
            self._validate_dataset()
        
        logger.info(f"Dataset initialized with {len(self.img_paths)} images")
    
    def _get_image_paths(self) -> List[str]:
        """Get all image file paths."""
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        img_paths = []
        
        for ext in img_extensions:
            img_paths.extend(self.img_dir.glob(f'*{ext}'))
            img_paths.extend(self.img_dir.glob(f'*{ext.upper()}'))
        
        return sorted([str(p) for p in img_paths])
    
    def _get_label_paths(self) -> List[str]:
        """Get corresponding label file paths."""
        label_paths = []
        
        for img_path in self.img_paths:
            img_name = Path(img_path).stem
            label_path = self.label_dir / f"{img_name}.txt"
            label_paths.append(str(label_path))
        
        return label_paths
    
    def _validate_dataset(self) -> None:
        """Validate that all images have corresponding labels."""
        missing_labels = []
        
        for img_path, label_path in zip(self.img_paths, self.label_paths):
            if not Path(label_path).exists():
                missing_labels.append(label_path)
        
        if missing_labels:
            logger.warning(f"Missing labels for {len(missing_labels)} images")
            # Remove images without labels
            valid_pairs = [(img, lbl) for img, lbl in zip(self.img_paths, self.label_paths) 
                          if Path(lbl).exists()]
            self.img_paths = [pair[0] for pair in valid_pairs]
            self.label_paths = [pair[1] for pair in valid_pairs]
        
        logger.info(f"Dataset validation complete. {len(self.img_paths)} valid samples")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.img_paths)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, target) where target contains boxes and labels
        """
        try:
            img_path = self.img_paths[idx]
            label_path = self.label_paths[idx]
            
            # Load image
            img = self._load_image(img_path)
            
            # Load labels
            boxes, classes = self._load_labels(label_path, img.shape[:2])
            
            # Create target dictionary
            target = {
                "boxes": torch.tensor(boxes, dtype=torch.float32),
                "labels": torch.tensor(classes, dtype=torch.long)
            }
            
            # Apply transforms if provided
            if self.transforms:
                img, target = self.transforms(img, target)
            
            return img, target
            
        except Exception as e:
            logger.error(f"Error loading sample {idx}: {str(e)}")
            # Return a dummy sample to avoid crashing
            return self._get_dummy_sample()
    
    def _load_image(self, img_path: str) -> np.ndarray:
        """Load and preprocess image."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not load image: {img_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize if needed
        if self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size))
        
        return img
    
    def _load_labels(self, label_path: str, img_shape: Tuple[int, int]) -> Tuple[List[List[float]], List[int]]:
        """
        Load YOLO format labels and convert to absolute coordinates.
        
        Args:
            label_path: Path to the label file
            img_shape: Shape of the image (height, width)
            
        Returns:
            Tuple of (boxes, classes) where boxes are in [x1, y1, x2, y2] format
        """
        boxes, classes = [], []
        h, w = img_shape
        
        if not Path(label_path).exists():
            return boxes, classes
        
        try:
            with open(label_path, 'r') as f:
                for line in f.read().strip().splitlines():
                    if not line.strip():
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    
                    cls, x_c, y_c, bw, bh = map(float, parts)
                    
                    # Convert normalized coordinates to absolute
                    x1 = (x_c - bw/2) * w
                    y1 = (y_c - bh/2) * h
                    x2 = (x_c + bw/2) * w
                    y2 = (y_c + bh/2) * h
                    
                    # Ensure coordinates are within image bounds
                    x1 = max(0, min(x1, w))
                    y1 = max(0, min(y1, h))
                    x2 = max(0, min(x2, w))
                    y2 = max(0, min(y2, h))
                    
                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue
                    
                    boxes.append([x1, y1, x2, y2])
                    classes.append(int(cls))
        
        except Exception as e:
            logger.warning(f"Error loading labels from {label_path}: {str(e)}")
        
        return boxes, classes
    
    def _get_dummy_sample(self) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
        """Return a dummy sample for error cases."""
        img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        target = {
            "boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long)
        }
        return img, target
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get the distribution of classes in the dataset."""
        class_counts = {}
        
        for label_path in self.label_paths:
            if not Path(label_path).exists():
                continue
            
            try:
                with open(label_path, 'r') as f:
                    for line in f.read().strip().splitlines():
                        if not line.strip():
                            continue
                        
                        parts = line.split()
                        if len(parts) >= 1:
                            cls = int(float(parts[0]))
                            class_counts[cls] = class_counts.get(cls, 0) + 1
            except Exception as e:
                logger.warning(f"Error reading {label_path}: {str(e)}")
        
        return class_counts
    
    @staticmethod
    def create_from_config(config_path: str, split: str = 'train') -> 'SignatureDataset':
        """
        Create dataset from YOLO configuration file.
        
        Args:
            config_path: Path to the YOLO data configuration file
            split: Dataset split ('train' or 'val')
            
        Returns:
            SignatureDataset instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        dataset_root = Path(config['path'])
        img_dir = dataset_root / f"{split}/images"
        label_dir = dataset_root / f"{split}/labels"
        
        return SignatureDataset(
            img_dir=str(img_dir),
            label_dir=str(label_dir)
        )