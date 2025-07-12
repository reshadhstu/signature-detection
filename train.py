#!/usr/bin/env python3
"""
Training script for signature detection using YOLO11.

This script provides a complete training pipeline for signature detection,
including data loading, model training, validation, and result visualization.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model import SignatureDetectionModel
from dataset import SignatureDataset
from utils import setup_logging, save_config, create_directories

# Configure logging
logger = logging.getLogger(__name__)

class SignatureTrainer:
    """
    Main trainer class for signature detection.
    
    This class handles the complete training pipeline including:
    - Data loading and validation
    - Model initialization
    - Training loop
    - Validation and metrics
    - Model saving
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the trainer.
        
        Args:
            config_path: Path to the training configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.model = None
        self.train_dataset = None
        self.val_dataset = None
        
        # Setup directories
        self.output_dir = Path(self.config.get('output_dir', 'runs/train'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(self.output_dir / 'train.log')
    
    def _load_config(self) -> Dict[str, Any]:
        """Load training configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {str(e)}")
            raise
    
    def setup_data(self) -> None:
        """Setup training and validation datasets."""
        try:
            data_config = self.config['data']
            
            # Create datasets
            self.train_dataset = SignatureDataset.create_from_config(
                config_path=data_config['config_path'],
                split='train'
            )
            
            self.val_dataset = SignatureDataset.create_from_config(
                config_path=data_config['config_path'],
                split='val'
            )
            
            logger.info(f"Training dataset: {len(self.train_dataset)} samples")
            logger.info(f"Validation dataset: {len(self.val_dataset)} samples")
            
            # Log class distribution
            train_dist = self.train_dataset.get_class_distribution()
            val_dist = self.val_dataset.get_class_distribution()
            
            logger.info(f"Training class distribution: {train_dist}")
            logger.info(f"Validation class distribution: {val_dist}")
            
        except Exception as e:
            logger.error(f"Failed to setup data: {str(e)}")
            raise
    
    def setup_model(self) -> None:
        """Setup the detection model."""
        try:
            model_config = self.config['model']
            
            self.model = SignatureDetectionModel(
                model_size=model_config.get('size', 'n'),
                num_classes=model_config.get('num_classes', 1),
                class_names=model_config.get('class_names', ['signature'])
            )
            
            logger.info("Model setup completed")
            logger.info(f"Model info: {self.model.get_model_info()}")
            
        except Exception as e:
            logger.error(f"Failed to setup model: {str(e)}")
            raise
    
    def train(self) -> Dict[str, Any]:
        """
        Execute the training process.
        
        Returns:
            Training results dictionary
        """
        try:
            logger.info("Starting training process...")
            
            # Training parameters
            train_config = self.config['train']
            data_config_path = self.config['data']['config_path']
            
            # Start training
            results = self.model.train(
                data_config=data_config_path,
                epochs=train_config.get('epochs', 100),
                batch_size=train_config.get('batch_size', 16),
                img_size=train_config.get('img_size', 640),
                learning_rate=train_config.get('learning_rate', 0.01),
                patience=train_config.get('patience', 50),
                save_dir=str(self.output_dir)
            )
            
            logger.info("Training completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate the trained model.
        
        Returns:
            Validation results dictionary
        """
        try:
            logger.info("Starting validation...")
            
            eval_config = self.config.get('eval', {})
            data_config_path = self.config['data']['config_path']
            
            results = self.model.evaluate(
                data_config=data_config_path,
                split='val',
                conf_threshold=eval_config.get('conf_thresh', 0.25),
                iou_threshold=eval_config.get('iou_thresh', 0.45)
            )
            
            logger.info("Validation completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise
    
    def save_model(self) -> None:
        """Save the trained model."""
        try:
            model_path = self.output_dir / 'best_model.pt'
            self.model.save_model(str(model_path))
            logger.info(f"Model saved to: {model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            raise
    
    def run(self) -> None:
        """Execute the complete training pipeline."""
        try:
            logger.info("=== Starting Signature Detection Training ===")
            
            # Save configuration
            save_config(self.config, self.output_dir / 'config.yaml')
            
            # Setup data and model
            self.setup_data()
            self.setup_model()
            
            # Train the model
            train_results = self.train()
            
            # Validate the model
            val_results = self.validate()
            
            # Save the model
            self.save_model()
            
            logger.info("=== Training Pipeline Completed Successfully ===")
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {str(e)}")
            raise


def main():
    """Main function to run the training script."""
    parser = argparse.ArgumentParser(description='Train YOLO11 for signature detection')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to training configuration file')
    parser.add_argument('--output-dir', type=str, default='runs/train',
                        help='Output directory for training results')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume training from checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use for training (auto, cpu, cuda)')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Update config with command line arguments
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update config with command line arguments
        config['output_dir'] = args.output_dir
        config['device'] = device
        config['workers'] = args.workers
        config['seed'] = args.seed
        
        if args.resume:
            config['resume'] = args.resume
        
        # Create trainer and run
        trainer = SignatureTrainer(args.config)
        trainer.config.update(config)
        trainer.run()
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()