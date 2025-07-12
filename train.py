#!/usr/bin/env python3
"""
Training script for signature detection using YOLO11.
This script uses the configuration file and model wrapper for training.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model import SignatureDetectionModel
from dataset import SignatureDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from: {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

def validate_dataset(config: dict) -> bool:
    """Validate dataset configuration and paths."""
    try:
        data_config = config['data']
        data_yaml_path = data_config['config_path']
        
        if not Path(data_yaml_path).exists():
            logger.error(f"Data config file not found: {data_yaml_path}")
            return False
        
        # Load data config
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            data_config = yaml.safe_load(f)
        
        dataset_root = Path(data_config['path'])
        
        # Check if paths exist
        train_imgs = dataset_root / "train/images"
        train_labels = dataset_root / "train/labels"
        val_imgs = dataset_root / "val/images"
        val_labels = dataset_root / "val/labels"
        
        paths_exist = all([
            train_imgs.exists(),
            train_labels.exists(),
            val_imgs.exists(),
            val_labels.exists()
        ])
        
        if not paths_exist:
            logger.error("Some dataset paths do not exist")
            return False
        
        # Count files
        train_img_count = len(list(train_imgs.glob("*.jpg")))
        val_img_count = len(list(val_imgs.glob("*.jpg")))
        
        logger.info(f"Dataset validation passed:")
        logger.info(f"  Train images: {train_img_count}")
        logger.info(f"  Val images: {val_img_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {str(e)}")
        return False

def train_model(config: dict):
    """Train the signature detection model."""
    try:
        # Get configuration
        model_config = config['model']
        training_config = config['training']
        data_config = config['data']
        
        # Initialize model
        logger.info("Initializing model...")
        model = SignatureDetectionModel(
            model_name=model_config['architecture'],
            config=config
        )
        
        # Print model info
        model_info = model.get_model_info()
        logger.info(f"Model info: {model_info}")
        
        # Train model
        logger.info("Starting training...")
        results = model.train(
            data_yaml=data_config['config_path'],
            epochs=training_config['epochs'],
            batch_size=training_config['batch_size'],
            img_size=training_config['img_size'],
            lr0=training_config['lr0'],
            optimizer=training_config['optimizer'],
            patience=training_config['patience'],
            workers=training_config['workers'],
            device=training_config['device'],
            save_period=training_config['save_period'],
            project=training_config['project'],
            name=training_config['name']
        )
        
        logger.info("Training completed successfully!")
        
        # Save model info
        save_training_summary(results, config)
        
        return results
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

def save_training_summary(results, config: dict):
    """Save training summary to file."""
    try:
        summary = {
            'model_architecture': config['model']['architecture'],
            'training_epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['lr0'],
            'optimizer': config['training']['optimizer'],
            'training_completed': True,
            'results_path': config['training']['project']
        }
        
        summary_path = Path(config['training']['project']) / "training_summary.yaml"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            yaml.dump(summary, f, default_flow_style=False)
        
        logger.info(f"Training summary saved to: {summary_path}")
        
    except Exception as e:
        logger.warning(f"Failed to save training summary: {str(e)}")

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train signature detection model')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate dataset without training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Validate dataset
    if not validate_dataset(config):
        logger.error("Dataset validation failed. Exiting.")
        sys.exit(1)
    
    if args.validate_only:
        logger.info("Dataset validation completed. Exiting.")
        sys.exit(0)
    
    # Check device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Update config with actual device
    config['training']['device'] = device
    
    # Train model
    try:
        results = train_model(config)
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
