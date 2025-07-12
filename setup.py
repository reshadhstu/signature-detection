#!/usr/bin/env python3
"""
Setup script for signature detection project.
This script helps users set up the environment and verify everything works.
"""

import os
import sys
import subprocess
from pathlib import Path
import importlib.util

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def install_requirements():
    """Install required packages."""
    print("📦 Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                      check=True, capture_output=True, text=True)
        print("✅ Packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install packages: {e}")
        return False

def check_package(package_name):
    """Check if a package is installed."""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def verify_installation():
    """Verify that all required packages are installed."""
    print("🔍 Verifying installation...")
    
    required_packages = [
        "torch",
        "torchvision", 
        "ultralytics",
        "opencv-python",
        "numpy",
        "matplotlib",
        "seaborn",
        "yaml",
        "pandas",
        "tqdm",
        "scikit-learn"
    ]
    
    all_installed = True
    for package in required_packages:
        # Handle package name differences
        check_name = package
        if package == "opencv-python":
            check_name = "cv2"
        elif package == "yaml":
            check_name = "yaml"
        
        if check_package(check_name):
            print(f"✅ {package}")
        else:
            print(f"❌ {package}")
            all_installed = False
    
    return all_installed

def check_dataset():
    """Check if the dataset is properly set up."""
    print("📊 Checking dataset...")
    
    project_root = Path(__file__).parent
    data_config = project_root / "data" / "signature" / "data.yaml"
    train_images = project_root / "data" / "signature" / "train" / "images"
    train_labels = project_root / "data" / "signature" / "train" / "labels"
    val_images = project_root / "data" / "signature" / "val" / "images"
    val_labels = project_root / "data" / "signature" / "val" / "labels"
    
    checks = [
        (data_config, "Data configuration file"),
        (train_images, "Training images directory"),
        (train_labels, "Training labels directory"),
        (val_images, "Validation images directory"),
        (val_labels, "Validation labels directory")
    ]
    
    all_present = True
    for path, description in checks:
        if path.exists():
            if path.is_dir():
                count = len(list(path.glob("*")))
                print(f"✅ {description} ({count} files)")
            else:
                print(f"✅ {description}")
        else:
            print(f"❌ {description} - not found")
            all_present = False
    
    return all_present

def test_simple_import():
    """Test importing key modules."""
    print("🧪 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        import torchvision
        print(f"✅ Torchvision {torchvision.__version__}")
        
        from ultralytics import YOLO
        print("✅ Ultralytics YOLO")
        
        import cv2
        print(f"✅ OpenCV {cv2.__version__}")
        
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import matplotlib
        print(f"✅ Matplotlib {matplotlib.__version__}")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    project_root = Path(__file__).parent
    dirs_to_create = [
        "runs",
        "runs/train",
        "runs/val", 
        "runs/predict"
    ]
    
    for dir_name in dirs_to_create:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ {dir_name}/")

def main():
    """Main setup function."""
    print("🚀 Setting up Signature Detection Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Verify installation
    if not verify_installation():
        print("❌ Some packages are missing. Please install them manually.")
        return False
    
    # Test imports
    if not test_simple_import():
        return False
    
    # Create directories
    create_directories()
    
    # Check dataset
    dataset_ok = check_dataset()
    
    print("\n" + "=" * 50)
    if dataset_ok:
        print("🎉 Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Train the model: python train_simple.py")
        print("2. Evaluate the model: python evaluate_simple.py")
        print("3. Make predictions: python predict_simple.py --source path/to/image.jpg --save")
    else:
        print("⚠️  Setup completed with warnings!")
        print("📋 Dataset not found. Please:")
        print("1. Download the signature dataset")
        print("2. Place images in data/signature/train/images/ and data/signature/val/images/")
        print("3. Place labels in data/signature/train/labels/ and data/signature/val/labels/")
        print("4. Ensure data/signature/data.yaml is properly configured")
        print("\nThen you can:")
        print("1. Train the model: python train_simple.py")
        print("2. Evaluate the model: python evaluate_simple.py")
        print("3. Make predictions: python predict_simple.py --source path/to/image.jpg --save")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
