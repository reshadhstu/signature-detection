#!/usr/bin/env python3
"""
Prediction/Inference script for signature detection.

This script provides inference capabilities for signature detection using
a trained YOLO model. It can process single images, directories of images,
or video files.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from model import SignatureDetectionModel
from utils import setup_logging, save_results, visualize_predictions

# Configure logging
logger = logging.getLogger(__name__)

class SignaturePredictor:
    """
    Predictor class for signature detection inference.
    
    This class provides inference capabilities for signature detection
    on various input types (images, directories, videos).
    """
    
    def __init__(self, model_path: str, output_dir: str = "runs/predict"):
        """
        Initialize the predictor.
        
        Args:
            model_path: Path to the trained model
            output_dir: Output directory for prediction results
        """
        self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        setup_logging(self.output_dir / 'prediction.log')
        
        # Initialize model
        self.model = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the trained model."""
        try:
            self.model = SignatureDetectionModel()
            self.model.load_model(self.model_path)
            logger.info(f"Model loaded from: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def predict_single_image(self, 
                           image_path: str,
                           conf_threshold: float = 0.25,
                           iou_threshold: float = 0.45,
                           save_results: bool = True,
                           show_results: bool = False) -> Dict[str, Any]:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to the image file
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            save_results: Whether to save results
            show_results: Whether to display results
            
        Returns:
            Prediction results dictionary
        """
        try:
            logger.info(f"Predicting on image: {image_path}")
            
            # Make prediction
            results = self.model.predict(
                source=image_path,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
                save_results=False
            )
            
            if results:
                result = results[0]
                
                # Log results
                num_detections = len(result['boxes'])
                logger.info(f"Found {num_detections} signature(s)")
                
                if num_detections > 0:
                    for i, (box, conf, class_name) in enumerate(zip(
                        result['boxes'], result['confidences'], result['class_names'])):
                        logger.info(f"  Detection {i+1}: {class_name} (confidence: {conf:.3f})")
                
                # Save/show results
                if save_results or show_results:
                    img_name = Path(image_path).stem
                    save_path = self.output_dir / f"{img_name}_prediction.png" if save_results else None
                    
                    visualize_predictions(
                        image_path=image_path,
                        predictions=result,
                        save_path=str(save_path) if save_path else None,
                        show_labels=True,
                        show_conf=True
                    )
                    
                    if not show_results:
                        plt.close()
                
                return result
            else:
                logger.info("No detections found")
                return {
                    'image_path': image_path,
                    'boxes': [],
                    'confidences': [],
                    'classes': [],
                    'class_names': []
                }
                
        except Exception as e:
            logger.error(f"Prediction failed for {image_path}: {str(e)}")
            raise
    
    def predict_directory(self, 
                         directory_path: str,
                         conf_threshold: float = 0.25,
                         iou_threshold: float = 0.45,
                         save_results: bool = True,
                         max_images: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Make predictions on all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            save_results: Whether to save results
            max_images: Maximum number of images to process
            
        Returns:
            List of prediction results
        """
        try:
            directory_path = Path(directory_path)
            logger.info(f"Predicting on directory: {directory_path}")
            
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(directory_path.glob(f'*{ext}'))
                image_files.extend(directory_path.glob(f'*{ext.upper()}'))
            
            image_files = sorted(image_files)
            
            if max_images:
                image_files = image_files[:max_images]
            
            logger.info(f"Found {len(image_files)} images to process")
            
            # Process each image
            all_results = []
            for i, img_path in enumerate(image_files):
                logger.info(f"Processing image {i+1}/{len(image_files)}: {img_path.name}")
                
                result = self.predict_single_image(
                    image_path=str(img_path),
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold,
                    save_results=save_results,
                    show_results=False
                )
                
                all_results.append(result)
            
            # Save summary
            if save_results:
                summary = {
                    'total_images': len(image_files),
                    'total_detections': sum(len(r['boxes']) for r in all_results),
                    'images_with_detections': sum(1 for r in all_results if len(r['boxes']) > 0),
                    'conf_threshold': conf_threshold,
                    'iou_threshold': iou_threshold,
                    'results': all_results
                }
                
                save_results(summary, self.output_dir / 'directory_predictions.json', 'json')
                logger.info(f"Directory prediction summary saved")
            
            return all_results
            
        except Exception as e:
            logger.error(f"Directory prediction failed: {str(e)}")
            raise
    
    def predict_video(self, 
                     video_path: str,
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45,
                     save_video: bool = True,
                     frame_skip: int = 1) -> List[Dict[str, Any]]:
        """
        Make predictions on video frames.
        
        Args:
            video_path: Path to video file
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            save_video: Whether to save annotated video
            frame_skip: Process every nth frame
            
        Returns:
            List of prediction results for each frame
        """
        try:
            logger.info(f"Predicting on video: {video_path}")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            logger.info(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Setup video writer if saving
            if save_video:
                output_path = self.output_dir / f"{Path(video_path).stem}_predictions.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            
            # Process frames
            frame_results = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_skip == 0:
                    # Save frame temporarily
                    temp_frame_path = self.output_dir / f"temp_frame_{frame_count}.jpg"
                    cv2.imwrite(str(temp_frame_path), frame)
                    
                    # Make prediction
                    result = self.predict_single_image(
                        image_path=str(temp_frame_path),
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        save_results=False,
                        show_results=False
                    )
                    
                    result['frame_number'] = frame_count
                    frame_results.append(result)
                    
                    # Draw predictions on frame
                    if save_video:
                        annotated_frame = self._annotate_frame(frame, result)
                        out.write(annotated_frame)
                    
                    # Clean up temp file
                    temp_frame_path.unlink()
                    
                    if frame_count % (frame_skip * 30) == 0:
                        logger.info(f"Processed frame {frame_count}/{total_frames}")
                
                frame_count += 1
            
            # Clean up
            cap.release()
            if save_video:
                out.release()
                logger.info(f"Annotated video saved to: {output_path}")
            
            # Save results
            video_summary = {
                'video_path': str(video_path),
                'total_frames': total_frames,
                'processed_frames': len(frame_results),
                'total_detections': sum(len(r['boxes']) for r in frame_results),
                'frames_with_detections': sum(1 for r in frame_results if len(r['boxes']) > 0),
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'frame_results': frame_results
            }
            
            save_results(video_summary, self.output_dir / 'video_predictions.json', 'json')
            logger.info(f"Video prediction results saved")
            
            return frame_results
            
        except Exception as e:
            logger.error(f"Video prediction failed: {str(e)}")
            raise
    
    def _annotate_frame(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """
        Annotate frame with prediction results.
        
        Args:
            frame: Input frame
            result: Prediction results
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        boxes = result.get('boxes', [])
        confidences = result.get('confidences', [])
        class_names = result.get('class_names', [])
        
        for box, conf, class_name in zip(boxes, confidences, class_names):
            x1, y1, x2, y2 = map(int, box)
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return annotated
    
    def batch_predict(self, 
                     sources: List[str],
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45,
                     save_results: bool = True) -> List[Dict[str, Any]]:
        """
        Make predictions on multiple sources.
        
        Args:
            sources: List of source paths (images, directories, videos)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            save_results: Whether to save results
            
        Returns:
            List of all prediction results
        """
        try:
            logger.info(f"Batch prediction on {len(sources)} sources")
            
            all_results = []
            
            for i, source in enumerate(sources):
                logger.info(f"Processing source {i+1}/{len(sources)}: {source}")
                
                source_path = Path(source)
                
                if source_path.is_file():
                    # Check if it's a video or image
                    if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                        results = self.predict_video(
                            video_path=source,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold,
                            save_video=save_results
                        )
                    else:
                        results = [self.predict_single_image(
                            image_path=source,
                            conf_threshold=conf_threshold,
                            iou_threshold=iou_threshold,
                            save_results=save_results
                        )]
                elif source_path.is_dir():
                    results = self.predict_directory(
                        directory_path=source,
                        conf_threshold=conf_threshold,
                        iou_threshold=iou_threshold,
                        save_results=save_results
                    )
                else:
                    logger.warning(f"Source not found: {source}")
                    continue
                
                all_results.extend(results)
            
            logger.info(f"Batch prediction completed. Total results: {len(all_results)}")
            return all_results
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise


def main():
    """Main function to run the prediction script."""
    parser = argparse.ArgumentParser(description='Signature detection inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--source', type=str, required=True,
                        help='Source path (image, directory, or video)')
    parser.add_argument('--output-dir', type=str, default='runs/predict',
                        help='Output directory for results')
    parser.add_argument('--conf-threshold', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.45,
                        help='IoU threshold for NMS')
    parser.add_argument('--save-results', action='store_true',
                        help='Save prediction results')
    parser.add_argument('--show-results', action='store_true',
                        help='Display prediction results')
    parser.add_argument('--max-images', type=int, default=None,
                        help='Maximum number of images to process (for directories)')
    parser.add_argument('--frame-skip', type=int, default=1,
                        help='Process every nth frame (for videos)')
    
    args = parser.parse_args()
    
    try:
        # Create predictor
        predictor = SignaturePredictor(
            model_path=args.model,
            output_dir=args.output_dir
        )
        
        # Check source type and predict
        source_path = Path(args.source)
        
        if source_path.is_file():
            if source_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                # Video prediction
                results = predictor.predict_video(
                    video_path=args.source,
                    conf_threshold=args.conf_threshold,
                    iou_threshold=args.iou_threshold,
                    save_video=args.save_results,
                    frame_skip=args.frame_skip
                )
            else:
                # Single image prediction
                results = predictor.predict_single_image(
                    image_path=args.source,
                    conf_threshold=args.conf_threshold,
                    iou_threshold=args.iou_threshold,
                    save_results=args.save_results,
                    show_results=args.show_results
                )
        elif source_path.is_dir():
            # Directory prediction
            results = predictor.predict_directory(
                directory_path=args.source,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
                save_results=args.save_results,
                max_images=args.max_images
            )
        else:
            logger.error(f"Source not found: {args.source}")
            sys.exit(1)
        
        logger.info("Prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)


if __name__ == '__main__':
    main()
