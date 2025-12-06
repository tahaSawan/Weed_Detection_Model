"""
Batch Inference Script for Weed Detection
Processes all images in a directory and saves results with coordinates.
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import argparse
import json
import csv
from datetime import datetime
from tqdm import tqdm
import warnings

# Suppress FutureWarnings
warnings.filterwarnings('ignore', category=FutureWarning)

def load_model(weights_path='best.pt', conf_threshold=0.25, iou_threshold=0.45):
    """Load YOLOv5 model."""
    print(f"\n🔍 Loading YOLOv5 model from {weights_path}...")
    
    if not Path(weights_path).exists():
        print(f"❌ ERROR: Model file not found: {weights_path}")
        return None
    
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, 
                          force_reload=False, trust_repo=True)
    model.conf = conf_threshold
    model.iou = iou_threshold
    print(f"✅ Model loaded successfully")
    print(f"   Classes: {model.names}")
    print(f"   Confidence threshold: {conf_threshold}")
    return model

def calculate_accurate_center(frame, x1, y1, x2, y2):
    """
    Calculate accurate center coordinates using edge detection and center of mass.
    Same method as webcam_inference.py for consistency.
    """
    # Default to bbox center
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    
    # Extract the region within bbox
    bbox_region = frame[max(0, y1):min(frame.shape[0], y2), 
                       max(0, x1):min(frame.shape[1], x2)]
    
    if bbox_region.size > 0 and bbox_region.shape[0] > 15 and bbox_region.shape[1] > 15:
        try:
            # Convert to grayscale
            gray_region = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Use Canny edge detection to find weed edges
            edges = cv2.Canny(gray_region, 50, 150)
            
            # Calculate center of mass of edges
            moments = cv2.moments(edges)
            if moments["m00"] > 50:
                cm_x = int(moments["m10"] / moments["m00"])
                cm_y = int(moments["m01"] / moments["m00"])
                center_x = x1 + cm_x
                center_y = y1 + cm_y
            else:
                # Method 2: Fallback - use Otsu threshold + center of mass
                _, thresh = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                moments = cv2.moments(thresh)
                if moments["m00"] > 100:
                    cm_x = int(moments["m10"] / moments["m00"])
                    cm_y = int(moments["m01"] / moments["m00"])
                    center_x = x1 + cm_x
                    center_y = y1 + cm_y
                else:
                    # Method 3: Final fallback - weighted center (upper portion)
                    bbox_height = y2 - y1
                    center_y = int(y1 + bbox_height * 0.15)
        except:
            pass
    
    return center_x, center_y

def process_image(model, image_path, output_dir, save_annotated=True):
    """
    Process a single image and return detection results.
    """
    # Read image
    frame = cv2.imread(str(image_path))
    if frame is None:
        return None
    
    # Run inference
    results = model(frame)
    
    # Get detections
    detections = results.xyxy[0]
    
    # Process detections
    detection_data = []
    annotated_frame = frame.copy()
    
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_name = results.names[int(cls)]
        
        # Calculate accurate center
        center_x, center_y = calculate_accurate_center(frame, x1, y1, x2, y2)
        
        # Store detection data
        detection_data.append({
            'class': class_name,
            'confidence': float(conf),
            'center_x': center_x,
            'center_y': center_y,
            'bbox_x1': x1,
            'bbox_y1': y1,
            'bbox_x2': x2,
            'bbox_y2': y2,
            'width': x2 - x1,
            'height': y2 - y1
        })
        
        # Draw on annotated frame
        if save_annotated:
            # Bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Center point
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.circle(annotated_frame, (center_x, center_y), 8, (255, 255, 255), 2)
            
            # Label
            label = f'{class_name} {conf:.2f}'
            coord_text = f'Center: ({center_x}, {center_y})'
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            label_y = max(y1 - 30, label_size[1] + 10)
            cv2.rectangle(annotated_frame, (x1, label_y - label_size[1] - 5), 
                        (x1 + label_size[0], label_y + 5), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, label_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            # Center coordinates
            coord_size, _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            coord_y = label_y + 20
            cv2.rectangle(annotated_frame, (x1, coord_y - coord_size[1] - 3), 
                        (x1 + coord_size[0], coord_y + 3), (0, 0, 0), -1)
            cv2.putText(annotated_frame, coord_text, (x1, coord_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save annotated image if requested
    if save_annotated and detection_data:
        output_path = output_dir / f"annotated_{image_path.name}"
        cv2.imwrite(str(output_path), annotated_frame)
    
    return {
        'image': image_path.name,
        'image_path': str(image_path),
        'detections': detection_data,
        'detection_count': len(detection_data)
    }

def main():
    parser = argparse.ArgumentParser(
        description="Batch Inference for Weed Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all images in Sample-Dataset folder
  python batch_inference.py --source Sample-Dataset
  
  # Custom confidence threshold
  python batch_inference.py --source Sample-Dataset --conf 0.5
  
  # Save results to custom directory
  python batch_inference.py --source Sample-Dataset --output results/
        """
    )
    
    parser.add_argument('--source', type=str, default='Sample-Dataset',
                       help='Path to directory containing images (default: Sample-Dataset)')
    parser.add_argument('--weights', type=str, default='best.pt',
                       help='Path to model weights (default: best.pt)')
    parser.add_argument('--output', type=str, default='batch_results',
                       help='Output directory for results (default: batch_results)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IOU threshold (default: 0.45)')
    parser.add_argument('--save-images', action='store_true',
                       help='Save annotated images (default: False)')
    parser.add_argument('--format', choices=['json', 'csv', 'both'], default='both',
                       help='Output format for coordinates (default: both)')
    
    args = parser.parse_args()
    
    # Validate inputs
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"❌ ERROR: Source directory not found: {args.source}")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in source_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"❌ ERROR: No images found in {args.source}")
        return
    
    print(f"\n📁 Found {len(image_files)} images in {args.source}")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    print(f"📂 Output directory: {output_dir}")
    
    # Load model
    model = load_model(args.weights, args.conf, args.iou)
    if model is None:
        return
    
    # Process all images
    print(f"\n🚀 Starting batch inference...")
    print("=" * 60)
    
    all_results = []
    total_detections = 0
    
    # Process with progress bar
    for image_path in tqdm(image_files, desc="Processing images"):
        result = process_image(model, image_path, output_dir, args.save_images)
        if result:
            all_results.append(result)
            total_detections += result['detection_count']
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.format in ['json', 'both']:
        json_path = output_dir / f"detections_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n💾 JSON results saved: {json_path}")
    
    if args.format in ['csv', 'both']:
        csv_path = output_dir / f"detections_{timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['Image', 'Detection', 'Class', 'Confidence', 
                           'Center_X', 'Center_Y', 'BBox_X1', 'BBox_Y1', 
                           'BBox_X2', 'BBox_Y2', 'Width', 'Height'])
            
            # Data
            for result in all_results:
                for i, det in enumerate(result['detections']):
                    writer.writerow([
                        result['image'],
                        i + 1,
                        det['class'],
                        f"{det['confidence']:.4f}",
                        det['center_x'],
                        det['center_y'],
                        det['bbox_x1'],
                        det['bbox_y1'],
                        det['bbox_x2'],
                        det['bbox_y2'],
                        det['width'],
                        det['height']
                    ])
        print(f"💾 CSV results saved: {csv_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("🟢 Batch Inference Summary:")
    print(f"   Total images processed: {len(all_results)}")
    print(f"   Total detections: {total_detections}")
    print(f"   Images with detections: {sum(1 for r in all_results if r['detection_count'] > 0)}")
    print(f"   Average detections per image: {total_detections/len(all_results):.2f}")
    print(f"   Results saved in: {output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()

