import cv2
import torch
import time
import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import warnings

# Suppress FutureWarnings from YOLOv5 (the torch.cuda.amp.autocast warnings)
warnings.filterwarnings('ignore', category=FutureWarning)

# ---------------------------------------------------------
# 1. CAMERA INITIALIZATION (Using Index - FIXES COLOR!)
# ---------------------------------------------------------
def init_camera(camera_index=0, width=1280, height=720):
    """
    Initialize camera using index instead of device path.
    This fixes the grayscale issue on Linux!
    High resolution (1280x720) for better detection quality.
    """
    print(f"🔧 Opening camera {camera_index}...")
    
    # Open by INDEX, not device path - this is the key!
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ ERROR: Unable to open camera {camera_index}!")
        sys.exit(1)
    
    # Set resolution and FPS
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Verify settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"🎥 Camera initialized:")
    print(f"   Resolution: {actual_width}x{actual_height}")
    print(f"   FPS: {actual_fps}")
    
    # Test read to verify color
    ret, test_frame = cap.read()
    if ret and len(test_frame.shape) == 3:
        b, g, r = cv2.split(test_frame)
        if not np.array_equal(b, g):
            print("✅ COLOR MODE CONFIRMED!")
        else:
            print("⚠️ Warning: Camera appears to be in grayscale mode")
    
    return cap

# ---------------------------------------------------------
# 2. LOAD YOLO MODEL
# ---------------------------------------------------------
def load_yolo(weights_path="best.pt", conf_threshold=0.25, iou_threshold=0.45):
    """
    Load YOLOv5 model with customizable thresholds.
    """
    print(f"\n🔍 Loading YOLOv5 model from {weights_path}...")
    
    if not Path(weights_path).exists():
        print(f"❌ ERROR: Model file not found: {weights_path}")
        sys.exit(1)
    
    model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path, force_reload=False, trust_repo=True)
    model.conf = conf_threshold  # Confidence threshold
    model.iou = iou_threshold     # NMS IOU threshold
    print(f"✅ Model loaded successfully")
    print(f"   Classes: {model.names}")
    print(f"   Confidence threshold: {conf_threshold}")
    print(f"   IOU threshold: {iou_threshold}")
    return model

# ---------------------------------------------------------
# 3. DRAW ENHANCED UI OVERLAY
# ---------------------------------------------------------
def draw_ui_overlay(frame, fps, detections, total_detections, frame_count, conf_threshold):
    """
    Draw a professional UI overlay with stats and info.
    """
    h, w = frame.shape[:2]
    
    # Semi-transparent overlay background
    overlay = frame.copy()
    
    # Top bar with stats
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # FPS display (top left)
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Detection count (top center)
    det_text = f"Detections: {detections} (Total: {total_detections})"
    text_size = cv2.getTextSize(det_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, det_text, ((w - text_size[0]) // 2, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Frame count (top right)
    frame_text = f"Frame: {frame_count}"
    text_size = cv2.getTextSize(frame_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    cv2.putText(frame, frame_text, (w - text_size[0] - 10, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    
    # Confidence threshold (second line)
    conf_text = f"Confidence: {conf_threshold:.2f}"
    cv2.putText(frame, conf_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # Controls hint (bottom)
    controls_text = "Controls: [Q]uit | [S]ave Frame | [R]ecord | [+/-] Confidence"
    text_size = cv2.getTextSize(controls_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.rectangle(frame, (0, h - 25), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(frame, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, controls_text, ((w - text_size[0]) // 2, h - 8), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

# ---------------------------------------------------------
# 4. MAIN INFERENCE LOOP
# ---------------------------------------------------------
def run_inference(weights="best.pt", camera_index=0, width=1280, height=720, 
                  conf_threshold=0.25, iou_threshold=0.45, save_dir="saved_frames", 
                  record_video=False):
    """
    Enhanced real-time inference with better UI and features.
    """
    model = load_yolo(weights, conf_threshold, iou_threshold)
    cap = init_camera(camera_index, width, height)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)
    
    # Video recording setup
    video_writer = None
    recording = False
    if record_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_filename = f"weed_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = save_path / video_filename
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (actual_width, actual_height))
        recording = True
        print(f"📹 Recording video to: {video_path}")

    print("\n🚀 Starting Real-Time Inference...")
    print("=" * 60)
    print("Controls:")
    print("  [Q] - Quit")
    print("  [S] - Save current frame")
    print("  [R] - Toggle video recording")
    print("  [+] - Increase confidence threshold")
    print("  [-] - Decrease confidence threshold")
    print("=" * 60)
    
    time.sleep(0.5)
    
    frame_count = 0
    fps_start_time = time.time()
    fps = 0.0
    total_detections = 0
    current_conf = conf_threshold
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("⚠️ Frame read failed.")
                time.sleep(0.1)
                continue

            frame_count += 1
            
            # Calculate FPS (smooth average)
            if frame_count % 10 == 0:
                elapsed = time.time() - fps_start_time
                fps = 10 / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
            
            # Update model confidence if changed
            if current_conf != model.conf:
                model.conf = current_conf
            
            # YOLO inference
            # CRITICAL: YOLOv5 model can handle BGR directly, but render() returns RGB
            # So we need to convert the rendered result back to BGR for OpenCV
            inference_start = time.time()
            results = model(frame)  # Pass BGR frame directly (YOLOv5 handles it)
            inference_time = (time.time() - inference_start) * 1000  # ms
            
            # Get detection info
            detections = len(results.xyxy[0])
            total_detections += detections
            
            # Render detections on frame
            # Use original frame and draw manually to preserve correct colors
            annotated = frame.copy()  # Start with original BGR frame (correct colors)
            
            # Draw bounding boxes manually on the original frame
            detection_coords = []  # Store coordinates for logging
            for detection in results.xyxy[0]:
                x1, y1, x2, y2, conf, cls = detection
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_name = results.names[int(cls)]
                
                # Calculate more accurate center coordinates
                # Method: Use edge detection + center of mass to find actual weed center
                # This is more accurate than simple bbox center
                
                # Extract the region within bbox
                bbox_region = frame[max(0, y1):min(frame.shape[0], y2), 
                                   max(0, x1):min(frame.shape[1], x2)]
                
                # Default to bbox center (fallback)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                if bbox_region.size > 0 and bbox_region.shape[0] > 15 and bbox_region.shape[1] > 15:
                    try:
                        # Convert to grayscale
                        gray_region = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2GRAY)
                        
                        # Method 1: Use Canny edge detection to find weed edges
                        # Then find center of mass of edges (more accurate than threshold)
                        edges = cv2.Canny(gray_region, 50, 150)
                        
                        # Calculate center of mass of edges
                        moments = cv2.moments(edges)
                        if moments["m00"] > 50:  # If we have enough edge pixels
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
                                # Weeds are usually detected from top, focus on upper 30%
                                bbox_height = y2 - y1
                                center_y = int(y1 + bbox_height * 0.15)  # 15% down from top
                    except:
                        # If anything fails, use simple bbox center
                        pass
                
                # Store coordinates
                detection_coords.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'center': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2)
                })
                
                # Draw bounding box in green (BGR format)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw center point as a circle
                cv2.circle(annotated, (center_x, center_y), 5, (0, 0, 255), -1)  # Red center dot
                cv2.circle(annotated, (center_x, center_y), 8, (255, 255, 255), 2)  # White outline
                
                # Draw label with background (includes center coordinates)
                label = f'{class_name} {conf:.2f}'
                coord_text = f'Center: ({center_x}, {center_y})'
                
                # Main label
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                label_y = max(y1 - 30, label_size[1] + 10)
                cv2.rectangle(annotated, (x1, label_y - label_size[1] - 5), 
                            (x1 + label_size[0], label_y + 5), (0, 255, 0), -1)
                cv2.putText(annotated, label, (x1, label_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Center coordinates text (below main label)
                coord_size, _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                coord_y = label_y + 20
                cv2.rectangle(annotated, (x1, coord_y - coord_size[1] - 3), 
                            (x1 + coord_size[0], coord_y + 3), (0, 0, 0), -1)
                cv2.putText(annotated, coord_text, (x1, coord_y), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw enhanced UI overlay
            annotated = draw_ui_overlay(annotated, fps, detections, total_detections, 
                                       frame_count, current_conf)
            
            # Add inference time info
            inf_time_text = f"Inference: {inference_time:.1f}ms"
            cv2.putText(annotated, inf_time_text, (10, 75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
            
            # Recording indicator
            if recording and video_writer:
                cv2.circle(annotated, (annotated.shape[1] - 20, 20), 8, (0, 0, 255), -1)
                cv2.putText(annotated, "REC", (annotated.shape[1] - 50, 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Display
            cv2.imshow("Weed Detection - YOLOv5 [Enhanced]", annotated)
            
            # Write to video if recording
            if recording and video_writer:
                video_writer.write(annotated)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("s"):
                # Save current frame and coordinates
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = save_path / f"detection_{timestamp}_frame{frame_count}.jpg"
                cv2.imwrite(str(filename), annotated)
                print(f"💾 Frame saved: {filename}")
                
                # Also save coordinates to a text file
                if detection_coords:
                    coord_filename = save_path / f"coordinates_{timestamp}_frame{frame_count}.txt"
                    with open(coord_filename, 'w') as f:
                        f.write(f"Frame {frame_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.write("=" * 50 + "\n")
                        for i, coord_info in enumerate(detection_coords):
                            center = coord_info['center']
                            bbox = coord_info['bbox']
                            f.write(f"\nDetection {i+1}:\n")
                            f.write(f"  Class: {coord_info['class']}\n")
                            f.write(f"  Confidence: {coord_info['confidence']:.4f}\n")
                            f.write(f"  Center Coordinates: ({center[0]}, {center[1]})\n")
                            f.write(f"  Bounding Box: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})\n")
                            f.write(f"  Width: {bbox[2] - bbox[0]}px, Height: {bbox[3] - bbox[1]}px\n")
                    print(f"📝 Coordinates saved: {coord_filename}")
            elif key == ord("r"):
                # Toggle recording
                if recording:
                    if video_writer:
                        video_writer.release()
                    recording = False
                    print("⏹️  Recording stopped")
                else:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_filename = f"weed_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                    video_path = save_path / video_filename
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    video_writer = cv2.VideoWriter(str(video_path), fourcc, 30.0, (actual_width, actual_height))
                    recording = True
                    print(f"🔴 Recording started: {video_path}")
            elif key == ord("+") or key == ord("="):
                # Increase confidence
                current_conf = min(0.95, current_conf + 0.05)
                print(f"📈 Confidence threshold: {current_conf:.2f}")
            elif key == ord("-") or key == ord("_"):
                # Decrease confidence
                current_conf = max(0.05, current_conf - 0.05)
                print(f"📉 Confidence threshold: {current_conf:.2f}")
            
            # Print detection info with center coordinates (clean, no fluff)
            if detections > 0 and frame_count % 30 == 0:
                print(f"\n📊 Frame {frame_count} - {detections} weed(s) detected:")
                for i, coord_info in enumerate(detection_coords):
                    center = coord_info['center']
                    print(f"   Weed {i+1}: Center=({center[0]}, {center[1]}) | Confidence={coord_info['confidence']:.2f}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        if video_writer:
            video_writer.release()
        cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "=" * 60)
        print("🟢 Inference Session Summary:")
        print(f"   Total frames processed: {frame_count}")
        print(f"   Total detections: {total_detections}")
        print(f"   Average FPS: {fps:.1f}")
        print(f"   Frames saved in: {save_path}")
        print("=" * 60)

# ---------------------------------------------------------
# 5. ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="YOLOv5 Real-Time Weed Detection - Enhanced Version",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python webcam_inference.py
  
  # Custom camera and resolution
  python webcam_inference.py --camera 1 --width 1920 --height 1080
  
  # Adjust confidence threshold
  python webcam_inference.py --conf 0.5
  
  # Start with video recording enabled
  python webcam_inference.py --record
        """
    )
    
    parser.add_argument("--weights", default="best.pt", 
                       help="Path to YOLOv5 model weights (default: best.pt)")
    parser.add_argument("--camera", type=int, default=0, 
                       help="Camera index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, 
                       help="Frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, 
                       help="Frame height (default: 720)")
    parser.add_argument("--conf", type=float, default=0.25, 
                       help="Confidence threshold 0.0-1.0 (default: 0.25)")
    parser.add_argument("--iou", type=float, default=0.45, 
                       help="IOU threshold for NMS 0.0-1.0 (default: 0.45)")
    parser.add_argument("--save-dir", default="saved_frames", 
                       help="Directory to save frames and videos (default: saved_frames)")
    parser.add_argument("--record", action="store_true", 
                       help="Start video recording immediately")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 <= args.conf <= 1.0:
        print("❌ ERROR: Confidence threshold must be between 0.0 and 1.0")
        sys.exit(1)
    if not 0.0 <= args.iou <= 1.0:
        print("❌ ERROR: IOU threshold must be between 0.0 and 1.0")
        sys.exit(1)
    
    run_inference(
        weights=args.weights,
        camera_index=args.camera,
        width=args.width,
        height=args.height,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        save_dir=args.save_dir,
        record_video=args.record
    )