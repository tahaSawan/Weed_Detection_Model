"""
FIXED GUI - Works EXACTLY like batch_inference.py
No filtering, same model settings, same processing
"""

import cv2
import torch
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore', category=FutureWarning)

def calculate_accurate_center(frame, x1, y1, x2, y2):
    """
    Calculate accurate center - EXACTLY like batch_inference.py
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

class WeedDetectionGUI:
    def __init__(self, root, model_path='best.pt', conf_threshold=0.25):
        self.root = root
        self.root.title("🌿 Weed Detection - Fixed (Same as Batch Processing)")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.model = None
        self.current_image = None
        self.current_results = None
        
        self.load_model()
        self.create_widgets()
    
    def load_model(self):
        """Load model - EXACTLY like batch_inference.py"""
        try:
            print("Loading model...")
            self.model = torch.hub.load("ultralytics/yolov5", "custom", 
                                      path=self.model_path, force_reload=False, trust_repo=True)
            self.model.conf = self.conf_threshold
            self.model.iou = 0.45  # Same as batch_inference.py
            print("✅ Model loaded!")
            print(f"   Confidence: {self.conf_threshold} (same as batch)")
            print(f"   IOU: 0.45 (same as batch)")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {e}")
            self.root.destroy()
    
    def create_widgets(self):
        """Create GUI widgets."""
        # Title
        title_frame = tk.Frame(self.root, bg='#1e1e1e')
        title_frame.pack(pady=15)
        
        title_label = tk.Label(title_frame, text="🌿 Weed Detection - Fixed Version", 
                              font=('Arial', 24, 'bold'), bg='#1e1e1e', fg='#4CAF50')
        title_label.pack()
        
        subtitle = tk.Label(title_frame, text="Works EXACTLY like batch_inference.py", 
                           font=('Arial', 12), bg='#1e1e1e', fg='#888888')
        subtitle.pack()
        
        # Control panel
        control_frame = tk.Frame(self.root, bg='#2b2b2b', relief=tk.RAISED, bd=2)
        control_frame.pack(pady=15, padx=20, fill=tk.X)
        
        # Buttons
        btn_frame = tk.Frame(control_frame, bg='#2b2b2b')
        btn_frame.pack(pady=10)
        
        select_btn = tk.Button(btn_frame, text="📁 Select Image", 
                              command=self.select_image,
                              font=('Arial', 12, 'bold'),
                              bg='#4CAF50', fg='white',
                              padx=25, pady=12,
                              cursor='hand2', relief=tk.RAISED, bd=3)
        select_btn.pack(side=tk.LEFT, padx=10)
        
        save_btn = tk.Button(btn_frame, text="💾 Save Results", 
                            command=self.save_results,
                            font=('Arial', 12, 'bold'),
                            bg='#2196F3', fg='white',
                            padx=25, pady=12,
                            cursor='hand2', relief=tk.RAISED, bd=3,
                            state=tk.DISABLED)
        save_btn.pack(side=tk.LEFT, padx=10)
        self.save_btn = save_btn
        
        # Settings
        settings_frame = tk.Frame(control_frame, bg='#2b2b2b')
        settings_frame.pack(pady=10)
        
        # Confidence threshold
        conf_frame = tk.Frame(settings_frame, bg='#2b2b2b')
        conf_frame.pack(side=tk.LEFT, padx=20)
        
        conf_label = tk.Label(conf_frame, text="Confidence:", 
                             font=('Arial', 11, 'bold'), bg='#2b2b2b', fg='white')
        conf_label.pack(side=tk.LEFT, padx=5)
        
        self.conf_var = tk.DoubleVar(value=self.conf_threshold)
        conf_scale = tk.Scale(conf_frame, from_=0.1, to=1.0, resolution=0.05,
                             orient=tk.HORIZONTAL, variable=self.conf_var,
                             bg='#2b2b2b', fg='white', 
                             length=200, command=self.update_confidence)
        conf_scale.pack(side=tk.LEFT, padx=5)
        
        self.conf_value_label = tk.Label(conf_frame, text=f"{self.conf_threshold:.2f}", 
                                        font=('Arial', 11, 'bold'), 
                                        bg='#2b2b2b', fg='#FFC107', width=5)
        self.conf_value_label.pack(side=tk.LEFT, padx=5)
        
        # Info label
        info_label = tk.Label(settings_frame, 
                             text="⚠️ No filtering applied (same as batch processing)", 
                             font=('Arial', 10), bg='#2b2b2b', fg='#FFC107')
        info_label.pack(side=tk.LEFT, padx=20)
        
        # Main content
        content_frame = tk.Frame(self.root, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Image display
        image_frame = tk.Frame(content_frame, bg='#1e1e1e', relief=tk.RAISED, bd=2)
        image_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        image_title = tk.Label(image_frame, text="Detection Results", 
                              font=('Arial', 14, 'bold'), 
                              bg='#1e1e1e', fg='white')
        image_title.pack(pady=10)
        
        self.image_label = tk.Label(image_frame, bg='#1e1e1e', 
                                    text="No image selected\n\nClick 'Select Image' to begin")
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results panel
        results_frame = tk.Frame(content_frame, bg='#1e1e1e', relief=tk.RAISED, bd=2, width=450)
        results_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        results_frame.pack_propagate(False)
        
        results_title = tk.Label(results_frame, text="Detection Statistics", 
                                font=('Arial', 14, 'bold'), 
                                bg='#1e1e1e', fg='white')
        results_title.pack(pady=10)
        
        # Scrollable text
        scroll_frame = tk.Frame(results_frame, bg='#1e1e1e')
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = tk.Scrollbar(scroll_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(scroll_frame, 
                                   font=('Courier', 10),
                                   bg='#2b2b2b', fg='#00ff00',
                                   yscrollcommand=scrollbar.set,
                                   wrap=tk.WORD)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        # Status bar
        self.status_label = tk.Label(self.root, text="Ready - Select an image to begin", 
                                     font=('Arial', 10), 
                                     bg='#1e1e1e', fg='#4CAF50',
                                     anchor=tk.W, padx=15, pady=8)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def update_confidence(self, value=None):
        """Update confidence - EXACTLY like batch_inference.py"""
        if self.model:
            self.conf_threshold = self.conf_var.get()
            self.model.conf = self.conf_threshold
            self.model.iou = 0.45  # Keep same as batch
            self.conf_value_label.config(text=f"{self.conf_threshold:.2f}")
            if self.current_image is not None:
                self.process_image(self.current_image)
    
    def select_image(self):
        """Open file dialog."""
        file_path = filedialog.askopenfilename(
            title="Select Weed Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.current_image = file_path
            self.status_label.config(text=f"Processing: {Path(file_path).name}...", fg='#FFC107')
            self.root.update()
            self.process_image(file_path)
    
    def process_image(self, image_path):
        """Process image - EXACTLY like batch_inference.py"""
        try:
            # Read image
            frame = cv2.imread(image_path)
            if frame is None:
                messagebox.showerror("Error", "Failed to load image!")
                return
            
            # Run inference - EXACTLY like batch_inference.py
            results = self.model(frame)
            self.current_results = results
            
            # Get detections - EXACTLY like batch_inference.py
            detections = results.xyxy[0]  # Tensor, same as batch_inference
            
            # Process detections - EXACTLY like batch_inference.py
            detection_data = []
            annotated_frame = frame.copy()
            
            for detection in detections:  # Iterate directly like batch_inference
                x1, y1, x2, y2, conf, cls = detection
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                class_name = results.names[int(cls)]
                
                # Calculate accurate center - EXACTLY like batch_inference
                center_x, center_y = calculate_accurate_center(frame, x1, y1, x2, y2)
                
                # Store data - EXACTLY like batch_inference
                detection_data.append({
                    'class': class_name,
                    'confidence': float(conf),
                    'center': (center_x, center_y),
                    'bbox': (x1, y1, x2, y2)
                })
                
                # Draw - EXACTLY like batch_inference.py
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
            
            self.current_annotated = annotated_frame
            self.current_detections = detection_data
            
            # Display
            self.display_image(annotated_frame)
            self.display_results(image_path, detection_data, frame.shape)
            
            self.save_btn.config(state=tk.NORMAL)
            
            self.status_label.config(text=f"✅ Processed: {len(detection_data)} detection(s) (same as batch processing)", 
                                    fg='#4CAF50')
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to process image: {e}")
            self.status_label.config(text="❌ Error processing image", fg='#f44336')
    
    def display_image(self, frame):
        """Display image in GUI."""
        h, w = frame.shape[:2]
        max_w, max_h = 900, 700
        
        if w > max_w or h > max_h:
            scale = min(max_w / w, max_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        photo = ImageTk.PhotoImage(image=pil_image)
        
        self.image_label.config(image=photo, text="")
        self.image_label.image = photo
    
    def display_results(self, image_path, detections, image_shape):
        """Display results."""
        self.results_text.delete(1.0, tk.END)
        
        result_text = "=" * 60 + "\n"
        result_text += "DETECTION RESULTS\n"
        result_text += "(Same as batch_inference.py)\n"
        result_text += "=" * 60 + "\n\n"
        
        result_text += f"Image: {Path(image_path).name}\n"
        result_text += f"Size: {image_shape[1]}×{image_shape[0]} pixels\n"
        result_text += f"Detections: {len(detections)}\n"
        result_text += f"Confidence: {self.conf_threshold:.2f}\n"
        result_text += f"IOU: 0.45 (same as batch)\n"
        result_text += "\n" + "-" * 60 + "\n\n"
        
        if detections:
            for i, det in enumerate(detections, 1):
                center = det['center']
                bbox = det['bbox']
                result_text += f"Detection #{i}:\n"
                result_text += f"  Class: {det['class']}\n"
                result_text += f"  Confidence: {det['confidence']:.4f} ({det['confidence']*100:.2f}%)\n"
                result_text += f"  Center: ({center[0]}, {center[1]})\n"
                result_text += f"  BBox: ({bbox[0]}, {bbox[1]}) to ({bbox[2]}, {bbox[3]})\n"
                result_text += f"  Size: {bbox[2] - bbox[0]}×{bbox[3] - bbox[1]} px\n"
                result_text += "\n"
        else:
            result_text += "No weeds detected.\n"
            result_text += "Try lowering confidence threshold.\n"
        
        result_text += "=" * 60 + "\n"
        
        self.results_text.insert(1.0, result_text)
    
    def save_results(self):
        """Save results."""
        if not hasattr(self, 'current_annotated'):
            messagebox.showwarning("Warning", "No image processed yet!")
            return
        
        save_dir = filedialog.askdirectory(title="Select Save Directory")
        if not save_dir:
            return
        
        save_path = Path(save_dir)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_name = Path(self.current_image).stem
        
        # Save annotated image
        annotated_path = save_path / f"annotated_{image_name}_{timestamp}.jpg"
        cv2.imwrite(str(annotated_path), self.current_annotated)
        
        # Save JSON
        json_path = save_path / f"coordinates_{image_name}_{timestamp}.json"
        output_data = {
            'image': Path(self.current_image).name,
            'timestamp': timestamp,
            'confidence_threshold': self.conf_threshold,
            'iou_threshold': 0.45,
            'note': 'Same processing as batch_inference.py',
            'detections': [
                {
                    'class': det['class'],
                    'confidence': det['confidence'],
                    'center_x': det['center'][0],
                    'center_y': det['center'][1],
                    'bbox_x1': det['bbox'][0],
                    'bbox_y1': det['bbox'][1],
                    'bbox_x2': det['bbox'][2],
                    'bbox_y2': det['bbox'][3]
                }
                for det in self.current_detections
            ]
        }
        
        with open(json_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        messagebox.showinfo("Success", 
                          f"Results saved!\n\n"
                          f"Image: {annotated_path.name}\n"
                          f"Data: {json_path.name}")

def main():
    root = tk.Tk()
    app = WeedDetectionGUI(root, model_path='best.pt', conf_threshold=0.25)
    root.mainloop()

if __name__ == '__main__':
    main()

