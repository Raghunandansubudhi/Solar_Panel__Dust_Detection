#!/usr/bin/env python3
"""
Real-Time Solar Panel Dust Detection using Webcam
GPU-accelerated inference with live camera feed
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import time
import os
from datetime import datetime

class SolarPanelDustDetector:
    def __init__(self, model_path='models/final_solar_dust_model.h5'):
        """Initialize the dust detector with trained model"""
        
        print("🚀 Initializing Solar Panel Dust Detector...")
        
        # Check if model exists
        if not os.path.exists(model_path):
            print(f"❌ Model not found at {model_path}")
            print("Please train the model first using gpu_training.py")
            return
        
        # Load model
        try:
            self.model = load_model(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return
        
        # Setup GPU if available
        self.gpu_available = self.setup_gpu()
        
        # Class labels
        self.labels = {0: 'Clean', 1: 'Dusty'}
        self.colors = {0: (0, 255, 0), 1: (0, 0, 255)}  # Green for clean, Red for dusty
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Detection history for smoothing
        self.prediction_history = []
        self.history_size = 5
        
    def setup_gpu(self):
        """Setup GPU for inference"""
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("✅ GPU acceleration enabled for inference")
                return True
            except Exception as e:
                print(f"⚠️  GPU setup warning: {e}")
                return False
        else:
            print("⚠️  No GPU detected - using CPU for inference")
            return False
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        # Resize to model input size
        img = cv2.resize(frame, (224, 224))
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to array and normalize
        img_array = image.img_to_array(img_rgb) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict_dust_level(self, frame):
        """Predict dust level from frame"""
        # Preprocess frame
        img_array = self.preprocess_frame(frame)
        
        # Make prediction
        prediction = self.model.predict(img_array, verbose=0)
        
        # Get probability and class
        dust_probability = prediction[0][0]
        predicted_class = int(dust_probability > 0.5)
        
        return predicted_class, dust_probability
    
    def smooth_prediction(self, prediction, probability):
        """Smooth predictions using history"""
        self.prediction_history.append((prediction, probability))
        
        # Keep only recent history
        if len(self.prediction_history) > self.history_size:
            self.prediction_history.pop(0)
        
        # Calculate average probability
        avg_probability = np.mean([p[1] for p in self.prediction_history])
        
        # Use smoothed probability for final prediction
        smoothed_prediction = int(avg_probability > 0.5)
        
        return smoothed_prediction, avg_probability
    
    def update_fps(self):
        """Update FPS counter"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_info(self, frame, prediction, probability, fps):
        """Draw information on frame"""
        # Get prediction info
        label = self.labels[prediction]
        color = self.colors[prediction]
        confidence = probability * 100
        
        # Create info text
        info_text = f"Status: {label} ({confidence:.1f}%)"
        fps_text = f"FPS: {fps}"
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 100), color, 2)
        
        # Draw text
        cv2.putText(frame, info_text, (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, fps_text, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw status indicator
        status_size = 50
        cv2.circle(frame, (frame.shape[1] - 60, 40), status_size//2, color, -1)
        cv2.circle(frame, (frame.shape[1] - 60, 40), status_size//2, (255, 255, 255), 2)
        
        return frame
    
    def run_detection(self, camera_index=0, save_detections=False):
        """Run real-time dust detection"""
        
        print(f"📹 Starting camera detection (Camera {camera_index})...")
        print("Press 'q' to quit, 's' to save current frame, 'r' to reset history")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Create output directory if saving detections
        if save_detections:
            output_dir = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_dir, exist_ok=True)
            print(f"💾 Saving detections to: {output_dir}")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading from camera")
                    break
                
                # Make prediction
                prediction, probability = self.predict_dust_level(frame)
                
                # Smooth prediction
                smoothed_prediction, smoothed_probability = self.smooth_prediction(prediction, probability)
                
                # Update FPS
                self.update_fps()
                
                # Draw information on frame
                frame = self.draw_info(frame, smoothed_prediction, smoothed_probability, self.current_fps)
                
                # Display frame
                cv2.imshow('Solar Panel Dust Detection - Real-Time', frame)
                
                # Save detection if requested
                if save_detections and frame_count % 30 == 0:  # Save every 30 frames
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    label = self.labels[smoothed_prediction]
                    filename = f"{output_dir}/{timestamp}_{label}_{smoothed_probability:.2f}.jpg"
                    cv2.imwrite(filename, frame)
                
                frame_count += 1
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    label = self.labels[smoothed_prediction]
                    filename = f"screenshot_{timestamp}_{label}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"📸 Screenshot saved: {filename}")
                elif key == ord('r'):
                    # Reset prediction history
                    self.prediction_history = []
                    print("🔄 Prediction history reset")
                
        except KeyboardInterrupt:
            print("\n⚠️  Detection interrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            
            if save_detections:
                print(f"✅ Detection session completed. Images saved in: {output_dir}")
            
            print("👋 Detection session ended")

def main():
    """Main function to run real-time detection"""
    
    print("=" * 60)
    print("🌞 REAL-TIME SOLAR PANEL DUST DETECTION")
    print("=" * 60)
    
    # Check for available models
    model_paths = [
        'models/final_solar_dust_model.h5',
        'models/best_solar_dust_model.h5',
        'models/fine_tuned_solar_dust_model.h5'
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("❌ No trained model found!")
        print("Please train a model first using:")
        print("  python gpu_training.py")
        return
    
    # Initialize detector
    detector = SolarPanelDustDetector(model_path)
    
    # Get user preferences
    print(f"\n📋 Configuration:")
    print(f"   Model: {model_path}")
    print(f"   GPU acceleration: {'✅' if detector.gpu_available else '❌'}")
    
    # Ask for camera index
    try:
        camera_index = int(input("\n📹 Enter camera index (0 for default): ") or "0")
    except ValueError:
        camera_index = 0
    
    # Ask if user wants to save detections
    save_detections = input("\n💾 Save detection images? (y/N): ").lower().startswith('y')
    
    # Start detection
    detector.run_detection(camera_index=camera_index, save_detections=save_detections)

if __name__ == "__main__":
    main()
