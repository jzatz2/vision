import cv2
import numpy as np
import torch
import pyrealsense2 as rs
import os
import time
import math
from PyQt5.QtCore import QObject, pyqtSignal

# Optional: workaround for duplicate library error.
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class BoltDetectorSignals(QObject):
    """Signal class for bolt detector to communicate with GUI"""
    model_loaded = pyqtSignal(bool, str)  # success, message
    detection_complete = pyqtSignal(np.ndarray)  # bolt centers
    error_occurred = pyqtSignal(str)  # error message

class BoltDetector:
    """
    Class to detect bolts in RealSense point cloud data using YOLOv5.
    Designed to work with an existing RealSense thread rather than creating its own.
    """
    
    def __init__(self, rs_thread, load_model_now=False, 
                model_path="best.pt"):  # Default to looking for best.pt in the current directory
        """
        Initialize the bolt detector.
        
        Args:
            rs_thread: RealSenseThread instance that provides point cloud data
            load_model_now: Whether to load the model immediately or later
            model_path: Path to the YOLOv5 model weights. Defaults to 'best.pt' in the current directory.
                        Can be a local file path or a model identifier for torch.hub.
        """
        self.rs_thread = rs_thread
        self.model_path = model_path
        self.model = None
        self.signals = BoltDetectorSignals()
        
        # Detection parameters
        self.confidence_threshold = 0.1
        self.average_depth_threshold = -1  # default until updated
        self.average_depth_offset = 0.0635  # in meters
        
        if load_model_now:
            self.load_model()
    
    def is_model_loaded(self):
        """Check if the model is loaded"""
        return self.model is not None
    
    def load_model(self):
        """Load the YOLOv5 model from a local file only. Will not download if not present."""
        start_time = time.time()
        try:
            print("[DEBUG] Loading YOLOv5 model locally...")

            # Ensure the model file exists locally; if not, do not attempt to download.
            if os.path.isfile(self.model_path):
                print(f"[DEBUG] Local model file found: {self.model_path}")
                try:
                    # Directly load the model from the local file.
                    self.model = torch.hub.load(
                        'ultralytics/yolov5', 
                        'custom', 
                        path=self.model_path, 
                        force_reload=False,
                        trust_repo=True,
                        verbose=False
                    )
                except Exception as e:
                    error_msg = f"Failed to load model from local file: {e}"
                    print(f"[ERROR] {error_msg}")
                    self.signals.model_loaded.emit(False, error_msg)
                    return False
            else:
                error_msg = f"Local model file not found: {self.model_path}"
                print(f"[ERROR] {error_msg}")
                self.signals.model_loaded.emit(False, error_msg)
                return False

            if self.model is None:
                raise ValueError("Model loading returned None")

            print('[DEBUG] CUDA available:', torch.cuda.is_available())
            using_cuda = next(self.model.parameters()).is_cuda
            print("[DEBUG] Model using CUDA (GPU):", using_cuda)

            end_time = time.time()
            load_time = end_time - start_time
            print(f"[DEBUG] Model loaded successfully in {load_time:.2f} seconds")
            self.signals.model_loaded.emit(True, f"Model loaded successfully in {load_time:.2f} seconds")
            return True

        except Exception as e:
            end_time = time.time()
            load_time = end_time - start_time
            error_msg = f"Failed to load model after {load_time:.2f} seconds: {str(e)}"
            print(f"[ERROR] {error_msg}")
            self.signals.model_loaded.emit(False, error_msg)
            return False

            
        except Exception as e:
            end_time = time.time()
            load_time = end_time - start_time
            error_msg = f"Failed to load model after {load_time:.2f} seconds: {str(e)}"
            print(f"[ERROR] {error_msg}")
            
            # Only print traceback for non-common errors
            if "No module named" not in str(e) and "not a valid" not in str(e):
                import traceback
                traceback.print_exc()
                
            self.signals.model_loaded.emit(False, error_msg)
            return False
    
    def detect_bolts(self):
        """
        Detect bolts in the current point cloud data.
        Uses the RealSense thread's data rather than creating a new pipeline.
        """
        start_time = time.time()
        
        try:
            # First make sure model is loaded
            if not self.is_model_loaded():
                print("[DEBUG] Model not loaded, loading now...")
                success = self.load_model()
                if not success:
                    self.signals.error_occurred.emit("Failed to load model automatically. Please try again.")
                    return
            
            # Wait for frames from the RealSense thread
            if self.rs_thread.points_np is None or len(self.rs_thread.points_np) == 0:
                self.signals.error_occurred.emit("No point cloud data available")
                return
            
            # Get frames directly from the thread's pipeline
            print("[DEBUG] Getting frames from RealSense...")
            frames = self.rs_thread.pipeline.wait_for_frames()
            aligned = self.rs_thread.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
            
            if not depth_frame or not color_frame:
                self.signals.error_occurred.emit("Failed to get frames from camera")
                return
            
            # Create color image for YOLOv5
            color_image = np.asanyarray(color_frame.get_data())
            
            # Run inference with YOLOv5
            inference_start = time.time()
            print("[DEBUG] Running YOLOv5 inference...")
            results = self.model(color_image)
            inference_time = time.time() - inference_start
            print(f"[DEBUG] Inference completed in {inference_time:.2f} seconds")
            
            # Get 3D coordinates for each detection
            centers_xyz = self.get_center_xyz_for_detections(results, depth_frame)
            
            # Calculate total detection time
            total_time = time.time() - start_time
            
            print(f"[DEBUG] Detected {len(centers_xyz)} bolts in {total_time:.2f} seconds")
            if len(centers_xyz) > 0:
                print(f"[DEBUG] Bolt coordinates: {centers_xyz}")
            
            # Emit signal with bolt positions
            self.signals.detection_complete.emit(centers_xyz)
            
        except Exception as e:
            error_msg = f"Error during bolt detection: {e}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            self.signals.error_occurred.emit(error_msg)
    
    def get_center_xyz_for_detections(self, results, depth_frame):
        """
        For each detected bounding box in YOLO results,
        compute the center pixel, clamp it, and deproject it to 3D using depth_intrinsics.
        Returns a NumPy array of XYZ coordinates.
        """
        centers_xyz = []
        bboxes = results.xyxy[0].cpu().numpy()
        
        # Get the correct intrinsics from the depth frame
        depth_profile = depth_frame.get_profile().as_video_stream_profile()
        depth_intrinsics = depth_profile.get_intrinsics()
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox[:4]
            confidence = bbox[4]
            
            # Skip detections below confidence threshold
            if confidence < self.confidence_threshold:
                continue
                
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2

            depth_img = np.asanyarray(depth_frame.get_data())
            height, width = depth_img.shape
            x_center_int = int(np.clip(round(x_center), 0, width - 1))
            y_center_int = int(np.clip(round(y_center), 0, height - 1))

            # Use a spatial averaging approach for more reliable depth
            kernel_size = 5
            x_min = max(0, x_center_int - kernel_size//2)
            x_max = min(width, x_center_int + kernel_size//2 + 1)
            y_min = max(0, y_center_int - kernel_size//2)
            y_max = min(height, y_center_int + kernel_size//2 + 1)
            
            depth_region = depth_img[y_min:y_max, x_min:x_max]
            valid_depths = depth_region[depth_region > 0]
            
            if len(valid_depths) > 0:
                # Use median for robustness against outliers
                depth_value = np.median(valid_depths) * depth_frame.get_units()
                center_xyz = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x_center_int, y_center_int], depth_value)
                centers_xyz.append(center_xyz)
        
        return np.array(centers_xyz)