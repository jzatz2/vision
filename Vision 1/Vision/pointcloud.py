import time
import numpy as np
import pyrealsense2 as rs
from PyQt5.QtCore import QThread, pyqtSignal, QObject

class RealSenseThread(QThread):
    # Define signals for communication
    error_occurred = pyqtSignal(str)
    pipeline_stopped = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.stop_flag = False

        # Configure pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.profile = self.pipeline.start(self.config)

        # Depth sensor and depth scale
        device = self.profile.get_device()
        self.depth_sensor = device.first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()

        # Align depth to color
        self.align = rs.align(rs.stream.color)

        # Optional filters
        self.decimation_filter = rs.decimation_filter()
        self.temporal_filter = rs.temporal_filter()
        self.decimation_magnitude = 2
        self.use_decimation = True
        self.use_temporal = True

        # Auto-exposure
        self.auto_exposure_enabled = True

        # Storage for point cloud and metrics
        self.pc = rs.pointcloud()
        self.points_np = None
        self.colors_np = None
        self.roi_mask = None

        self.fill_rate = 0.0
        self.center_distance = 0.0
        self.mean_error = 0.0
        self.std_error = 0.0

        # ROI bounds (in meters)
        self.x_min, self.x_max = -0.5, 0.5
        self.y_min, self.y_max = -0.5, 0.5
        self.z_min, self.z_max = 0.35, 0.45

    def run(self):
        try:
            while not self.stop_flag:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                depth_frame = aligned.get_depth_frame()
                color_frame = aligned.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                # Apply filters if enabled
                if self.use_decimation:
                    self.decimation_filter.set_option(rs.option.filter_magnitude, self.decimation_magnitude)
                    depth_frame = self.decimation_filter.process(depth_frame)
                if self.use_temporal:
                    depth_frame = self.temporal_filter.process(depth_frame)

                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Auto "ground truth" from the center pixel
                h, w = depth_image.shape
                center_z = depth_image[h // 2, w // 2]
                self.center_distance = center_z * self.depth_scale

                # Generate point cloud
                self.pc.map_to(color_frame)
                points = self.pc.calculate(depth_frame)

                v = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
                t = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)

                c = []
                valid_pts = 0
                Hc, Wc, _ = color_image.shape
                for (u, v2), (x3, y3, z3) in zip(t, v):
                    if z3 > 0:
                        valid_pts += 1
                    x_img = min(int(u * Wc), Wc - 1)
                    y_img = min(int(v2 * Hc), Hc - 1)
                    b, g, r = color_image[y_img, x_img] / 255.0
                    c.append([r, g, b])

                total_pts = len(v)
                self.fill_rate = valid_pts / total_pts if total_pts > 0 else 0

                # Apply ROI filter
                if total_pts > 0:
                    v = self.apply_3d_roi(v)
                    c = np.asarray(c)[self.roi_mask]
                total_pts_roi = len(v)

                # Compute error vs. center distance (example)
                if total_pts_roi > 0 and self.center_distance > 0:
                    distances = np.linalg.norm(v, axis=1)
                    err = np.abs(distances - self.center_distance)
                    self.mean_error = np.mean(err)
                    self.std_error = np.std(err)
                else:
                    self.mean_error = 0.0
                    self.std_error = 0.0

                self.points_np = v
                self.colors_np = c

                time.sleep(0.01)
        except Exception as e:
            error_msg = f"RealSense thread error: {e}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(error_msg)
        finally:
            if self.pipeline:
                self.pipeline.stop()
                print("[DEBUG] RealSense pipeline stopped")
                self.pipeline_stopped.emit()

    def stop(self):
        """Signal the thread to stop"""
        self.stop_flag = True
        self.wait()  # Wait for the thread to finish

    def apply_3d_roi(self, v):
        """Apply 3D region of interest filtering to point cloud vertices"""
        mask = (
            (v[:, 0] >= self.x_min) & (v[:, 0] <= self.x_max) &
            (v[:, 1] >= self.y_min) & (v[:, 1] <= self.y_max) &
            (v[:, 2] >= self.z_min) & (v[:, 2] <= self.z_max)
        )
        self.roi_mask = mask
        return v[mask]

    def set_xmin(self, val): self.x_min = val
    def set_xmax(self, val): self.x_max = val
    def set_ymin(self, val): self.y_min = val
    def set_ymax(self, val): self.y_max = val
    def set_zmin(self, val): self.z_min = val
    def set_zmax(self, val): self.z_max = val
    def set_decimation_magnitude(self, val): self.decimation_magnitude = int(val)
    def enable_decimation(self, enable): self.use_decimation = enable
    def enable_temporal(self, enable): self.use_temporal = enable
    def enable_auto_exposure(self, enable): self.auto_exposure_enabled = enable