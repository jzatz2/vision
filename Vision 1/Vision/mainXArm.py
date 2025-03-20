import sys
import numpy as np
import time
import threading
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

# Import other modules
from pointcloud import RealSenseThread
from Detection import BoltDetector
from robot_controller import RobotController
from gui import UnifiedRobotGUI

# Create a bridge class for Open3D rendering that doesn't open a separate window
import open3d as o3d

class Open3DRenderer:
    """A class to handle Open3D rendering in the main process without opening a window"""
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.vis = None
        self.pcd = None
        self.spheres = []
        self.initialized = False
        self.view_params_set = False  # Track whether view parameters have been set
        
        # Create visualizer for off-screen rendering
        self.vis = o3d.visualization.Visualizer()
        # Create a hidden window (won't be shown)
        self.vis.create_window(width=self.width, height=self.height, visible=False)
        
        # Create and add point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        
        # Configure view to match original view orientation
        view_control = self.vis.get_view_control()
        # In the original code, the view is set to look at the scene from a position
        # that works well with the rotated point cloud
        view_control.set_front([0, 0, -1])  # Looking along -Z (toward the scene)
        view_control.set_up([0, -1, 0])     # -Y is up
        view_control.change_field_of_view(step=25)  # Wide field of view

        # Set a camera distance that shows the scene well
        try:
            view_control.set_lookat([0, 0, 0.4])  # Look at this point in space
            view_control.set_zoom(0.5)  # Zoomed out enough to see context
        except Exception as e:
            print(f"[WARNING] Could not set camera position details: {e}")
        
        # Set rendering options
        render_option = self.vis.get_render_option()
        render_option.point_size = 3.0
        render_option.background_color = np.array([0.05, 0.05, 0.05])
        render_option.show_coordinate_frame = True
        
        # Store the original coordinate frame parameters - initialize after visualizer
        self.coord_frame_position = np.array([0, 0, 0])  # Fixed position
        self.coord_frame_size = 0.15  # Fixed size
        self.coord_frame = None  # Will be created later
        
        self.initialized = True
        print("[DEBUG] Open3D renderer initialized")
        
        # Create the coordinate frame with fixed properties - do this last
        self.create_fixed_coordinate_frame()
    
    def create_fixed_coordinate_frame(self):
        """Creates a coordinate frame with fixed properties that won't change with view transformations"""
        # Remove existing coordinate frame if present
        if hasattr(self, 'coord_frame') and self.coord_frame is not None:
            self.vis.remove_geometry(self.coord_frame)
        
        # Create a new coordinate frame
        self.coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=self.coord_frame_size, 
            origin=self.coord_frame_position
        )
        
        # Apply the rotation as in the original code
        rotation = np.array([
            [0, 1, 0],   # X (Red) axis points up
            [1, 0, 0],   # Y (Green) axis points right 
            [0, 0, -1]   # Z (Blue) axis points toward the flange
        ])
        
        # Create rotation matrix
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = rotation
        
        # Apply rotation around position
        self.coord_frame.translate(-self.coord_frame_position)
        self.coord_frame.transform(rotation_matrix)
        self.coord_frame.translate(self.coord_frame_position)
        
        # Add to visualizer
        self.vis.add_geometry(self.coord_frame)
        print("[DEBUG] Created fixed coordinate frame")
    
    def update_point_cloud(self, points, colors, camera_pose=None):
        """Update the point cloud data without affecting the coordinate frame"""
        if self.pcd is None or not self.initialized:
            return
            
        if len(points) == 0:
            return
            
        try:
            # First set the point cloud with the original points
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Apply the same rotation as in the original code to the point cloud
            rotation_matrix = np.array([
                [1, 0, 0],    # Leave X as is
                [0, -1, 0],   # Invert Y
                [0, 0, -1]    # Invert Z
            ])
            
            # Apply the rotation to the point cloud
            center = np.array([0, 0, 0])  # Rotate around origin
            self.pcd.rotate(rotation_matrix, center=center)
            
            # Update the geometry in the visualizer
            self.vis.update_geometry(self.pcd)
            
            # Update flag only on first call
            if not self.view_params_set:
                self.view_params_set = True
                
            # Make sure the coordinate frame remains in the scene
            if self.coord_frame is None:
                self.create_fixed_coordinate_frame()
                
            # Debug
            num_points = len(points)
            print(f"[DEBUG] Updated point cloud with {num_points} points")
                
        except Exception as e:
            print(f"[ERROR] Failed to update point cloud: {e}")
            import traceback
            traceback.print_exc()
    
    def update_coordinate_frame(self, position=None):
        """Only updates the coordinate frame when explicitly requested"""
        # This method now only recreates the coordinate frame at its fixed position
        # Ignoring any position parameter to keep it fixed
        self.create_fixed_coordinate_frame()
        print(f"[DEBUG] Refreshed fixed coordinate frame")
    
    def add_sphere(self, position, radius=0.005, color=(1, 0, 0)):
        """Add a sphere at the specified position"""
        try:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            sphere.paint_uniform_color(color)
            sphere.translate(position)
            self.vis.add_geometry(sphere)
            self.spheres.append(sphere)
            print(f"[DEBUG] Added sphere at {position}")
        except Exception as e:
            print(f"[ERROR] Failed to add sphere: {e}")
    
    def clear_spheres(self):
        """Remove all spheres from the visualizer"""
        try:
            for sphere in self.spheres:
                self.vis.remove_geometry(sphere)
            self.spheres = []
            print("[DEBUG] Cleared all spheres")
        except Exception as e:
            print(f"[ERROR] Failed to clear spheres: {e}")
    
    def reset_view(self):
        """Reset the camera view without changing the coordinate frame"""
        try:
            # Get the view control from the visualizer
            view_control = self.vis.get_view_control()
            
            # Set to match the original view
            view_control.set_front([0, 0, -1])  # Looking along -Z
            view_control.set_up([0, -1, 0])     # -Y is up
            view_control.change_field_of_view(step=25)
            
            # Try to set the camera pivot point
            try:
                view_control.set_lookat([0, 0, 0.4])
                view_control.set_zoom(0.5)
            except Exception as e:
                print(f"[WARNING] Could not set camera parameters on reset: {e}")
            
            # No need to update coordinate frame as it will stay fixed
            print("[DEBUG] View reset to match original view orientation")
        except Exception as e:
            print(f"[ERROR] Failed to reset view: {e}")
    
    def render_frame(self):
        """Render a frame and return the resulting image"""
        if not self.initialized:
            return None
            
        try:
            self.vis.poll_events()
            self.vis.update_renderer()
            img = self.vis.capture_screen_float_buffer(do_render=True)
            return np.asarray(img) if img is not None else None
        except Exception as e:
            print(f"[ERROR] Failed to render frame: {e}")
            return None
    
    def destroy(self):
        """Clean up resources"""
        if self.vis:
            self.vis.destroy_window()
            self.vis = None
            print("[DEBUG] Open3D renderer destroyed")

def main():
    """Main function that initializes components and runs the application"""
    print("[DEBUG] Running main application")
    
    try:
        # Create QApplication
        app = QApplication(sys.argv)
        
        # Initialize the RealSense thread
        print("[DEBUG] Initializing RealSense thread...")
        rs_thread = RealSenseThread()
        rs_thread.start()
        print("[DEBUG] RealSense thread started")
        
        # Create the Open3D renderer
        print("[DEBUG] Initializing Open3D renderer...")
        renderer = Open3DRenderer(width=640, height=480)
        
        # Create GUI with bolt detector and robot controller
        bolt_detector = BoltDetector(rs_thread)
        robot_controller = RobotController()
        window = UnifiedRobotGUI(rs_thread, bolt_detector, robot_controller, disable_open3d=True)
        window.show()
        
        # Connect signals between GUI and renderer
        window.bolt_detected.connect(lambda bolt_centers: handle_bolt_detection(renderer, bolt_centers))
        window.clear_bolts.connect(lambda: renderer.clear_spheres())
        window.view_reset.connect(lambda: renderer.reset_view())
        
        # Connect mouse signals from point cloud display to renderer
        window.point_cloud_display.mouse_moved.connect(lambda dx, dy: handle_mouse_movement(renderer, dx, dy))
        window.point_cloud_display.mouse_wheel.connect(lambda delta: handle_mouse_wheel(renderer, delta))
        
        # Connect panning signal directly to the handler function
        window.point_cloud_display.pan_signal.connect(lambda dx, dy: handle_mouse_pan(renderer, dx, dy))
        
        # Set up a timer for rendering and point cloud updates
        def update_visualization():
            # Only update the point cloud if live updates are enabled
            if window.live_updates and not window.frozen_cloud_data:
                if rs_thread.points_np is not None and len(rs_thread.points_np) > 0:
                    # Check if the point cloud has changed significantly
                    current_points_len = len(rs_thread.points_np)
                    if not hasattr(update_visualization, 'last_points_len') or abs(update_visualization.last_points_len - current_points_len) > 100:
                        # Make a copy of point cloud data to avoid race conditions
                        points = np.copy(rs_thread.points_np)
                        colors = np.copy(rs_thread.colors_np)
                        
                        # Only update if there are valid points
                        if len(points) > 0:
                            # Don't pass camera_pose after initial setup to prevent view resets
                            camera_pose = None
                            renderer.update_point_cloud(points, colors, camera_pose)
                            
                            # Update metrics in GUI
                            window.label_num_points.setText(f"Number of points: {len(points)}")
                            window.label_center_dist.setText(f"Center Dist: {rs_thread.center_distance:.3f} m")
                            window.label_fill_rate.setText(f"Fill Rate: {rs_thread.fill_rate * 100:.2f}%")
                            window.label_mean_error.setText(f"Mean Error: {rs_thread.mean_error:.3f} m")
                            window.label_std_error.setText(f"Std Dev: {rs_thread.std_error:.3f} m")
                            
                        # Remember the number of points for the next comparison
                        update_visualization.last_points_len = current_points_len
                
            # Render and update the display (always do this part)
            img = renderer.render_frame()
            if img is not None:
                window.point_cloud_display.update_image(img)
        
        # Create rendering timer
        render_timer = QTimer()
        render_timer.timeout.connect(update_visualization)
        render_timer.start(100)  # 10 fps for better performance
        
        # Run application
        exit_code = app.exec_()
        
        # Clean up resources
        rs_thread.stop()
        renderer.destroy()
        
        print("[DEBUG] Application closed with exit code:", exit_code)
        return exit_code
        
    except Exception as e:
        print(f"[ERROR] Error in main function: {e}")
        import traceback
        traceback.print_exc()
        return 1

def handle_bolt_detection(renderer, bolt_centers):
    """Handle bolt detection results by updating visualization"""
    if len(bolt_centers) > 0:
        # Clear existing spheres first
        renderer.clear_spheres()
        
        # Add new spheres for each bolt
        for center in bolt_centers:
            renderer.add_sphere(
                position=[float(center[0]), float(center[1]), float(center[2])],
                radius=0.005,
                color=[1, 0, 0]  # Red
            )

def handle_mouse_movement(renderer, dx, dy):
    """Handle mouse movement for camera rotation"""
    try:
        view_control = renderer.vis.get_view_control()
        
        # Rotation with adjusted sensitivity
        view_control.rotate(dx * 0.5, dy * 0.5)
        print(f"[DEBUG] Camera rotation: dx={dx}, dy={dy}")
    except Exception as e:
        print(f"[ERROR] Failed to handle mouse movement: {e}")

def handle_mouse_wheel(renderer, delta):
    """Handle mouse wheel for camera zoom"""
    try:
        view_control = renderer.vis.get_view_control()
        
        # More responsive zoom
        if delta > 0:
            view_control.change_field_of_view(step=-3)
        else:
            view_control.change_field_of_view(step=3)
        
        print(f"[DEBUG] Zoom: delta={delta}")
    except Exception as e:
        print(f"[ERROR] Failed to handle mouse wheel: {e}")

def handle_mouse_pan(renderer, dx, dy):
    """Handle mouse movement for camera panning"""
    try:
        view_control = renderer.vis.get_view_control()
        
        # Enhanced pan sensitivity
        pan_speed = 0.01  # Increased from 0.005 for more responsive panning
        view_control.translate(dx * pan_speed, dy * pan_speed)
        print(f"[DEBUG] Camera pan: dx={dx}, dy={dy}")
    except Exception as e:
        print(f"[ERROR] Failed to handle mouse pan: {e}")

if __name__ == "__main__":
    sys.exit(main())