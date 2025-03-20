import time
import numpy as np
import math
from PyQt5.QtCore import QObject, pyqtSignal, QThread

# Import XArm SDK
try:
    from xarm.wrapper import XArmAPI
except ImportError:
    print("[WARNING] XArm SDK not found. Robot control will be simulated.")
    XArmAPI = None

class RobotControllerSignals(QObject):
    """Signal class for robot controller to communicate with GUI"""
    connection_status = pyqtSignal(bool, str)
    movement_status = pyqtSignal(str)
    movement_progress = pyqtSignal(int, int)  # current bolt, total bolts
    operation_completed = pyqtSignal()
    error_occurred = pyqtSignal(str)

class RobotMovementThread(QThread):
    """QThread for robot movement operations"""
    
    def __init__(self, controller, bolt_positions, speed, z_offset):
        super().__init__()
        self.controller = controller
        self.bolt_positions = bolt_positions
        self.speed = speed
        self.z_offset = z_offset
        self.signals = controller.signals
        self.stop_flag = False
    
    def run(self):
        try:
            self.signals.movement_status.emit("Starting bolt unscrewing sequence")
            
            # Sort bolts radially around center
            if len(self.bolt_positions) > 1:
                self.bolt_positions = self._sort_bolts_radially(self.bolt_positions)
            
            # For simulation or when XArmAPI is not available
            if XArmAPI is None or self.controller.robot is None:
                self.signals.movement_status.emit("Simulating tool position movements")
                
                for i, bolt in enumerate(self.bolt_positions):
                    if self.stop_flag:
                        self.signals.movement_status.emit("Operation cancelled")
                        return
                        
                    # Convert from meters to millimeters for XArm
                    rel_x, rel_y, rel_z = bolt[0] * 1000, bolt[1] * 1000, bolt[2] * 1000
                    
                    # Create hover position with fixed hover distance
                    hover_pos = [rel_x, rel_y, rel_z + self.controller.z_offset, 0, 0, 0]
                    
                    self.signals.movement_status.emit(f"Moving to bolt {i+1}/{len(self.bolt_positions)}")
                    self.signals.movement_progress.emit(i+1, len(self.bolt_positions))
                    print(f"[DEBUG] Bolt {i+1} position: ({rel_x:.2f}, {rel_y:.2f}, {rel_z:.2f}) mm")
                    print(f"[DEBUG] Simulating move to hover position: {hover_pos}")
                    time.sleep(2)  # Simulate movement time
                    
                self.signals.movement_status.emit("Returning to home position")
                time.sleep(2)
                self.signals.operation_completed.emit()
                return
            
            # Real robot movement
            # First, move to home position
            self.signals.movement_status.emit("Moving to home position")
            code = self.controller.robot.set_position(*self.controller.home_position, speed=self.speed, wait=True)
            if code != 0:
                error_msg = f"Failed to move to home position: {code}"
                print(f"[ERROR] {error_msg}")
                self.signals.error_occurred.emit(error_msg)
                return
            
            # Define a fixed hover distance (50mm)
            hover_distance = 50  # mm
            
            # Process each bolt
            for i, bolt in enumerate(self.bolt_positions):
                if self.stop_flag:
                    self.signals.movement_status.emit("Operation cancelled")
                    break
                    
                # Convert from meters to millimeters for XArm
                rel_x, rel_y, rel_z = bolt[0] * 1000, bolt[1] * 1000, bolt[2] * 1000
                
                # Create hover position with fixed hover distance
                hover_pos = [rel_x, rel_y, rel_z + hover_distance, 0, 0, 0]
                
                self.signals.movement_status.emit(f"Moving to bolt {i+1}/{len(self.bolt_positions)}")
                self.signals.movement_progress.emit(i+1, len(self.bolt_positions))
                print(f"[DEBUG] Bolt {i+1} position: ({rel_x:.2f}, {rel_y:.2f}, {rel_z:.2f}) mm")
                
                # Move to hover position
                code = self.controller.robot.set_tool_position(*hover_pos, speed=self.speed, wait=True)
                if code != 0:
                    error_msg = f"Failed to move to hover position: {code}"
                    print(f"[ERROR] {error_msg}")
                    self.signals.error_occurred.emit(error_msg)
                    continue
                
                # Wait at the hover position
                self.signals.movement_status.emit(f"At bolt {i+1}/{len(self.bolt_positions)}. Waiting...")
                time.sleep(3)
            
            # Return to home position
            self.signals.movement_status.emit("Returning to home position")
            self.controller.robot.set_position(*self.controller.home_position, speed=self.speed, wait=True)
            
            self.signals.movement_status.emit("Unscrewing sequence completed")
            self.signals.operation_completed.emit()
            
        except Exception as e:
            error_msg = f"Error during unscrewing sequence: {e}"
            print(f"[ERROR] {error_msg}")
            import traceback
            traceback.print_exc()
            self.signals.error_occurred.emit(error_msg)
    
    def stop(self):
        """Signal the thread to stop"""
        self.stop_flag = True
    
    def _sort_bolts_radially(self, bolt_positions):
        """Sort bolts radially around their center"""
        # Calculate center of all bolts
        center_x = sum(bolt[0] for bolt in bolt_positions) / len(bolt_positions)
        center_y = sum(bolt[1] for bolt in bolt_positions) / len(bolt_positions)
        
        # Calculate angle for each bolt from center
        bolt_angles = []
        for i, bolt in enumerate(bolt_positions):
            angle = math.atan2(bolt[1] - center_y, bolt[0] - center_x)
            bolt_angles.append((i, angle))
        
        # Sort by angle
        bolt_angles.sort(key=lambda x: x[1])
        
        # Reorder bolt positions
        sorted_bolts = [bolt_positions[i] for i, _ in bolt_angles]
        
        print("[DEBUG] Bolts sorted radially around center")
        return np.array(sorted_bolts)

class RobotController(QObject):
    def __init__(self, robot_ip="192.168.1.203"):
        """Initialize the robot controller"""
        super().__init__()
        self.robot = None
        self.robot_ip = robot_ip
        self.connected = False
        self.home_position = [215, 0, 315, 180, -90, 0]  # Default home position [x, y, z, roll, pitch, yaw]
        self.z_offset = 150  # 0.15m in millimeters
        
        # Create signals object and thread
        self.signals = RobotControllerSignals()
        self.movement_thread = None
    
    def is_connected(self):
        """Check if the robot is connected"""
        return self.connected
    
    def set_z_offset(self, offset):
        """Set the Z offset value"""
        self.z_offset = offset
    
    def connect_robot(self):
        """Connect to the XArm robot"""
        try:
            if XArmAPI is None:
                # Simulate connection for testing
                print("[DEBUG] Simulating XArm connection (SDK not found)")
                self.connected = True
                self.signals.connection_status.emit(True, "Robot connection simulated (SDK not found)")
                return True, "Robot connection simulated (SDK not found)"

            print(f"[DEBUG] Connecting to XArm at {self.robot_ip}...")
            
            # Create a new XArmAPI instance
            self.robot = XArmAPI(self.robot_ip, baud_checkset=False)
            
            # Check connection
            if self.robot.connected:
                print("[DEBUG] Robot connected successfully")
                
                # Set up robot parameters
                self.robot.clean_error()
                self.robot.motion_enable(enable=True)
                self.robot.set_mode(0)  # Set to position mode
                self.robot.set_state(state=0)  # Set to ready state
                
                # Set safe move parameters
                self.robot.set_tcp_load(2, [0, 0, 0])  # Set tool weight (kg) and CoG
                self.robot.set_collision_sensitivity(1)  # Set collision sensitivity (1-5)
                
                self.connected = True
                self.signals.connection_status.emit(True, "Robot connected successfully")
                return True, "Robot connected successfully"
            else:
                print("[ERROR] Failed to connect to robot")
                self.signals.connection_status.emit(False, "Failed to connect to robot")
                return False, "Failed to connect to robot"
                
        except Exception as e:
            error_msg = f"Failed to connect to robot: {e}"
            print(f"[ERROR] {error_msg}")
            self.signals.connection_status.emit(False, error_msg)
            return False, error_msg
    
    def disconnect(self):
        """Disconnect from the robot"""
        # Stop any ongoing movements
        if self.movement_thread and self.movement_thread.isRunning():
            self.movement_thread.stop()
            self.movement_thread.wait()
        
        if XArmAPI is not None and self.robot is not None:
            try:
                self.robot.disconnect()
                print("[DEBUG] Robot disconnected")
            except:
                pass
            finally:
                self.connected = False
                self.robot = None
    
    def unscrew_bolts(self, bolt_positions, speed, z_offset=None):
        """
        Execute the unscrewing sequence for the given bolt positions in a separate thread
        
        Args:
            bolt_positions: List of [x, y, z] coordinates in meters
            speed: Movement speed in mm/s
            z_offset: Optional Z offset in mm (if None, use the default)
        """
        if z_offset is not None:
            self.z_offset = z_offset
        
        # If a movement is already in progress, don't start another one
        if self.movement_thread and self.movement_thread.isRunning():
            print("[DEBUG] Movement already in progress")
            self.signals.error_occurred.emit("Movement already in progress")
            return
        
        # Create and start movement thread
        self.movement_thread = RobotMovementThread(self, bolt_positions, speed, self.z_offset)
        self.movement_thread.start()