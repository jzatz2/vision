import sys
import numpy as np
import math
import time
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox,
    QDoubleSpinBox, QPushButton, QMessageBox, QGroupBox, QGridLayout, QSplitter, QTabWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QMutex, QThread
from PyQt5.QtGui import QFont, QImage, QPixmap

class StatusIndicator(QLabel):
    def __init__(self, label_text="Status"):
        super().__init__()
        self.label_text = label_text
        self.setMinimumWidth(150)
        self.setMaximumHeight(30)
        self.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setBold(True)
        self.setFont(font)
        self.set_status(False)
        
    def set_status(self, connected):
        if connected:
            self.setText(f"{self.label_text}: Connected")
            self.setStyleSheet("background-color: #4CAF50; color: white; border-radius: 5px; padding: 3px;")
        else:
            self.setText(f"{self.label_text}: Not Connected")
            self.setStyleSheet("background-color: #F44336; color: white; border-radius: 5px; padding: 3px;")

class PointCloudDisplay(QLabel):
    mouse_moved = pyqtSignal(int, int)
    mouse_pressed = pyqtSignal(int, int)
    mouse_released = pyqtSignal(int, int)
    mouse_wheel = pyqtSignal(int)
    pan_signal = pyqtSignal(int, int)
    pan_started_signal = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #151515; border: 1px solid #444;")
        self.setText("Point Cloud Display\n(Waiting for data...)")
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.mouse_is_pressed = False
        self.last_pos = None
        self.pan_started = False
        self.right_button_pressed = False
        self.movement_threshold = 1  # Minimum pixel movement to trigger events
        
    def update_image(self, img_array):
        if img_array is None:
            return
        img_array = (img_array * 255).astype(np.uint8)
        h, w, c = img_array.shape
        bytes_per_line = 3 * w
        qt_image = QImage(img_array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.setPixmap(pixmap.scaled(self.width(), self.height(), Qt.KeepAspectRatio))
    
    def mousePressEvent(self, event):
        self.mouse_is_pressed = True
        self.last_pos = event.pos()
        
        # Left button for rotation
        if event.button() == Qt.LeftButton:
            self.right_button_pressed = False
            self.pan_started = False
            self.mouse_pressed.emit(event.x(), event.y())
        # Right or middle button for panning
        elif event.button() == Qt.RightButton or event.button() == Qt.MiddleButton:
            self.right_button_pressed = True
            self.pan_started = True
            self.pan_started_signal.emit(event.x(), event.y())
            print(f"[DEBUG] Pan started at {event.x()}, {event.y()}")
            
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.mouse_is_pressed = False
        self.right_button_pressed = False
        self.pan_started = False
        self.last_pos = None
        
        self.mouse_released.emit(event.x(), event.y())
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        if self.mouse_is_pressed and self.last_pos is not None:
            dx = event.x() - self.last_pos.x()
            dy = event.y() - self.last_pos.y()
            
            # Only emit signals if movement exceeds threshold
            if abs(dx) > self.movement_threshold or abs(dy) > self.movement_threshold:
                button_type = "Right button" if self.right_button_pressed else "Left button"
                print(f"[DEBUG] Mouse moved: {button_type}, dx={dx}, dy={dy}")
                
                # If panning is active (right button), emit pan signal
                if self.pan_started or self.right_button_pressed:
                    self.pan_signal.emit(dx, dy)
                    print(f"[DEBUG] Panning: dx={dx}, dy={dy}")
                else:
                    # Otherwise it's rotation (left button)
                    self.mouse_moved.emit(dx, dy)
                
                # Update last position after processing movement
                self.last_pos = event.pos()
        
        super().mouseMoveEvent(event)
    
    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        self.mouse_wheel.emit(delta)
        super().wheelEvent(event)

class ManualJointControl(QWidget):
    joint_value_changed = pyqtSignal(int, float)
    
    def __init__(self, num_joints=6):
        super().__init__()
        self.num_joints = num_joints
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.joint_sliders = []
        self.joint_spinboxes = []
        
        for i in range(num_joints):
            label = QLabel(f"J{i+1}:")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-1000, 1000)
            slider.setValue(0)
            slider.setTracking(True)
            slider.setObjectName(f"joint_slider_{i}")
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-180, 180)
            spinbox.setValue(0)
            spinbox.setSingleStep(1)
            spinbox.setObjectName(f"joint_spinbox_{i}")
            slider.valueChanged.connect(lambda v, idx=i: self._on_slider_changed(idx, v))
            spinbox.valueChanged.connect(lambda v, idx=i: self._on_spinbox_changed(idx, v))
            self.layout.addWidget(label, i, 0)
            self.layout.addWidget(slider, i, 1)
            self.layout.addWidget(spinbox, i, 2)
            self.joint_sliders.append(slider)
            self.joint_spinboxes.append(spinbox)
    
    def _on_slider_changed(self, joint_idx, value):
        joint_value = value / 1000 * 180
        self.joint_spinboxes[joint_idx].blockSignals(True)
        self.joint_spinboxes[joint_idx].setValue(joint_value)
        self.joint_spinboxes[joint_idx].blockSignals(False)
        self.joint_value_changed.emit(joint_idx, joint_value)
    
    def _on_spinbox_changed(self, joint_idx, value):
        slider_value = int(value / 180 * 1000)
        self.joint_sliders[joint_idx].blockSignals(True)
        self.joint_sliders[joint_idx].setValue(slider_value)
        self.joint_sliders[joint_idx].blockSignals(False)
        self.joint_value_changed.emit(joint_idx, value)
    
    def set_joint_values(self, values):
        for i, value in enumerate(values):
            if i < self.num_joints:
                self.joint_spinboxes[i].blockSignals(True)
                self.joint_spinboxes[i].setValue(value)
                self.joint_spinboxes[i].blockSignals(False)
                slider_value = int(value / 180 * 1000)
                self.joint_sliders[i].blockSignals(True)
                self.joint_sliders[i].setValue(slider_value)
                self.joint_sliders[i].blockSignals(False)

class PositionControl(QWidget):
    position_changed = pyqtSignal(list)
    
    def __init__(self):
        super().__init__()
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        pos_label = QLabel("Position (mm):")
        pos_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.layout.addWidget(pos_label, 0, 0, 1, 3)
        self.pos_labels = ["X:", "Y:", "Z:"]
        self.pos_spinboxes = []
        
        for i, label in enumerate(self.pos_labels):
            self.layout.addWidget(QLabel(label), i+1, 0)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-1000, 1000)
            spinbox.setValue(0)
            spinbox.setSingleStep(5)
            spinbox.setDecimals(1)
            spinbox.setObjectName(f"pos_{label.lower()[0]}")
            spinbox.valueChanged.connect(self._on_value_changed)
            self.layout.addWidget(spinbox, i+1, 1)
            self.pos_spinboxes.append(spinbox)
        
        orient_label = QLabel("Orientation (deg):")
        orient_label.setFont(QFont("Arial", 10, QFont.Bold))
        self.layout.addWidget(orient_label, 4, 0, 1, 3)
        self.orient_labels = ["Roll:", "Pitch:", "Yaw:"]
        self.orient_spinboxes = []
        
        for i, label in enumerate(self.orient_labels):
            self.layout.addWidget(QLabel(label), i+5, 0)
            spinbox = QDoubleSpinBox()
            spinbox.setRange(-180, 180)
            spinbox.setValue(0)
            spinbox.setSingleStep(5)
            spinbox.setDecimals(1)
            spinbox.setObjectName(f"orient_{label.lower()[:-1]}")
            spinbox.valueChanged.connect(self._on_value_changed)
            self.layout.addWidget(spinbox, i+5, 1)
            self.orient_spinboxes.append(spinbox)
        
        self.move_button = QPushButton("Move To Position")
        self.move_button.clicked.connect(self._on_move_clicked)
        self.layout.addWidget(self.move_button, 8, 0, 1, 2)
    
    def _on_value_changed(self):
        pass
    
    def _on_move_clicked(self):
        values = []
        for spinbox in self.pos_spinboxes:
            values.append(spinbox.value())
        for spinbox in self.orient_spinboxes:
            values.append(spinbox.value())
        self.position_changed.emit(values)
    
    def set_position_values(self, values):
        if len(values) >= 6:
            for i in range(3):
                self.pos_spinboxes[i].blockSignals(True)
                self.pos_spinboxes[i].setValue(values[i])
                self.pos_spinboxes[i].blockSignals(False)
            for i in range(3):
                self.orient_spinboxes[i].blockSignals(True)
                self.orient_spinboxes[i].setValue(values[i+3])
                self.orient_spinboxes[i].blockSignals(False)

class UnifiedRobotGUI(QMainWindow):
    # Add signals for bolt detection and visualization
    bolt_detected = pyqtSignal(np.ndarray)
    clear_bolts = pyqtSignal()
    view_reset = pyqtSignal()
    
    def __init__(self, rs_thread, bolt_detector=None, robot_controller=None, disable_open3d=False):
        super().__init__()
        print("[DEBUG] Initializing Unified GUI")
        self.setWindowTitle("Robot Control & Point Cloud Visualization")
        
        self.rs_thread = rs_thread
        
        # Create components if not provided
        if bolt_detector is None:
            from Detection import BoltDetector
            self.bolt_detector = BoltDetector(rs_thread)
        else:
            self.bolt_detector = bolt_detector
            
        if robot_controller is None:
            from robot_controller import RobotController
            self.robot_controller = RobotController()
        else:
            self.robot_controller = robot_controller
        
        self.bolt_detector.signals.model_loaded.connect(self.on_model_loaded)
        self.bolt_detector.signals.detection_complete.connect(self.on_detection_complete)
        self.robot_controller.signals.connection_status.connect(self.on_robot_connection_status)
        self.robot_controller.signals.movement_status.connect(self.on_robot_movement_status)
        self.robot_controller.signals.operation_completed.connect(self.on_operation_completed)
        
        # State variables
        self.bolt_positions = []
        self.live_updates = True
        self.frozen_cloud_data = False
        self.is_unscrewing = False
        self.disable_open3d = disable_open3d
        
        self.camera_status = StatusIndicator("Camera")
        self.robot_status = StatusIndicator("Robot")
        self.status_message = QLabel("Ready")
        
        self.point_cloud_display = PointCloudDisplay()
        self.resize(1280, 800)
        self._setup_ui()
        
        # Setup timer for status updates
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self.update_status)
        self.status_timer.start(500)

    def _setup_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)
        self.left_panel = QWidget()
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setContentsMargins(5, 5, 5, 5)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)
        self.right_layout.setContentsMargins(0, 0, 0, 0)
        self.right_layout.setSpacing(0)
        self.splitter.addWidget(self.left_panel)
        self.splitter.addWidget(self.right_panel)
        self.splitter.setSizes([int(self.width() * 0.2), int(self.width() * 0.8)])
        self.right_layout.addWidget(self.point_cloud_display, 1)
        self.tabs = QTabWidget()
        self.left_layout.addWidget(self.tabs)
        self.point_cloud_tab = QWidget()
        self.point_cloud_layout = QVBoxLayout(self.point_cloud_tab)
        self.tabs.addTab(self.point_cloud_tab, "Point Cloud")
        self.robot_tab = QWidget()
        self.robot_layout = QVBoxLayout(self.robot_tab)
        self.tabs.addTab(self.robot_tab, "Robot Control")
        self._add_point_cloud_controls()
        self._add_robot_controls()
        self._add_status_bar()

    def _add_point_cloud_controls(self):
        actions_group = QGroupBox("Actions")
        actions_layout = QHBoxLayout()
        actions_group.setLayout(actions_layout)
        
        self.detect_bolts_button = QPushButton("Detect Bolts")
        self.detect_bolts_button.clicked.connect(self.detect_bolts)
        actions_layout.addWidget(self.detect_bolts_button)
        
        self.unscrew_bolts_button = QPushButton("Unscrew Bolts")
        self.unscrew_bolts_button.clicked.connect(self.unscrew_bolts)
        self.unscrew_bolts_button.setEnabled(False)
        actions_layout.addWidget(self.unscrew_bolts_button)
        
        self.resume_updates_button = QPushButton("Resume Live Updates")
        self.resume_updates_button.clicked.connect(self.resume_live_updates)
        self.resume_updates_button.setEnabled(False)
        actions_layout.addWidget(self.resume_updates_button)
        
        # Add reset view button
        self.reset_view_button = QPushButton("Reset View")
        self.reset_view_button.clicked.connect(self.reset_view)
        actions_layout.addWidget(self.reset_view_button)
        
        self.point_cloud_layout.addWidget(actions_group)

        # Filter controls
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout()
        filter_group.setLayout(filter_layout)
        decimation_layout = QHBoxLayout()
        decimation_layout.addWidget(QLabel("Decimation:"))
        self.decimation_slider = QSlider(Qt.Horizontal)
        self.decimation_slider.setRange(1, 8)
        self.decimation_slider.setValue(2)
        self.decimation_slider.valueChanged.connect(self.on_decimation_changed)
        decimation_layout.addWidget(self.decimation_slider)
        self.decimation_checkbox = QCheckBox("Enable")
        self.decimation_checkbox.setChecked(True)
        self.decimation_checkbox.stateChanged.connect(self.on_decimation_toggled)
        decimation_layout.addWidget(self.decimation_checkbox)
        filter_layout.addLayout(decimation_layout)
        temporal_layout = QHBoxLayout()
        temporal_layout.addWidget(QLabel("Temporal Filter:"))
        self.temporal_checkbox = QCheckBox("Enable")
        self.temporal_checkbox.setChecked(True)
        self.temporal_checkbox.stateChanged.connect(self.on_temporal_toggled)
        temporal_layout.addWidget(self.temporal_checkbox)
        filter_layout.addLayout(temporal_layout)
        exposure_layout = QHBoxLayout()
        exposure_layout.addWidget(QLabel("Auto-Exposure:"))
        self.auto_exposure_checkbox = QCheckBox("Enable")
        self.auto_exposure_checkbox.setChecked(True)
        self.auto_exposure_checkbox.stateChanged.connect(self.on_auto_exposure_toggled)
        exposure_layout.addWidget(self.auto_exposure_checkbox)
        filter_layout.addLayout(exposure_layout)
        self.point_cloud_layout.addWidget(filter_group)

        # ROI controls
        roi_group = QGroupBox("Region of Interest (ROI)")
        roi_layout = QGridLayout()
        roi_group.setLayout(roi_layout)
        roi_layout.addWidget(QLabel("X Min (m):"), 0, 0)
        self.x_min_spin = QDoubleSpinBox()
        self.x_min_spin.setRange(-1.0, 1.0)
        self.x_min_spin.setValue(-0.25)  # Widened from -0.20
        self.x_min_spin.setSingleStep(0.01)
        self.x_min_spin.valueChanged.connect(self.on_xmin_changed)
        roi_layout.addWidget(self.x_min_spin, 0, 1)
        roi_layout.addWidget(QLabel("X Max (m):"), 0, 2)
        self.x_max_spin = QDoubleSpinBox()
        self.x_max_spin.setRange(-1.0, 1.0)
        self.x_max_spin.setValue(0.25)  # Widened from 0.20
        self.x_max_spin.setSingleStep(0.01)
        self.x_max_spin.valueChanged.connect(self.on_xmax_changed)
        roi_layout.addWidget(self.x_max_spin, 0, 3)
        roi_layout.addWidget(QLabel("Y Min (m):"), 1, 0)
        self.y_min_spin = QDoubleSpinBox()
        self.y_min_spin.setRange(-1.0, 1.0)
        self.y_min_spin.setValue(-0.25)  # Widened from -0.20
        self.y_min_spin.setSingleStep(0.01)
        self.y_min_spin.valueChanged.connect(self.on_ymin_changed)
        roi_layout.addWidget(self.y_min_spin, 1, 1)
        roi_layout.addWidget(QLabel("Y Max (m):"), 1, 2)
        self.y_max_spin = QDoubleSpinBox()
        self.y_max_spin.setRange(-1.0, 1.0)
        self.y_max_spin.setValue(0.25)  # Widened from 0.20
        self.y_max_spin.setSingleStep(0.01)
        self.y_max_spin.valueChanged.connect(self.on_ymax_changed)
        roi_layout.addWidget(self.y_max_spin, 1, 3)
        roi_layout.addWidget(QLabel("Z Min (m):"), 2, 0)
        self.z_min_spin = QDoubleSpinBox()
        self.z_min_spin.setRange(0.0, 2.0)
        self.z_min_spin.setValue(0.1)  # Widened from 0.35
        self.z_min_spin.setSingleStep(0.01)
        self.z_min_spin.valueChanged.connect(self.on_zmin_changed)
        roi_layout.addWidget(self.z_min_spin, 2, 1)
        roi_layout.addWidget(QLabel("Z Max (m):"), 2, 2)
        self.z_max_spin = QDoubleSpinBox()
        self.z_max_spin.setRange(0.0, 2.0)
        self.z_max_spin.setValue(0.60)  # Widened from 0.55
        self.z_max_spin.setSingleStep(0.01)
        self.z_max_spin.valueChanged.connect(self.on_zmax_changed)
        roi_layout.addWidget(self.z_max_spin, 2, 3)
        self.point_cloud_layout.addWidget(roi_group)
        
        # Detection parameters
        detection_group = QGroupBox("Detection Parameters")
        detection_layout = QGridLayout()
        detection_group.setLayout(detection_layout)
        detection_layout.addWidget(QLabel("Bolt Detection Scale:"), 0, 0)
        self.bolt_scale_spin = QDoubleSpinBox()
        self.bolt_scale_spin.setRange(0.1, 2.0)
        self.bolt_scale_spin.setValue(0.5)
        self.bolt_scale_spin.setSingleStep(0.05)
        detection_layout.addWidget(self.bolt_scale_spin, 0, 1)
        self.point_cloud_layout.addWidget(detection_group)
        
        # Metrics display
        metrics_group = QGroupBox("Metrics")
        metrics_layout = QVBoxLayout()
        metrics_group.setLayout(metrics_layout)
        self.label_center_dist = QLabel("Center Dist: 0.000 m")
        metrics_layout.addWidget(self.label_center_dist)
        self.label_fill_rate = QLabel("Fill Rate: 0.00%")
        metrics_layout.addWidget(self.label_fill_rate)
        self.label_mean_error = QLabel("Mean Error: 0.000 m")
        metrics_layout.addWidget(self.label_mean_error)
        self.label_std_error = QLabel("Std Dev: 0.000 m")
        metrics_layout.addWidget(self.label_std_error)
        self.point_cloud_layout.addWidget(metrics_group)
        
        # Debug info
        debug_group = QGroupBox("Debug Info")
        debug_layout = QVBoxLayout()
        debug_group.setLayout(debug_layout)
        self.label_num_points = QLabel("Number of points: 0")
        debug_layout.addWidget(self.label_num_points)
        self.point_cloud_layout.addWidget(debug_group)
        
        # Add stretch at the end
        self.point_cloud_layout.addStretch(1)

    def _add_robot_controls(self):
        connect_layout = QHBoxLayout()
        self.connect_robot_button = QPushButton("Connect Robot")
        self.connect_robot_button.clicked.connect(self.connect_robot)
        connect_layout.addWidget(self.connect_robot_button)
        self.robot_layout.addLayout(connect_layout)
        motion_group = QGroupBox("Motion Settings")
        motion_layout = QGridLayout()
        motion_group.setLayout(motion_layout)
        motion_layout.addWidget(QLabel("Speed (mm/s):"), 0, 0)
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(0.1, 100.0)
        self.speed_spin.setValue(5)
        self.speed_spin.setSingleStep(5.0)
        motion_layout.addWidget(self.speed_spin, 0, 1)
        motion_layout.addWidget(QLabel("Z Offset (mm):"), 1, 0)
        self.z_offset_spin = QDoubleSpinBox()
        self.z_offset_spin.setRange(10.0, 300.0)
        self.z_offset_spin.setValue(150.0)
        self.z_offset_spin.setSingleStep(10.0)
        self.z_offset_spin.valueChanged.connect(self.on_z_offset_changed)
        motion_layout.addWidget(self.z_offset_spin, 1, 1)
        self.robot_layout.addWidget(motion_group)
        position_group = QGroupBox("Position Control")
        position_layout = QVBoxLayout()
        position_group.setLayout(position_layout)
        self.position_control = PositionControl()
        self.position_control.position_changed.connect(self.on_position_changed)
        position_layout.addWidget(self.position_control)
        self.robot_layout.addWidget(position_group)
        joint_group = QGroupBox("Joint Control")
        joint_layout = QVBoxLayout()
        joint_group.setLayout(joint_layout)
        self.joint_control = ManualJointControl()
        self.joint_control.joint_value_changed.connect(self.on_joint_changed)
        joint_layout.addWidget(self.joint_control)
        self.robot_layout.addWidget(joint_group)
        self.emergency_stop_button = QPushButton("EMERGENCY STOP")
        self.emergency_stop_button.setMinimumHeight(50)
        self.emergency_stop_button.setStyleSheet("background-color: #F44336; color: white; font-weight: bold; font-size: 16px;")
        self.emergency_stop_button.clicked.connect(self.emergency_stop)
        self.robot_layout.addWidget(self.emergency_stop_button)
        status_group = QGroupBox("Robot Status")
        status_layout = QVBoxLayout()
        status_group.setLayout(status_layout)
        self.label_robot_movement = QLabel("Status: Idle")
        status_layout.addWidget(self.label_robot_movement)
        self.robot_layout.addWidget(status_group)
        self.robot_layout.addStretch(1)

    def _add_status_bar(self):
        status_bar = QWidget()
        status_bar_layout = QHBoxLayout(status_bar)
        status_bar_layout.setContentsMargins(5, 5, 5, 5)
        status_bar.setMaximumHeight(40)
        status_bar_layout.addWidget(self.camera_status)
        status_bar_layout.addWidget(self.robot_status)
        status_bar_layout.addStretch(1)
        status_bar_layout.addWidget(self.status_message)
        self.right_layout.addWidget(status_bar)

    def update_point_cloud(self):
        """Update the point cloud display with data from RealSense thread"""
        if not self.live_updates or self.frozen_cloud_data:
            return
            
        # Skip if no point cloud data is available
        if self.rs_thread.points_np is None or len(self.rs_thread.points_np) == 0:
            return
            
        # Update metrics display
        point_count = len(self.rs_thread.points_np)
        self.label_num_points.setText(f"Number of points: {point_count}")
        self.label_center_dist.setText(f"Center Dist: {self.rs_thread.center_distance:.3f} m")
        self.label_fill_rate.setText(f"Fill Rate: {self.rs_thread.fill_rate * 100:.2f}%")
        self.label_mean_error.setText(f"Mean Error: {self.rs_thread.mean_error:.3f} m")
        self.label_std_error.setText(f"Std Dev: {self.rs_thread.std_error:.3f} m")

    def reset_view(self):
        """Reset the camera view to default position"""
        self.view_reset.emit()
        self.status_message.setText("View reset")
        print("[DEBUG] View reset requested")

    def update_status(self):
        """Update status indicators"""
        camera_connected = self.rs_thread is not None and self.rs_thread.points_np is not None
        self.camera_status.set_status(camera_connected)
        robot_connected = self.robot_controller.is_connected()
        self.robot_status.set_status(robot_connected)

    def detect_bolts(self):
        """Initiate bolt detection - first load model if needed"""
        print("[DEBUG] Detecting bolts...")
        self.frozen_cloud_data = True
        self.status_message.setText("Starting bolt detection...")
        if self.bolt_detector.is_model_loaded():
            self.detect_bolts_real()
        else:
            self.detect_bolts_button.setEnabled(False)
            self.detect_bolts_button.setText("Loading Model...")
            QApplication.processEvents()
            self.bolt_detector.load_model()

    def detect_bolts_real(self):
        """Perform the actual bolt detection after model is loaded"""
        self.detect_bolts_button.setEnabled(False)
        self.detect_bolts_button.setText("Detecting...")
        QApplication.processEvents()
        self.frozen_cloud_data = True
        self.live_updates = False
        self.resume_updates_button.setEnabled(True)
        try:
            self.bolt_detector.detect_bolts()
        except Exception as e:
            print(f"[ERROR] Error during bolt detection: {e}")
            import traceback
            traceback.print_exc()
            self.status_message.setText(f"Error: {str(e)}")
            self.detect_bolts_button.setEnabled(True)
            self.detect_bolts_button.setText("Detect Bolts")

    def on_detection_complete(self, bolt_centers):
        """Handle completion of bolt detection"""
        try:
            if len(bolt_centers) > 0:
                bolt_centers = self.transform_bolt_coordinates(bolt_centers)
                print(f"[DEBUG] Transformed bolt centers: {bolt_centers}")
                self.bolt_positions = bolt_centers
                self.unscrew_bolts_button.setEnabled(True)
                self.status_message.setText(f"Detected {len(bolt_centers)} bolts")
                
                # Emit signal to visualize bolt centers
                self.bolt_detected.emit(bolt_centers)
            else:
                self.bolt_positions = []
                self.unscrew_bolts_button.setEnabled(False)
                self.status_message.setText("No bolts detected")
                
                # Emit signal to clear bolt visualizations
                self.clear_bolts.emit()
                
        except Exception as e:
            print(f"[ERROR] Error processing detection results: {e}")
            import traceback
            traceback.print_exc()
            self.status_message.setText(f"Error: {str(e)}")
            
        finally:
            self.detect_bolts_button.setEnabled(True)
            self.detect_bolts_button.setText("Detect Bolts")

    def transform_bolt_coordinates(self, bolt_centers):
        """Apply a transformation to align bolt coordinates with point cloud"""
        transformed = []
        scale_factor = self.bolt_scale_spin.value()
        for center in bolt_centers:
            x, y, z = center
            transformed.append([x * scale_factor, y * scale_factor, z])
        return np.array(transformed)

    def resume_live_updates(self):
        """Resume live updates of the point cloud"""
        if self.is_unscrewing:
            QMessageBox.warning(self, "Operation in Progress", 
                               "Cannot resume live updates while unscrewing operation is in progress")
            return
            
        # Clear bolt visualizations
        self.clear_bolts.emit()
        
        self.frozen_cloud_data = False
        self.live_updates = True
        self.resume_updates_button.setEnabled(False)
        self.status_message.setText("Resumed live updates")
        print("[DEBUG] Resumed live updates")

    def connect_robot(self):
        """Connect to the robot using the robot controller"""
        self.status_message.setText("Connecting to robot...")
        status, message = self.robot_controller.connect_robot()
        if status:
            self.robot_status.set_status(True)
            self.status_message.setText("Robot connected successfully")
            QMessageBox.information(self, "Robot Connection", message)
        else:
            self.robot_status.set_status(False)
            self.status_message.setText(f"Robot connection failed: {message}")
            QMessageBox.critical(self, "Robot Connection Error", message)

    def unscrew_bolts(self):
        """Move the robot to each detected bolt position, wait, and then return home"""
        if not self.robot_controller.is_connected():
            QMessageBox.warning(self, "Robot Not Connected", 
                               "Please connect the robot before unscrewing bolts")
            return
            
        if len(self.bolt_positions) == 0:
            QMessageBox.warning(self, "No Bolts Detected", 
                               "Please detect bolts before unscrewing")
            return
        
        # Ask for confirmation
        reply = QMessageBox.question(self, "Confirm Operation", 
                                    f"Move robot to unscrew {len(self.bolt_positions)} bolts?",
                                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
            
        # Start unscrewing operation
        self.is_unscrewing = True
        
        # Disable buttons during operation
        self.unscrew_bolts_button.setEnabled(False)
        self.detect_bolts_button.setEnabled(False)
        self.resume_updates_button.setEnabled(False)
        self.connect_robot_button.setEnabled(False)
        
        # Get settings
        speed = self.speed_spin.value()
        z_offset = self.z_offset_spin.value()
        
        # Start the unscrewing operation
        self.status_message.setText("Starting unscrewing sequence...")
        self.robot_controller.unscrew_bolts(self.bolt_positions, speed, z_offset)

    def _update_ui_after_unscrewing(self):
        """Update UI after the unscrewing operation"""
        self.is_unscrewing = False
        self.unscrew_bolts_button.setEnabled(True)
        self.detect_bolts_button.setEnabled(True)
        self.resume_updates_button.setEnabled(True)
        self.connect_robot_button.setEnabled(True)
        self.status_message.setText("Bolt unscrewing sequence completed")
        QMessageBox.information(self, "Operation Complete", "Bolt unscrewing sequence completed")

    def on_position_changed(self, position_values):
        """Handle position control change"""
        if not self.robot_controller.is_connected():
            QMessageBox.warning(self, "Robot Not Connected", 
                               "Please connect the robot before sending movement commands")
            return
        
        try:
            print(f"[DEBUG] Moving robot to position: {position_values}")
            self.status_message.setText(f"Moving to position: {position_values}")
            
            # Here you would call the appropriate robot controller method
            # Example: self.robot_controller.move_to_position(position_values)
            
        except Exception as e:
            self.status_message.setText(f"Error: {str(e)}")
            QMessageBox.critical(self, "Movement Error", str(e))

    def on_joint_changed(self, joint_idx, value):
        """Handle joint control changes"""
        if not self.robot_controller.is_connected():
            return  # Don't warn here, as this happens during UI updates
        
        try:
            print(f"[DEBUG] Moving joint {joint_idx+1} to {value} degrees")
            
            # Here you would call the appropriate robot controller method
            # Example: self.robot_controller.move_joint(joint_idx, value)
            
        except Exception as e:
            self.status_message.setText(f"Error: {str(e)}")

    def emergency_stop(self):
        """Emergency stop all robot movement"""
        try:
            print("[DEBUG] EMERGENCY STOP triggered")
            self.status_message.setText("EMERGENCY STOP triggered")
            
            # Here you would call the emergency stop method on the robot controller
            # Example: self.robot_controller.emergency_stop()
            
            QMessageBox.warning(self, "Emergency Stop", "Robot movement stopped")
            
        except Exception as e:
            self.status_message.setText(f"Error during emergency stop: {str(e)}")
            QMessageBox.critical(self, "Emergency Stop Error", str(e))

    def on_model_loaded(self, success, message):
        """Callback when model loading is complete"""
        if success:
            print("[DEBUG] Model loaded successfully via BoltDetector")
            self.status_message.setText("Model loaded successfully")
            self.detect_bolts_real()
        else:
            print(f"[ERROR] Failed to load model: {message}")
            self.status_message.setText(f"Error: {message}")
            QMessageBox.critical(self, "Model Loading Error", f"Failed to load detection model: {message}")
            self.detect_bolts_button.setText("Detect Bolts")
            self.detect_bolts_button.setEnabled(True)
            self.frozen_cloud_data = False
            self.live_updates = True

    def on_robot_connection_status(self, connected, message):
        """Handle robot connection status changes"""
        if connected:
            self.robot_status.set_status(True)
            self.status_message.setText(message)
        else:
            self.robot_status.set_status(False)
            self.status_message.setText(message)

    def on_robot_movement_status(self, status):
        """Update robot movement status"""
        self.label_robot_movement.setText(f"Status: {status}")
        self.status_message.setText(status)

    def on_operation_completed(self):
        """Handle operation completed signal"""
        self._update_ui_after_unscrewing()

    # Event handlers for ROI and filter settings
    def on_decimation_changed(self, value):
        self.rs_thread.set_decimation_magnitude(value)
        
    def on_decimation_toggled(self, state):
        self.rs_thread.enable_decimation(state == Qt.Checked)
        
    def on_temporal_toggled(self, state):
        self.rs_thread.enable_temporal(state == Qt.Checked)
        
    def on_auto_exposure_toggled(self, state):
        self.rs_thread.enable_auto_exposure(state == Qt.Checked)
        
    def on_z_offset_changed(self, value):
        self.robot_controller.set_z_offset(value)
        
    def on_xmin_changed(self, value):
        self.rs_thread.set_xmin(value)
        
    def on_xmax_changed(self, value):
        self.rs_thread.set_xmax(value)
        
    def on_ymin_changed(self, value):
        self.rs_thread.set_ymin(value)
        
    def on_ymax_changed(self, value):
        self.rs_thread.set_ymax(value)
        
    def on_zmin_changed(self, value):
        self.rs_thread.set_zmin(value)
        
    def on_zmax_changed(self, value):
        self.rs_thread.set_zmax(value)
    
    def resizeEvent(self, event):
        """Handle window resize event"""
        super().resizeEvent(event)
        
        # Adjust splitter proportions on resize
        if event.size().width() > 0:
            # Maintain 20/80 split
            self.splitter.setSizes([int(self.width() * 0.2), int(self.width() * 0.8)])
    
    def closeEvent(self, event):
        """Clean up resources when window is closed"""
        # Stop any ongoing operations
        self.is_unscrewing = False
        
        # Disconnect the robot controller
        self.robot_controller.disconnect()
        
        super().closeEvent(event)