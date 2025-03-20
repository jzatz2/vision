"""
This module defines the standard commands and data structures for communication
between the main GUI application and the separated Open3D visualization process.
"""

# Command Types
RESET_VIEW = 'reset_view'
ADD_SPHERE = 'add_sphere'
CLEAR_SPHERES = 'clear_spheres'
ADD_MESH = 'add_mesh'
REMOVE_GEOMETRY = 'remove_geometry'
SET_BACKGROUND_COLOR = 'set_background_color'

# Data Types
POINT_CLOUD_DATA = 'point_cloud_data'
IMAGE_DATA = 'image'
STATUS_UPDATE = 'status'

# Command Generators
def create_reset_view_command():
    """Create a command to reset the camera view"""
    return {'type': RESET_VIEW}

def create_add_sphere_command(position, radius=0.005, color=(1, 0, 0)):
    """
    Create a command to add a sphere
    
    Args:
        position (list): [x, y, z] position
        radius (float): Sphere radius
        color (tuple): RGB color tuple with values 0-1
    """
    return {
        'type': ADD_SPHERE,
        'position': position,
        'radius': radius,
        'color': color
    }

def create_clear_spheres_command():
    """Create a command to clear all spheres"""
    return {'type': CLEAR_SPHERES}

def create_set_background_color_command(color=(0.05, 0.05, 0.05)):
    """
    Create a command to set the background color
    
    Args:
        color (tuple): RGB color tuple with values 0-1
    """
    return {
        'type': SET_BACKGROUND_COLOR,
        'color': color
    }

# Data Packet Generators
def create_point_cloud_data_packet(points, colors):
    """
    Create a data packet with point cloud data
    
    Args:
        points (ndarray): Nx3 array of points
        colors (ndarray): Nx3 array of colors
    """
    return {
        'points': points,
        'colors': colors
    }

def create_image_data_packet(image_array):
    """
    Create a data packet with image data
    
    Args:
        image_array (ndarray): HxWx3 image array
    """
    return {
        'type': IMAGE_DATA,
        'data': image_array
    }