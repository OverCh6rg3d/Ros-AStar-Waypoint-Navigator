#!/usr/bin/env python3
from typing import Tuple, Optional
from nav_msgs.msg import MapMetaData

def world_to_grid(x: float, y: float, grid_info: MapMetaData) -> Optional[Tuple[int, int]]:
    """
    Converts Gazebo world coordinates (meters) to grid array indices.
    
    Args:
        x (float): The X coordinate in the 3D world.
        y (float): The Y coordinate in the 3D world.
        grid_info (MapMetaData): The map's configuration (resolution, origin).
        
    Returns:
        Tuple[int, int]: The corresponding (x, y) indices in the 2D grid array.
                         Returns None if map data is missing.
    """
    if not grid_info:
        return None
    
    grid_x = int((x - grid_info.origin.position.x) / grid_info.resolution)
    grid_y = int((y - grid_info.origin.position.y) / grid_info.resolution)
    return (grid_x, grid_y)

def grid_to_world(grid_x: int, grid_y: int, grid_info: MapMetaData) -> Optional[Tuple[float, float]]:
    """
    Converts grid array indices back to Gazebo world coordinates (meters).
    
    Args:
        grid_x (int): The X index in the 2D grid array.
        grid_y (int): The Y index in the 2D grid array.
        grid_info (MapMetaData): The map's configuration (resolution, origin).
        
    Returns:
        Tuple[float, float]: The corresponding (x, y) coordinates in the 3D world.
                             Returns None if map data is missing.
    """
    if not grid_info:
        return None
    
    # adds half a resolution to target the physical center of the grid cell
    x = (grid_x * grid_info.resolution) + grid_info.origin.position.x + (grid_info.resolution / 2.0)
    y = (grid_y * grid_info.resolution) + grid_info.origin.position.y + (grid_info.resolution / 2.0)
    return (x, y)