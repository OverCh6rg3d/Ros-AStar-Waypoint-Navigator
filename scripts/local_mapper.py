#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, MapMetaData
import math

class LocalMapper:
    def __init__(self, width=100, height=100, resolution=0.05):
        self.width = width
        self.height = height
        self.resolution = resolution

        rospy.init_node('local_mapper_node', anonymous=True)
        
        # publishes OccupancyGrid map
        self.map_pub = rospy.Publisher('/custom_local_map', OccupancyGrid, queue_size=1)
        # subscribes for Lidar info  
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        rospy.loginfo("Local Mapper Node Started. Listening to LiDAR...")

    def scan_callback(self, msg):
        """ Triggered every time the LiDAR completes a 360-degree sweep """
        
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "base_scan" # aligned with the robot's center
        
        # map configuration
        grid_msg.info.resolution = self.resolution 
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        
        # the center of the grid is the robot
        grid_msg.info.origin.position.x = -(self.width * self.resolution) / 2.0
        grid_msg.info.origin.position.y = -(self.height * self.resolution) / 2.0
        grid_msg.info.origin.orientation.w = 1.0
        
        # 0 = free space, 100 = obstacle
        # assume the whole map is free by default
        map_data = [0] * (self.width * self.height)
        
        # --- MATH TO CONVERT LASER RANGES TO GRID CELLS ---
        current_angle = msg.angle_min
        
        for r in msg.ranges:
            # Check if the laser actually hit something within its reliable range
            if msg.range_min < r < msg.range_max:
                
                # polar to cartesian
                x = r * math.cos(current_angle)
                y = r * math.sin(current_angle)
                
                # meters to grid cell coordinates
                grid_x = int((x - grid_msg.info.origin.position.x) / self.resolution)
                grid_y = int((y - grid_msg.info.origin.position.y) / self.resolution)
                
                # check if the calculated cell is actually inside our grid boundaries
                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    
                    # convert 2D coordinates to 1D array index to match OccupancyGrid() requirements
                    index = (grid_y * self.width) + grid_x
                    
                    # mark this cell as a solid obstacle
                    map_data[index] = 100 
            
            # increment the angle for the next laser beam
            current_angle += msg.angle_increment
        
        grid_msg.data = map_data
        self.map_pub.publish(grid_msg)

if __name__ == '__main__':
    try:
        mapper = LocalMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass