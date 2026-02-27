#!/usr/bin/env python3
import rospy
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from tf.transformations import euler_from_quaternion

class LocalMapper:
    def __init__(self):
        rospy.init_node('local_mapper_node', anonymous=True)
        
        # --- ROS PARAMETERS ---
        self.resolution = rospy.get_param('~resolution', 0.05)
        self.width = rospy.get_param('~width', 200)
        self.height = rospy.get_param('~height', 200)
        self.robot_radius = rospy.get_param('~robot_radius', 0.15) # Default 15cm
        
        # calculate C-Space inflation in grid cells to account for robot radius
        self.inflation_radius = int(self.robot_radius / self.resolution)
        
        # Publishers & Subscribers
        self.map_pub = rospy.Publisher('/custom_local_map', OccupancyGrid, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # State Variables
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        # Persistent memory array so the robot remembers obstacles across laser sweeps
        self.persistent_map = [0] * (self.width * self.height)
        
        rospy.loginfo(f"Mapper Started. Map: {self.width}x{self.height}, Inflation: {self.inflation_radius} cells.")

    def odom_callback(self, msg: Odometry) -> None:
        """ Updates the robot's pose for global map alignment. """
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        _, _, self.robot_yaw = euler_from_quaternion([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])

    def scan_callback(self, msg: LaserScan) -> None:
        """ Processes LiDAR data and generates the inflated obstacle grid. """
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "odom" 
        
        grid_msg.info.resolution = self.resolution 
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        
        # Center the map on the global origin
        grid_msg.info.origin.position.x = -(self.width * self.resolution) / 2.0
        grid_msg.info.origin.position.y = -(self.height * self.resolution) / 2.0
        grid_msg.info.origin.orientation.w = 1.0
        
        current_angle = msg.angle_min
        
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                # Project laser hit into global coordinates
                global_angle = current_angle + self.robot_yaw
                x = self.robot_x + (r * math.cos(global_angle))
                y = self.robot_y + (r * math.sin(global_angle))
                
                grid_x = int((x - grid_msg.info.origin.position.x) / self.resolution)
                grid_y = int((y - grid_msg.info.origin.position.y) / self.resolution)
                
                # Apply C-Space Inflation
                for dx in range(-self.inflation_radius, self.inflation_radius + 1):
                    for dy in range(-self.inflation_radius, self.inflation_radius + 1):
                        inf_x, inf_y = grid_x + dx, grid_y + dy
                        
                        if 0 <= inf_x < self.width and 0 <= inf_y < self.height:
                            if math.hypot(dx, dy) <= self.inflation_radius:
                                # Save directly to persistent memory
                                self.persistent_map[(inf_y * self.width) + inf_x] = 100 
            
            current_angle += msg.angle_increment
        
        # Publish the persistent memory map
        grid_msg.data = self.persistent_map
        self.map_pub.publish(grid_msg)

if __name__ == '__main__':
    try:
        mapper = LocalMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass