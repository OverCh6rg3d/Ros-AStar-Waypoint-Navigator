#!/usr/bin/env python3
import rospy
import math
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from tf.transformations import euler_from_quaternion

class LocalMapper:
    def __init__(self, width=200, height=200, resolution=0.05):
        self.width = width
        self.height = height
        self.resolution = resolution

        rospy.init_node('local_mapper_node', anonymous=True)
        
        self.map_pub = rospy.Publisher('/custom_local_map', OccupancyGrid, queue_size=1)

        # listen for lidar to identify occupied spaces
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # listen to Odometry to know our place in the global world
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        
        # State variables for robot pose
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        
        rospy.loginfo("Global-Aligned Mapper Node Started...")

    def odom_callback(self, msg):
        """ Continuously updates the robot's pose for map alignment """
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, self.robot_yaw = euler_from_quaternion(orientation_list)

    def scan_callback(self, msg):
        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = rospy.Time.now()
        grid_msg.header.frame_id = "odom" # aligned with the world
        
        grid_msg.info.resolution = self.resolution 
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        
        # center the map
        grid_msg.info.origin.position.x = -(self.width * self.resolution) / 2.0
        grid_msg.info.origin.position.y = -(self.height * self.resolution) / 2.0
        grid_msg.info.origin.orientation.w = 1.0
        
        map_data = [0] * (self.width * self.height)
        current_angle = msg.angle_min
        
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                
                # add the robot's rotation to the laser angle to get the true world angle
                global_angle = current_angle + self.robot_yaw
                
                # project the laser hit from the robot's current world position
                x = self.robot_x + (r * math.cos(global_angle))
                y = self.robot_y + (r * math.sin(global_angle))
                
                grid_x = int((x - grid_msg.info.origin.position.x) / self.resolution)
                grid_y = int((y - grid_msg.info.origin.position.y) / self.resolution)
                
                # to account for robot's size, everything inflated by 3
                inflation_radius = 3 
                
                for dx in range(-inflation_radius, inflation_radius + 1):
                    for dy in range(-inflation_radius, inflation_radius + 1):
                        inf_x = grid_x + dx
                        inf_y = grid_y + dy
                        
                        if 0 <= inf_x < self.width and 0 <= inf_y < self.height:
                            if math.hypot(dx, dy) <= inflation_radius:
                                index = (inf_y * self.width) + inf_x
                                map_data[index] = 100 
            
            current_angle += msg.angle_increment
        
        grid_msg.data = map_data
        self.map_pub.publish(grid_msg)

if __name__ == '__main__':
    try:
        mapper = LocalMapper()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass