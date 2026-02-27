#!/usr/bin/env python3
import rospy
import math
import heapq
from typing import List, Tuple, Optional

# ROS Math and Messages
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist

# Custom Services and Helpers
from waypoint_nav.srv import AddWayPoint, AddWayPointRequest, AddWayPointResponse
import helpers

class WaypointNavigator:
    def __init__(self) -> None:
        rospy.init_node('waypoint_navigator_node', anonymous=True)
        
        # Publishers & Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.map_sub = rospy.Subscriber('/custom_local_map', OccupancyGrid, self.map_callback)
        
        # Service Server
        self.waypoint_service = rospy.Service('/add_waypoint', AddWayPoint, self.handle_add_waypoint)
        
        # State Variables
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.current_yaw: float = 0.0
        self.waypoints_queue: List[Tuple[float, float]] = []
        self.current_path: List[Tuple[float, float]] = []
        
        # Map Data
        self.current_grid: Optional[List[int]] = None
        self.grid_info: Optional[MapMetaData] = None
        
        rospy.loginfo("Waypoint Navigator Node Started. Waiting for waypoints...")

    # ==========================================
    # ROS CALLBACKS
    # ==========================================

    def map_callback(self, msg: OccupancyGrid) -> None:
        """ Updates the local map grid and metadata. """
        self.current_grid = msg.data
        self.grid_info = msg.info

    def odom_callback(self, msg: Odometry) -> None:
        """ Updates the robot's current position and Z-axis rotation. """
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Gets Z-axis from quaternion
        orientation_q = msg.pose.pose.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        _, _, yaw = euler_from_quaternion(orientation_list)
        
        self.current_yaw = yaw

    def handle_add_waypoint(self, req: AddWayPointRequest) -> AddWayPointResponse:
        """ Queues a new waypoint when the service is called. """
        new_waypoint = (req.x, req.y)
        self.waypoints_queue.append(new_waypoint)
        
        rospy.loginfo(f"New waypoint added: X={req.x}, Y={req.y}")
        return AddWayPointResponse(success=True, message="Waypoint queued successfully.")

    # ==========================================
    # NAVIGATION LOGIC
    # ==========================================

    def calculate_velocity_command(self, target_x: float, target_y: float) -> Tuple[Twist, bool]:
        """ 
        Calculates proportional control movement towards a target.
        Returns the velocity command and a boolean indicating if the target is reached.
        """
        cmd = Twist()
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        
        distance = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        
        # Calculate and normalize angle error between -pi and pi
        angle_error = target_angle - self.current_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        # Stop condition (within 0.1 meters)
        if distance < 0.1:
            return cmd, True  
            
        # Turning velocity
        cmd.angular.z = 1.5 * angle_error
        
        # Forward velocity (only drive if facing relatively forward)
        if abs(angle_error) < 0.5:
            cmd.linear.x = min(0.5 * distance, 0.22) # Max speed capped at 0.22
        else:
            cmd.linear.x = 0.0
            
        return cmd, False

    def a_star_search(self, start_world: Tuple[float, float], goal_world: Tuple[float, float]) -> List[Tuple[float, float]]:
        """ Calculates the shortest path avoiding obstacles using A*. """
        if not self.current_grid or not self.grid_info:
            return []

        # Helper functions to convert world coordinates to grid indices
        start_grid = helpers.world_to_grid(start_world[0], start_world[1], self.grid_info)
        goal_grid = helpers.world_to_grid(goal_world[0], goal_world[1], self.grid_info)
        
        if not start_grid or not goal_grid:
            return []

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

        # Priority queue to sort least to greatest path
        frontier = []
        heapq.heappush(frontier, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}

        while frontier:
            _, current_node = heapq.heappop(frontier)

            if current_node == goal_grid:
                # Reconstruct path backwards
                path = []
                while current_node in came_from:
                    world_coord = helpers.grid_to_world(current_node[0], current_node[1], self.grid_info)
                    if world_coord:
                        path.append(world_coord)
                    current_node = came_from[current_node]
                path.reverse()
                return path

            for dx, dy in directions:
                neighbor = (current_node[0] + dx, current_node[1] + dy)
                
                # Boundary check
                if not (0 <= neighbor[0] < self.grid_info.width and 0 <= neighbor[1] < self.grid_info.height):
                    continue
                
                # Obstacle check
                # find cell index because OccupancyGrid flattens 2D matrix into 1D array
                index = (neighbor[1] * self.grid_info.width) + neighbor[0]
                if self.current_grid[index] > 50: 
                    continue

                # Path cost calculation
                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0
                tentative_g_score = g_score[current_node] + move_cost

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    
                    # euclidean heuristic
                    h_score = math.hypot(goal_grid[0] - neighbor[0], goal_grid[1] - neighbor[1])
                    
                    f_score = tentative_g_score + h_score
                    
                    heapq.heappush(frontier, (f_score, neighbor))

        return []

    # ==========================================
    # MAIN LOOP
    # ==========================================

    def navigate_loop(self) -> None:
        """ The main control loop """
        rate = rospy.Rate(10)
        
        # variables for the stuck detection watchdog
        last_check_time = rospy.Time.now().to_sec()
        last_check_x = self.current_x
        last_check_y = self.current_y
        
        while not rospy.is_shutdown():
            if self.waypoints_queue:
                final_target_x, final_target_y = self.waypoints_queue[0]
                
                # ==========================================
                # 1. STUCK DETECTION
                # ==========================================
                current_time = rospy.Time.now().to_sec()
                
                # check every 3 seconds
                if current_time - last_check_time > 3.0:
                    dist_moved = math.hypot(self.current_x - last_check_x, self.current_y - last_check_y)
                    
                    # if moved less than 5 cm in 3 seconds, robot is stuck
                    if dist_moved < 0.05:
                        rospy.logerr(f"Robot is STUCK! Canceling waypoint ({final_target_x}, {final_target_y})")
                        self.waypoints_queue.pop(0)  # drop the impossible target
                        self.current_path = []       # clear the blocked path
                        self.cmd_vel_pub.publish(Twist()) # stop wheels
                        
                        # reset trackers and restart the loop to process the next waypoint
                        last_check_time = current_time
                        last_check_x = self.current_x
                        last_check_y = self.current_y
                        rate.sleep()
                        continue
                        
                    # if no stuck, update the snapshot for the next 3-second window
                    last_check_time = current_time
                    last_check_x = self.current_x
                    last_check_y = self.current_y

                # ==========================================
                # 2. PATHFINDING
                # ==========================================
                if not self.current_path:
                    start_world = (self.current_x, self.current_y)
                    goal_world = (final_target_x, final_target_y)
                    
                    self.current_path = self.a_star_search(start_world, goal_world)
                    
                    if not self.current_path:
                        rospy.logwarn("Target is completely blocked by walls! Canceling waypoint.")
                        self.waypoints_queue.pop(0) 
                        self.cmd_vel_pub.publish(Twist())
                        rate.sleep()
                        continue
                
                # ==========================================
                # 3. MOVEMENT EXECUTION
                # ==========================================
                sub_target_x, sub_target_y = self.current_path[0]
                cmd, target_reached = self.calculate_velocity_command(sub_target_x, sub_target_y)
                
                if target_reached:
                    self.current_path.pop(0)
                    
                    if not self.current_path:
                        rospy.loginfo(f"Final Waypoint reached! ({final_target_x}, {final_target_y})")
                        self.waypoints_queue.pop(0)
                
                self.cmd_vel_pub.publish(cmd)
                
            else:
                # Stop the robot when idle
                self.cmd_vel_pub.publish(Twist())
                
                # Keep the watchdog trackers updated while idle 
                # so it doesn't immediately trigger an error when the next command arrives
                last_check_time = rospy.Time.now().to_sec()
                last_check_x = self.current_x
                last_check_y = self.current_y
                
            rate.sleep()

if __name__ == '__main__':
    try:
        navigator = WaypointNavigator()
        navigator.navigate_loop()
    except rospy.ROSInterruptException:
        pass