#!/usr/bin/env python3
import rospy
import math
import heapq
from typing import List, Tuple, Optional

from tf.transformations import euler_from_quaternion
from nav_msgs.msg import OccupancyGrid, MapMetaData, Odometry
from geometry_msgs.msg import Twist
from waypoint_nav.srv import AddWayPoint, AddWayPointRequest, AddWayPointResponse
import helpers

class WaypointNavigator:
    def __init__(self) -> None:
        rospy.init_node('waypoint_navigator_node', anonymous=True)
        
        # --- ROS PARAMETERS ---
        self.max_v = rospy.get_param('~max_linear_speed', 0.22)
        self.p_gain_turn = rospy.get_param('~turn_speed_multiplier', 1.5)
        self.p_gain_fwd = rospy.get_param('~forward_speed_multiplier', 0.5)
        self.goal_tol = rospy.get_param('~goal_tolerance', 0.1)
        self.stuck_timeout = rospy.get_param('~stuck_timeout', 3.0)
        self.stuck_dist = rospy.get_param('~stuck_distance', 0.05)
        
        # Publishers & Subscribers
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)
        self.map_sub = rospy.Subscriber('/custom_local_map', OccupancyGrid, self.map_callback)
        self.waypoint_service = rospy.Service('/add_waypoint', AddWayPoint, self.handle_add_waypoint)
        
        # State Variables
        self.current_x: float = 0.0
        self.current_y: float = 0.0
        self.current_yaw: float = 0.0
        self.waypoints_queue: List[Tuple[float, float]] = []
        self.current_path: List[Tuple[float, float]] = []
        
        # Map & Watchdog Data
        self.current_grid: Optional[List[int]] = None
        self.grid_info: Optional[MapMetaData] = None
        self.last_check_time = rospy.Time.now().to_sec()
        self.last_check_x = 0.0
        self.last_check_y = 0.0
        
        rospy.loginfo("Navigator Started. Parameters loaded. Waiting for waypoints...")

    # --- CALLBACKS ---

    def map_callback(self, msg: OccupancyGrid) -> None:
        self.current_grid = msg.data
        self.grid_info = msg.info

    def odom_callback(self, msg: Odometry) -> None:
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])

    def handle_add_waypoint(self, req: AddWayPointRequest) -> AddWayPointResponse:
        self.waypoints_queue.append((req.x, req.y))
        rospy.loginfo(f"Queued new waypoint: X={req.x}, Y={req.y}")
        return AddWayPointResponse(success=True, message="Waypoint queued.")

    # --- HELPERS & LOGIC ---

    def _check_if_stuck(self) -> bool:
        """ Returns True if the robot has failed to move the required distance in the timeout window. """
        current_time = rospy.Time.now().to_sec()
        
        if current_time - self.last_check_time > self.stuck_timeout:
            dist_moved = math.hypot(self.current_x - self.last_check_x, self.current_y - self.last_check_y)
            
            # Reset trackers for the next window regardless of outcome
            self.last_check_time = current_time
            self.last_check_x = self.current_x
            self.last_check_y = self.current_y
            
            return dist_moved < self.stuck_dist
        return False

    def calculate_velocity_command(self, target_x: float, target_y: float) -> Tuple[Twist, bool]:
        cmd = Twist()
        dx, dy = target_x - self.current_x, target_y - self.current_y
        distance = math.hypot(dx, dy)
        
        if distance < self.goal_tol:
            return cmd, True  
            
        angle_error = math.atan2(dy, dx) - self.current_yaw
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))
        
        cmd.angular.z = self.p_gain_turn * angle_error
        
        if abs(angle_error) < 0.5:
            cmd.linear.x = min(self.p_gain_fwd * distance, self.max_v)
        else:
            cmd.linear.x = 0.0
            
        return cmd, False

    def a_star_search(self, start_world: Tuple[float, float], goal_world: Tuple[float, float]) -> List[Tuple[float, float]]:
        if not self.current_grid or not self.grid_info: return []

        start_grid = helpers.world_to_grid(start_world[0], start_world[1], self.grid_info)
        goal_grid = helpers.world_to_grid(goal_world[0], goal_world[1], self.grid_info)
        if not start_grid or not goal_grid: return []

        directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        frontier, came_from, g_score = [], {}, {start_grid: 0}
        heapq.heappush(frontier, (0, start_grid))

        while frontier:
            _, current_node = heapq.heappop(frontier)
            
            # Perform backtracking to find path if goal node explored
            if current_node == goal_grid:
                path = []
                while current_node in came_from:
                    coord = helpers.grid_to_world(current_node[0], current_node[1], self.grid_info)
                    if coord: path.append(coord)
                    current_node = came_from[current_node]
                path.reverse()
                return path

            # Check every neighboring node
            for dx, dy in directions:
                neighbor = (current_node[0] + dx, current_node[1] + dy)
                
                # out of bounds check
                if not (0 <= neighbor[0] < self.grid_info.width and 0 <= neighbor[1] < self.grid_info.height):
                    continue
                
                # occupied check
                if self.current_grid[(neighbor[1] * self.grid_info.width) + neighbor[0]] > 50: 
                    continue

                move_cost = 1.414 if dx != 0 and dy != 0 else 1.0

                tentative_g_score = g_score[current_node] + move_cost

                # add new unexplored neighbor if g cost is less
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current_node
                    g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + math.hypot(goal_grid[0] - neighbor[0], goal_grid[1] - neighbor[1])
                    heapq.heappush(frontier, (f_score, neighbor))

        return []

    # --- MAIN LOOP ---

    def navigate_loop(self) -> None:
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            if self.waypoints_queue:
                final_target_x, final_target_y = self.waypoints_queue[0]
                
                # 1. Stuck Detection
                if self._check_if_stuck():
                    rospy.logerr(f"STUCK! Dropping waypoint ({final_target_x}, {final_target_y})")
                    self.waypoints_queue.pop(0)
                    self.current_path.clear()
                    self.cmd_vel_pub.publish(Twist())
                    rate.sleep()
                    continue

                # 2. Pathfinding
                if not self.current_path:
                    self.current_path = self.a_star_search((self.current_x, self.current_y), (final_target_x, final_target_y))
                    
                    if not self.current_path:
                        rospy.logwarn("Target blocked! Canceling waypoint.")
                        self.waypoints_queue.pop(0) 
                        self.cmd_vel_pub.publish(Twist())
                        rate.sleep()
                        continue
                
                # 2.5 Path Validation
                # Check if newly discovered walls have blocked our current path
                if self.current_path and self.current_grid and self.grid_info:
                    path_blocked = False
                    # Check for obstacles ahead
                    for wp_x, wp_y in self.current_path[:5]: 
                        grid_coord = helpers.world_to_grid(wp_x, wp_y, self.grid_info)
                        if grid_coord:
                            idx = (grid_coord[1] * self.grid_info.width) + grid_coord[0]
                            # If next checkpoint breadcrumb is occupied, path is blocked
                            if self.current_grid[idx] > 50: 
                                path_blocked = True
                                break
                    
                    if path_blocked:
                        # Reset path
                        rospy.logwarn("New obstacle detected on path! Recalculating...")
                        self.current_path.clear()  # Wipes the bad path
                        self.cmd_vel_pub.publish(Twist()) # Hit the brakes
                        rate.sleep()
                        continue

                # 3. Movement Execution
                cmd, reached = self.calculate_velocity_command(self.current_path[0][0], self.current_path[0][1])
                
                if reached:
                    self.current_path.pop(0)
                    if not self.current_path:
                        rospy.loginfo(f"Final Waypoint reached! ({final_target_x}, {final_target_y})")
                        self.waypoints_queue.pop(0)
                
                self.cmd_vel_pub.publish(cmd)
                
            else:
                self.cmd_vel_pub.publish(Twist())
                self.last_check_time = rospy.Time.now().to_sec()
                self.last_check_x, self.last_check_y = self.current_x, self.current_y
                
            rate.sleep()

if __name__ == '__main__':
    try:
        WaypointNavigator().navigate_loop()
    except rospy.ROSInterruptException:
        pass