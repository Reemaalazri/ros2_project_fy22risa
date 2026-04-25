wait, lets check the logic again and be straightforward. first strt from the 2D pose estimate i have. then go towards first goal stop rotate all direction once then face the second goal repeat. thats all simple. i dont want it to lag or do something no expected import math
import time
import threading

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge, CvBridgeError
from nav2_msgs.action import NavigateToPose


class ColourNavigation(Node):
    def __init__(self):
        super().__init__("colour_navigation")

        self.bridge = CvBridge()

        # Camera subscriber
        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            10
        )

        # Laser subscriber for stopping distance
        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10
        )

        # Velocity publisher
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Nav2 action client
        self.action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # Colour flags
        self.red_seen = False
        self.green_seen = False
        self.blue_seen = False

        # Blue detection information
        self.blue_area = 0
        self.blue_centre_x = None
        self.image_width = 640

        # Front laser distance
        self.front_distance = None

        # Navigation state
        self.goal_finished = True
        self.goal_index = 0
        self.current_goal_handle = None

        # Task states
        self.final_finished = False
        self.blue_mode_started = False

        # 360 scan state
        self.scanning = False
        self.scan_start_time = None
        self.scan_duration = 16.0

        # Exploration waypoints only, not box coordinates
        self.waypoints = [
            (-0.26, 4.50),
            (7.00, 5.00),
            (7.00, -2.00),
            (-0.26, -2.40),
            (-1.80, 6.00),
            (-3.70, 3.70),
            (-10.00, 3.60),
            (-9.00, -14.00),
            (8.00, -12.00),
            (7.00, -4.00),
        ]

        self.get_logger().info("Colour navigation node started.")

    def get_yaw_to_next_waypoint(self, index):
        # Face roughly toward the next waypoint.
        current_x, current_y = self.waypoints[index]
        next_index = (index + 1) % len(self.waypoints)
        next_x, next_y = self.waypoints[next_index]

        return math.atan2(next_y - current_y, next_x - current_x)

    def scan_callback(self, msg):
        # Read only the front laser sector.
        ranges = np.array(msg.ranges)
        front_ranges = np.concatenate((ranges[0:15], ranges[-15:]))
        front_ranges = front_ranges[np.isfinite(front_ranges)]

        if len(front_ranges) > 0:
            self.front_distance = float(np.min(front_ranges))

    def image_callback(self, data):
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError:
            return

        self.image_width = frame.shape[1]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # HSV colour ranges
        red_lower_1 = np.array([0, 100, 100])
        red_upper_1 = np.array([10, 255, 255])
        red_lower_2 = np.array([170, 100, 100])
        red_upper_2 = np.array([180, 255, 255])

        green_lower = np.array([45, 80, 80])
        green_upper = np.array([75, 255, 255])

        blue_lower = np.array([100, 100, 80])
        blue_upper = np.array([130, 255, 255])

        # Colour masks
        red_mask_1 = cv2.inRange(hsv, red_lower_1, red_upper_1)
        red_mask_2 = cv2.inRange(hsv, red_lower_2, red_upper_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # Detect and highlight all RGB boxes
        self.process_colour(frame, red_mask, "red", (0, 0, 255))
        self.process_colour(frame, green_mask, "green", (0, 255, 0))
        self.process_colour(frame, blue_mask, "blue", (255, 0, 0))

        # Filtered camera view
        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, blue_mask)
        filtered = cv2.bitwise_and(frame, frame, mask=combined_mask)

        cv2.namedWindow("RGB detection", cv2.WINDOW_NORMAL)
        cv2.imshow("RGB detection", frame)
        cv2.resizeWindow("RGB detection", 640, 480)

        cv2.namedWindow("Filtered RGB view", cv2.WINDOW_NORMAL)
        cv2.imshow("Filtered RGB view", filtered)
        cv2.resizeWindow("Filtered RGB view", 640, 480)

        cv2.waitKey(3)

    def process_colour(self, frame, mask, colour_name, box_colour):
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            if colour_name == "blue":
                self.blue_seen = False
                self.blue_area = 0
                self.blue_centre_x = None
            return

        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Ignore small noise
        if area < 400:
            if colour_name == "blue":
                self.blue_seen = False
                self.blue_area = 0
                self.blue_centre_x = None
            return

        x, y, w, h = cv2.boundingRect(largest)
        cx = x + w // 2
        cy = y + h // 2

        cv2.rectangle(frame, (x, y), (x + w, y + h), box_colour, 2)
        cv2.circle(frame, (cx, cy), 5, box_colour, -1)
        cv2.putText(
            frame,
            colour_name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            box_colour,
            2
        )

        if colour_name == "red":
            self.red_seen = True
        elif colour_name == "green":
            self.green_seen = True
        elif colour_name == "blue":
            self.blue_seen = True
            self.blue_area = area
            self.blue_centre_x = cx

    def send_goal(self, x, y, yaw):
        if self.final_finished or self.blue_mode_started:
            return

        self.goal_finished = False
        self.scanning = False

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)

        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f"Sending exploration goal: x={x}, y={y}, yaw={yaw}")

        self.action_client.wait_for_server()
        future = self.action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()

        if not goal_handle.accepted:
            self.get_logger().info("Goal rejected.")
            self.goal_finished = True
            return

        self.current_goal_handle = goal_handle
        self.get_logger().info("Goal accepted.")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.goal_result_callback)

    def goal_result_callback(self, future):
        # When a waypoint is reached, rotate 360 before sending the next goal.
        if self.blue_mode_started or self.final_finished:
            return

        self.get_logger().info("Goal reached. Starting 360 scan.")
        self.goal_finished = False
        self.scanning = True
        self.scan_start_time = time.time()
        self.current_goal_handle = None

    def cancel_current_goal(self):
        # Cancel Nav2 goal immediately when blue is detected.
        if self.current_goal_handle is not None:
            self.get_logger().info("Cancelling Nav2 goal because blue was detected.")
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None

        self.goal_finished = False
        self.scanning = False
        self.scan_start_time = None
        self.stop_robot()

    def rotate_360_scan(self):
        # Rotate once after reaching a waypoint.
        if self.final_finished or self.blue_mode_started:
            self.stop_robot()
            return

        if self.scan_start_time is None:
            self.scan_start_time = time.time()

        elapsed = time.time() - self.scan_start_time
        twist = Twist()

        if elapsed < self.scan_duration:
            twist.angular.z = 0.4
            self.cmd_pub.publish(twist)
        else:
            self.stop_robot()
            self.scanning = False
            self.scan_start_time = None
            self.goal_finished = True
            self.get_logger().info("360 scan finished. Moving to next exploration goal.")

    def cancel_navigation_and_follow_blue(self):
        # Once blue is detected, stop waypoint navigation and follow blue.
        if self.final_finished:
            self.stop_robot()
            return

        if not self.blue_mode_started:
            self.blue_mode_started = True
            self.cancel_current_goal()
            self.get_logger().info("Blue detected. Switching to camera-based approach.")

        twist = Twist()

        # If blue is briefly lost, keep moving and turning gently.
        if self.blue_centre_x is None:
            twist.linear.x = 0.05
            twist.angular.z = 0.18
            self.cmd_pub.publish(twist)
            return

        image_centre = self.image_width / 2
        error = self.blue_centre_x - image_centre

        # Laser measures distance to box face.
        # 0.70m from face is roughly around 1.2m from the box centre.
        stop_distance = 0.70

        if self.front_distance is not None and self.front_distance <= stop_distance:
            self.get_logger().info("Blue box reached. Stopping about 1.2m from its centre.")
            self.stop_robot()
            self.final_finished = True
            return

        # Slightly faster blue approach.
        if abs(error) > 120:
            twist.angular.z = -0.0015 * error
            twist.linear.x = 0.06
        elif abs(error) > 50:
            twist.angular.z = -0.0012 * error
            twist.linear.x = 0.08
        else:
            twist.linear.x = 0.10
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

    def explore(self):
        # Do not send a waypoint if scanning, following blue, or finished.
        if self.final_finished or self.blue_mode_started or self.scanning:
            return

        # Send first waypoint immediately, then wait for scan before next one.
        if self.goal_finished:
            x, y = self.waypoints[self.goal_index]
            yaw = self.get_yaw_to_next_waypoint(self.goal_index)

            self.send_goal(x, y, yaw)
            self.goal_index = (self.goal_index + 1) % len(self.waypoints)

    def stop_robot(self):
        twist = Twist()
        self.cmd_pub.publish(twist)


def spin_node(node):
    rclpy.spin(node)


def main(args=None):
    rclpy.init(args=args)

    robot = ColourNavigation()

    thread = threading.Thread(target=spin_node, args=(robot,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():

            if robot.final_finished:
                robot.stop_robot()

            elif robot.blue_seen:
                robot.cancel_navigation_and_follow_blue()

            elif robot.scanning:
                robot.rotate_360_scan()

            else:
                robot.explore()

            time.sleep(0.2)

    except KeyboardInterrupt:
        pass

    robot.stop_robot()
    cv2.destroyAllWindows()
    robot.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()