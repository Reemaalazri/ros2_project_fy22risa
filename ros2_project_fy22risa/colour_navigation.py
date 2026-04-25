import math
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

        # Used to convert ROS Image messages into OpenCV images.
        self.bridge = CvBridge()

        # Subscribe to the robot camera feed.
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, 10
        )

        # Subscribe to the laser scanner so the robot can stop near the blue box.
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )

        # Publisher for direct velocity commands.
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Nav2 action client used to send autonomous navigation goals.
        self.action_client = ActionClient(self, NavigateToPose, "navigate_to_pose")

        # Flags showing whether each colour has been detected.
        self.red_seen = False
        self.green_seen = False
        self.blue_seen = False

        # Information about the detected blue box.
        self.blue_area = 0
        self.blue_centre_x = None
        self.image_width = 640

        # Closest laser reading in front of the robot.
        self.front_distance = None

        # Waypoint navigation state.
        self.goal_finished = True
        self.goal_index = 0
        self.current_goal_handle = None

        # Final task states.
        self.final_finished = False
        self.blue_mode_started = False

        # 360-degree scan state.
        self.scanning = False
        self.scan_start_time = None
        self.scan_duration = 11.0
        self.scan_speed = 0.6

        # Exploration waypoints. These are free-space points, not box coordinates.
        self.waypoints = [
            (-1.01, 4.32),
            (6.82, 5.03),
            (7.45, -1.72),
            (-0.86, -3.20),

            (-2.66, 3.85),
            (-10.26, 3.50),
            (-10.00, 3.60),
            (-9.00, -14.00),

            (8.00, -12.00),
            (7.00, -4.00),
        ]

        self.get_logger().info("Colour navigation node started.")

    def get_yaw_to_next_waypoint(self, index):
        # Calculate the direction the robot should face at this waypoint.
        x1, y1 = self.waypoints[index]
        x2, y2 = self.waypoints[(index + 1) % len(self.waypoints)]
        return math.atan2(y2 - y1, x2 - x1)

    def scan_callback(self, msg):
        # Use only the front laser sector for stopping near the blue box.
        ranges = np.array(msg.ranges)
        front_ranges = np.concatenate((ranges[0:15], ranges[-15:]))
        front_ranges = front_ranges[np.isfinite(front_ranges)]

        if len(front_ranges) > 0:
            self.front_distance = float(np.min(front_ranges))

    def image_callback(self, data):
        # Convert the ROS camera image into OpenCV BGR format.
        try:
            frame = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError:
            return

        self.image_width = frame.shape[1]

        # Convert BGR to HSV because HSV is more suitable for colour filtering.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red has two HSV ranges because red wraps around the hue scale.
        red_mask_1 = cv2.inRange(
            hsv,
            np.array([0, 100, 100]),
            np.array([10, 255, 255])
        )
        red_mask_2 = cv2.inRange(
            hsv,
            np.array([170, 100, 100]),
            np.array([180, 255, 255])
        )
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)

        # Green mask.
        green_mask = cv2.inRange(
            hsv,
            np.array([45, 80, 80]),
            np.array([75, 255, 255])
        )

        # Blue mask.
        blue_mask = cv2.inRange(
            hsv,
            np.array([100, 100, 80]),
            np.array([130, 255, 255])
        )

        # Detect each colour and draw its bounding box on the original camera feed.
        self.process_colour(frame, red_mask, "red", (0, 0, 255))
        self.process_colour(frame, green_mask, "green", (0, 255, 0))
        self.process_colour(frame, blue_mask, "blue", (255, 0, 0))

        # Combined filtered view for the assessment video.
        combined_mask = cv2.bitwise_or(red_mask, green_mask)
        combined_mask = cv2.bitwise_or(combined_mask, blue_mask)
        filtered = cv2.bitwise_and(frame, frame, mask=combined_mask)

        # Show original image with bounding boxes.
        cv2.namedWindow("RGB detection", cv2.WINDOW_NORMAL)
        cv2.imshow("RGB detection", frame)
        cv2.resizeWindow("RGB detection", 640, 480)

        # Show filtered RGB view.
        cv2.namedWindow("Filtered RGB view", cv2.WINDOW_NORMAL)
        cv2.imshow("Filtered RGB view", filtered)
        cv2.resizeWindow("Filtered RGB view", 640, 480)

        cv2.waitKey(3)

    def process_colour(self, frame, mask, colour_name, box_colour):
        # Find coloured regions in the mask.
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # If blue disappears, reset blue tracking.
        if len(contours) == 0:
            if colour_name == "blue":
                self.blue_seen = False
                self.blue_area = 0
                self.blue_centre_x = None
            return

        # Use the largest contour as the main detected object.
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)

        # Ignore small noisy detections.
        if area < 400:
            if colour_name == "blue":
                self.blue_seen = False
                self.blue_area = 0
                self.blue_centre_x = None
            return

        # Draw bounding box and centre point.
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

        # Update colour state.
        if colour_name == "red":
            self.red_seen = True
        elif colour_name == "green":
            self.green_seen = True
        elif colour_name == "blue":
            self.blue_seen = True
            self.blue_area = area
            self.blue_centre_x = cx

    def send_goal(self, x, y, yaw):
        # Do not send new Nav2 goals after blue is found or task is finished.
        if self.final_finished or self.blue_mode_started:
            return

        self.goal_finished = False
        self.scanning = False

        # Create a Nav2 NavigateToPose goal.
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = "map"
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()

        goal_msg.pose.pose.position.x = float(x)
        goal_msg.pose.pose.position.y = float(y)

        # Convert yaw angle to quaternion.
        goal_msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        goal_msg.pose.pose.orientation.w = math.cos(yaw / 2.0)

        self.get_logger().info(f"Sending exploration goal: x={x}, y={y}, yaw={yaw}")

        # Send goal asynchronously.
        self.action_client.wait_for_server()
        future = self.action_client.send_goal_async(goal_msg)
        future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        # Called when Nav2 accepts or rejects the goal.
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
        # Once a waypoint is reached, start a 360-degree scan.
        if self.blue_mode_started or self.final_finished:
            return

        self.get_logger().info("Goal reached. Starting 360 scan.")
        self.goal_finished = False
        self.scanning = True
        self.scan_start_time = time.time()
        self.current_goal_handle = None

    def cancel_current_goal(self):
        # Cancel the current Nav2 goal when blue is detected.
        if self.current_goal_handle is not None:
            self.get_logger().info("Cancelling Nav2 goal because blue was detected.")
            self.current_goal_handle.cancel_goal_async()
            self.current_goal_handle = None

        self.goal_finished = False
        self.scanning = False
        self.scan_start_time = None
        self.stop_robot()

    def rotate_360_scan(self):
        # Rotate in place after each waypoint to check all directions.
        if self.final_finished or self.blue_mode_started:
            self.stop_robot()
            return

        if self.scan_start_time is None:
            self.scan_start_time = time.time()

        elapsed = time.time() - self.scan_start_time
        twist = Twist()

        if elapsed < self.scan_duration:
            twist.angular.z = self.scan_speed
            self.cmd_pub.publish(twist)
        else:
            self.stop_robot()
            self.scanning = False
            self.scan_start_time = None
            self.goal_finished = True
            self.get_logger().info("360 scan finished. Moving to next exploration goal.")

    def cancel_navigation_and_follow_blue(self):
        # Once blue is detected, stop exploration and follow the blue box.
        if self.final_finished:
            self.stop_robot()
            return

        if not self.blue_mode_started:
            self.blue_mode_started = True
            self.cancel_current_goal()
            self.get_logger().info("Blue detected. Switching to camera-based approach.")

        twist = Twist()

        # If blue is briefly lost, move slowly and turn to recover it.
        if self.blue_centre_x is None:
            twist.linear.x = 0.05
            twist.angular.z = 0.18
            self.cmd_pub.publish(twist)
            return

        # Steer based on how far the blue box is from the image centre.
        image_centre = self.image_width / 2
        error = self.blue_centre_x - image_centre

        # Laser measures distance to the front face of the box.
        # Since the box is 1m deep, 0.55m from the face is roughly within 1m of the centre.
        stop_distance = 0.55

        if self.front_distance is not None and self.front_distance <= stop_distance:
            self.get_logger().info("Blue box reached. Stopping within about 1m of its centre.")
            self.stop_robot()
            self.final_finished = True
            return

        # Camera-based steering towards the blue box.
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
        # Do not explore if scanning, following blue or finished.
        if self.final_finished or self.blue_mode_started or self.scanning:
            return

        # Send the next waypoint only when the previous waypoint and scan are complete.
        if self.goal_finished:
            x, y = self.waypoints[self.goal_index]
            yaw = self.get_yaw_to_next_waypoint(self.goal_index)

            self.send_goal(x, y, yaw)
            self.goal_index = (self.goal_index + 1) % len(self.waypoints)

    def stop_robot(self):
        # Send zero velocity to stop the robot.
        self.cmd_pub.publish(Twist())


def spin_node(node):
    # Keep ROS callbacks running.
    rclpy.spin(node)


def main(args=None):
    rclpy.init(args=args)

    robot = ColourNavigation()

    # Run ROS callbacks in a separate thread while the main loop controls behaviour.
    thread = threading.Thread(target=spin_node, args=(robot,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            # Priority 1: stay stopped after finishing.
            if robot.final_finished:
                robot.stop_robot()

            # Priority 2: if blue is seen, cancel navigation and follow it.
            elif robot.blue_seen:
                robot.cancel_navigation_and_follow_blue()

            # Priority 3: after each waypoint, rotate 360 degrees.
            elif robot.scanning:
                robot.rotate_360_scan()

            # Priority 4: otherwise keep exploring.
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