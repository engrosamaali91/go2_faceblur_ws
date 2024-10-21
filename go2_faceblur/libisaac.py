#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO


class IsaacController(Node):

    def __init__(self):
        super().__init__('quadruped_controller')
        self.get_logger().info(
            f'{self.colorize("Activating FaceBlur Node!","yellow")}')

        # Parameters Declaration
        self.declare_parameters(namespace='', parameters=[
            ('robot_topic', 'quadruped'),
        ])
        self.param_robot_topic = self.get_parameter('robot_topic').value
        self.get_logger().info(
            f'{self.colorize(f"robot_topic: {self.param_robot_topic}","blue")}')

        # for converting between ros and opencv images
        self.bridge = CvBridge()

        # blur ration
        self.blur_ratio = 50

        # subscriber to camera image topic
        self.subscription = self.create_subscription(
            Image,
            '/go2/sensor/camera_raw',
            self.image_callback,
            5
        )

        # Initialize YOLO model
        self.model = YOLO(
            "/home/unitree/osama_go2_ws/src/go2_faceblur/assets/best.pt")
        self.names = self.model.names
        self.cv_image = None

        # ##################################################################

        # ROS Publishers
        # Create a publisher for the processed image
        self.publisher = self.create_publisher(
            Image, '/go2/camera/processed_image', 10)

        # publish at 50 hz
        self.timer = self.create_timer(1.0 / 50, self.timer_callback)

    def timer_callback(self):
        if self.cv_image is not None:
            self.get_logger().info(
                f'{self.colorize("Processing image...","yellow")}')
            # Run YOLO detection
            results = self.model.predict(self.cv_image)
            # Coordinates of detected objects
            boxes = results[0].boxes.xyxy.cpu().tolist()
            # Class IDs of detected objects
            clss = results[0].boxes.cls.cpu().tolist()
            if boxes is not None:
                for box, cls in zip(boxes, clss):
                    # Extract the detected object region
                    obj = self.cv_image[int(box[1]): int(
                        box[3]), int(box[0]): int(box[2])]
                    # Apply blur to the detected region
                    blur_obj = cv2.blur(
                        obj, (self.blur_ratio, self.blur_ratio))

                    # Replace the detected region with the blurred version
                    self.cv_image[int(box[1]): int(box[3]), int(
                        box[0]): int(box[2])] = blur_obj

            # Convert OpenCV image back to ROS Image message
            processed_image_msg = self.bridge.cv2_to_imgmsg(
                self.cv_image, "bgr8")
            # Publish the processed image
            self.publisher.publish(processed_image_msg)

    def image_callback(self, msg):
        self.get_logger().info(
            f'{self.colorize("Received image from camera","green")}', once=True)
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def colorize(self, text, color):
        color_codes = {
            'green': '\033[92m',
            'yellow': '\033[93m',
            'orange': '\033[38;5;208m',
            'blue': '\033[94m',
            'red': '\033[91m'
        }
        return color_codes[color] + text + '\033[0m'

    def destroy_node(self):
        self.get_logger().info(
            f'{self.colorize("Shutting down FaceBlur node","red")}')
        super().destroy_node()
