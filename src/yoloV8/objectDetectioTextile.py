import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import os
import pickle
import math 
import numpy as np
from geometry_msgs.msg import Point  # Import the Point message type


#Realsense camera calibration $realsense-viewer
# $ros2 launch realsense2_camera rs_launch.py camera_namespace:=textilePose camera_name:=D455

# model
model = YOLO("yolo-Weights/yolov8n.pt")

# object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

class ObjectPublisher(Node):
    def __init__(self):
        super().__init__('object_publisher')
        self.publisher_ = self.create_publisher(Point, 'object_pose_topic', 10)

    def publish_message(self, x):
        msg = Point()
        msg.x = x
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "x: %s"' % (msg.x))

        
class ImageSubscriber(Node):

    def __init__(self):
        super().__init__('image_subscriber')
        self.subscription = self.create_subscription(
            Image,
            '/D455/color/image_raw',
            self.listener_callback,
            10)
        self.subscription
        self.bridge = CvBridge()
        self.roi = None  # Make sure this line is part of the __init__ method
        self.window_opened = False
        self.load_roi()  # Load ROI from file when initializing

    def load_roi(self):
            # Check if ROI file exists
            if os.path.exists('cameraSettings/roi.pkl'):
                # Load the ROI if it exists
                with open('cameraSettings/roi.pkl', 'rb') as f:
                    self.roi = pickle.load(f)
                    print("ROI loaded from file:", self.roi)
            else:
                # If no ROI file is found, set to None
                self.roi = None
        

    def save_roi(self):
        # Save the ROI coordinates to a file
        with open('cameraSettings/roi.pkl', 'wb') as f:
            pickle.dump(self.roi, f)

    
    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if self.roi is None:
            # Display instructions
            cv2.putText(cv_image, "Select ROI and press SPACE to confirm or Q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.roi = cv2.selectROI(cv_image)
            print("ROI selected:", self.roi)
            self.window_opened = True
            cv2.destroyAllWindows()

        if self.window_opened or self.roi is not None:
            
            cv2.rectangle(cv_image, (int(self.roi[0]), int(self.roi[1])),
                (int(self.roi[0] + self.roi[2]), int(self.roi[1] + self.roi[3])), (255, 0, 255), 3)

            roi_img = cv_image[int(self.roi[1]):int(self.roi[1] + self.roi[3]), int(self.roi[0]):int(self.roi[0] + self.roi[2])]

            img = roi_img
            results = model(img, stream=True)

            # coordinates
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    confidence_threshold = 0.6  # Default confidence threshold
                    if box.conf >= confidence_threshold:

                        # bounding box
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                        # Calculate center coordinates
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        # put box in cam
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)

                        # Draw center point
                        cv2.circle(img, (center_x, center_y), 5, (255, 255, 0), -1)

                        # camera matrix (mtx) from calibration
                        mtx = np.load('cameraSettings/cam_mtx.npy')  # Load camera matrix from file

                        # Camera height above the conveyor belt in mm
                        Z = 405  # replace with actual height

                        # Convert pixel coordinates to world coordinates                    
                        X1 = x1 * Z / mtx[0, 0]
                        Y1 = y1 * Z / mtx[1, 1]
                        X2 = x2 * Z / mtx[0, 0]
                        Y2 = y2 * Z / mtx[1, 1]

                        # Calculate world center coordinates
                        center_X = round((X1 + X2) / 2)
                        center_Y = round((Y1 + Y2) / 2)
                
                        # confidence
                        # print("Confidence --->",confidence)

                        # class name
                        cls = int(box.cls[0])
                        # print("Class name -->", classNames[cls])

                        # object details
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 255, 255)
                        thickness = 1

                        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                        cv2.putText(img, f'Center: ({center_X}, {center_Y})', (center_x - 40, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Publish the center_X coordinates to a ROS topic
                        node = ObjectPublisher()
                        x = float(center_X)
                        node.publish_message(x)


                cv2.imshow('ROI with YOLO', img)
                self.save_roi() # Save the ROI to a file

                


                key = cv2.waitKey(1)
                if key == ord('q') or key == ord('Q'):
                    self.destroy_node()
                    cv2.destroyAllWindows()

    def destroy_node(self):
        super().destroy_node()
        os._exit(0) # Exit the program


def main(args=None):
    rclpy.init(args=args)
    node = ObjectPublisher()

    image_subscriber = ImageSubscriber()

    rclpy.spin(image_subscriber)
 
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()