import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import cv2
import os
import pickle
import numpy as np
from std_msgs.msg import Float32MultiArray

# Initialize YOLO model
model = YOLO("yolo-Weights/textiles.pt")

classNames = ["textile_Dark", "textile_Light", "textile_Multi"]

# Create a ROS node for publishing object information
class ObjectPublisher(Node):
    def __init__(self):
        super().__init__('object_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'object_pose_topic', 0)

    def publish_message(self, array):
        msg = Float32MultiArray()
        msg.data = array
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % (msg.data))

# Create a ROS node for subscribing to images
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
        self.roi = None  # Initialize ROI
        self.scanning_area = None  # Initialize scanning area
        self.window_opened = False
        self.load_roi()  # Load ROI from file when initializing
        self.published_ids = []
        self.object_published = False  # Track if an object has been published

    def load_roi(self):
        # Load ROI from file
        if os.path.exists('./cameraSettings/roi.pkl'):
            with open('./cameraSettings/roi.pkl', 'rb') as f:
                self.roi = pickle.load(f)
                print("ROI loaded from file:", self.roi)
        else:
            self.roi = None

    def save_roi(self):
        # Save ROI coordinates to file
        with open('./cameraSettings/roi.pkl', 'wb') as f:
            pickle.dump(self.roi, f)

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        if self.roi is None:
            # Display instructions to select ROI
            cv2.putText(cv_image, "Select ROI and press SPACE to confirm or Q to quit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.roi = cv2.selectROI(cv_image)
            print("ROI selected:", self.roi)
            self.window_opened = True
            cv2.destroyAllWindows()

        if self.window_opened or self.roi is not None:
            cv2.rectangle(cv_image, (int(self.roi[0]), int(self.roi[1])),
                          (int(self.roi[0] + self.roi[2]), int(self.roi[1] + self.roi[3])), (255, 0, 255), 3)

            roi_img = cv_image[int(self.roi[1]):int(self.roi[1] + self.roi[3]),
                      int(self.roi[0]):int(self.roi[0] + self.roi[2])]

            img = roi_img

            results = list(model.track(img, stream=True, persist=True))

            annotated_frame = results[0].plot()

            # Draw the scanning area on the frame
            if self.scanning_area is not None:
                cv2.rectangle(annotated_frame, (self.scanning_area[0], self.scanning_area[1]),
                              (self.scanning_area[0] + self.scanning_area[2], self.scanning_area[1] + self.scanning_area[3]), (0, 255, 255), 2)

            for r in results:
                boxes = r.boxes

                if not boxes:
                    node = ObjectPublisher()
                    x = [999.99, 999.99, 999.99, 999.99]  # No boxes detected
                    node.publish_message(x)

                for box in boxes:
                    confidence_threshold = 0.6
                    if box.conf >= confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 1)
                        cv2.circle(annotated_frame, (center_x, center_y), 5, (255, 255, 0), -1)

                        if self.is_inside_scanning_area(center_x, center_y):
                            cls = int(box.cls[0])
                            robot_pickup_value_x, robot_pickup_value_y = self.pixel_to_world_coordinates(x1, y1, x2, y2)
                            id = box.id if box.id is not None else 0.0

                            # Check if object has been published
                            if not self.object_published:
                                x = [round(float(robot_pickup_value_x), 2), round(float(robot_pickup_value_y), 2),
                                     float(cls), float(id)]
                                node = ObjectPublisher()
                                node.publish_message(x)
                                self.published_ids.append(box.id)
                                self.object_published = True  # Set object_published to True

            cv2.imshow('ROI with YOLO', annotated_frame)
            self.save_roi()

            key = cv2.waitKey(1)
            if key == ord('q') or key == ord('Q'):
                self.destroy_node()
                cv2.destroyAllWindows()

    def is_inside_scanning_area(self, center_x, center_y):
        # Check if the object's center falls within the scanning area
        if self.scanning_area is not None:
            scan_area_x, scan_area_y, scan_area_w, scan_area_h = self.scanning_area
            return scan_area_x <= center_x <= scan_area_x + scan_area_w and \
                   scan_area_y <= center_y <= scan_area_y + scan_area_h
        else:
            return False

    def pixel_to_world_coordinates(self, x1, y1, x2, y2):
        mtx = np.load('./cameraSettings/cam_mtx.npy')
        Z = 390  # Camera height above the conveyor belt in mm
        X1 = x1 * Z / mtx[0, 0]
        Y1 = y1 * Z / mtx[1, 1]
        X2 = x2 * Z / mtx[0, 0]
        Y2 = y2 * Z / mtx[1, 1]
        center_X = round((X1 + X2) / 2) / 100  # Convert to meters
        center_Y = round((Y1 + Y2) / 2) / 100  # Convert to meters
        return center_X, center_Y

    def set_scanning_area(self, x, y, w, h):
        # Set the scanning area coordinates
        self.scanning_area = (x, y, w, h)

    def destroy_node(self):
        super().destroy_node()
        os._exit(0)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectPublisher()
    image_subscriber = ImageSubscriber()
    # Set the scanning area (adjust these values as needed)
    image_subscriber.set_scanning_area(
        int(image_subscriber.roi[0]  * 0.0),
        int(image_subscriber.roi[1]  * 2.0),
        int(image_subscriber.roi[2] * 1.0),
        int(image_subscriber.roi[3] * 0.7)
    )
    rclpy.spin(image_subscriber)
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
