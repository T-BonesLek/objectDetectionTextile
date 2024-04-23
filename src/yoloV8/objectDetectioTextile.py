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
from std_msgs.msg import Float32MultiArray


#Realsense camera calibration $realsense-viewer
# $ros2 launch realsense2_camera rs_launch.py camera_namespace:=textilePose camera_name:=D455

# model
model = YOLO("yolo-Weights/textiles.pt")

classNames = ["textile_Dark", "textile_Light", "textile_Multi",]


class ObjectPublisher(Node):
    def __init__(self):
        super().__init__('object_publisher')
        self.publisher_ = self.create_publisher(Float32MultiArray, 'object_pose_topic', 0)

    def publish_message(self, array):
        msg = Float32MultiArray()
        msg.data = array
        self.publisher_.publish(msg)
        self.get_logger().info('Publishing: "%s"' % (msg.data))


    
        
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
        self.published_ids = []

    def load_roi(self):
            # Check if ROI file exists
            if os.path.exists('./cameraSettings/roi.pkl'):
                # Load the ROI if it exists
                with open('./cameraSettings/roi.pkl', 'rb') as f:
                    self.roi = pickle.load(f)
                    print("ROI loaded from file:", self.roi)
            else:
                # If no ROI file is found, set to None
                self.roi = None
        

    def save_roi(self):
        # Save the ROI coordinates to a file
        with open('./cameraSettings/roi.pkl', 'wb') as f:
            pickle.dump(self.roi, f)
            
    def listener_callback(self, msg):
        cls = 999.99 # Default class value aka No class

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

            # Detect objects in the frame
            results = list(model.track(img, stream=True, persist=True))
            
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # coordinates
            for r in results:
                boxes = r.boxes

                if not boxes:  # If there are no boxes
                    node = ObjectPublisher()
                    x = [float (999.99), float (999.99), float (999.99)]  # Replace with the message you want to send when no boxes are detected
                    node.publish_message(x)

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
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

                        # Draw center point
                        cv2.circle(annotated_frame, (center_x, center_y), 5, (255, 255, 0), -1)

                        # camera matrix (mtx) from calibration
                        mtx = np.load('./cameraSettings/cam_mtx.npy')  # Load camera matrix from file

                        # Camera height above the conveyor belt in mm
                        Z = 390  # replace with actual height !!!!!!! ALWAYS CHCK THE objectdetection coordinates and adjust the height, it seams to be the camera that is 10mm high

                        # Convert pixel coordinates to world coordinates                    
                        X1 = x1 * Z / mtx[0, 0]
                        Y1 = y1 * Z / mtx[1, 1]
                        X2 = x2 * Z / mtx[0, 0]
                        Y2 = y2 * Z / mtx[1, 1]

                        # Calculate world center coordinates
                        center_X = round((X1 + X2) / 2)/100  # Convert to meters
                        center_Y = round((Y1 + Y2) / 2)/100  # Convert to meters

                        # class name
                        cls = int(box.cls[0])
                        # print("Class name -->", classNames[cls])

                        cv2.putText(annotated_frame, f'Center: ({center_X}, {center_Y})', (center_x - 40, center_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        # Publish the center_X coordinates to a ROS topic
                        node = ObjectPublisher()

                        # # Check if the object's center is within the scan ROI
                        # if self.scan_roi is not None and scan_x <= center_x <= scan_x + scan_w and scan_y <= center_y <= scan_y + scan_h:
                        # Check if the object's ID has been published before
                        if box.id in self.published_ids:
                            continue  # Skip this object if its ID has been published before
                    
                        elif box.id not in self.published_ids:

                            robot_pickup_value_x = center_X
                            robot_pickup_value_y = center_Y
                            textile_class = cls

                            # Add the object's ID to the list of published IDs
                            self.published_ids.append(box.id)

                            ## Store object ID and center coordinates in the global array
                            # object_array.append((box.id, (robot_pickup_value_x, robot_pickup_value_y)))

                            if box.id == None:
                                id = 0.0
                            else:
                                id = box.id

                            x = [round(float(robot_pickup_value_x), 2), round(float(robot_pickup_value_y), 2), float(textile_class), float(id) ]# round to one decimal place
        
                            node.publish_message(x)
                            # print("object information:", object_array) # object information: [(tensor([1.]), (314, 356)), (tensor([2.]), (337, 572)), (tensor([3.]), (362, 104))]


                cv2.imshow('ROI with YOLO', annotated_frame)
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