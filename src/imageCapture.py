import rclpy
import cv2
import os
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rclpy.node import Node

class ImageSubscriber(Node):
    def __init__(self):
        super().__init__('image_subscriber')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/D455/color/image_raw',
            self.listener_callback,
            10)
        self.img_counter = 0
        # self.save_dir = "src/yoloV8/cameraSettings/checkerboardImages"
        self.save_dir = "src/yoloV8/objDetectionImagesBatch2"

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.latest_image = None

    def listener_callback(self, msg):
        # Convert the ROS Image message to a NumPy array
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def display_stream(self):
        while rclpy.ok():
            if self.latest_image is not None:
                # Display the image
                cv2.imshow('Image Window', self.latest_image)
                key = cv2.waitKey(1)

                # If the user presses the space bar, save the current frame
                if key == ord(' '):
                    img_name = os.path.join(self.save_dir, "image_{}.jpg".format(self.img_counter))
                    cv2.imwrite(img_name, self.latest_image)
                    print("{} written!".format(img_name))
                    self.img_counter += 1

                # If the user presses 'q', exit the loop
                elif key == ord('q'):
                    cv2.destroyAllWindows()
                    break

def main(args=None):
    rclpy.init(args=args)

    image_subscriber = ImageSubscriber()

    # Start a separate thread to display the video stream
    import threading
    display_thread = threading.Thread(target=image_subscriber.display_stream)
    display_thread.start()

    rclpy.spin(image_subscriber)

    # Destroy the node explicitly
    image_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()