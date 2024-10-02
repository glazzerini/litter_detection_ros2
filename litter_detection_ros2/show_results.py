import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from custom_msgs.msg import DetectionResults
from cv_bridge import CvBridge
import numpy as np
import cv2

class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')

        # Publishers
        self.detection_pub = self.create_publisher(CompressedImage, '/detection', 1)

        self.get_logger().error("here")

        # Subscribers
        self.detection_data_sub = self.create_subscription(DetectionResults, '/detection_data', self.listener_callback, 1)

        # CvBridge to convert ROS Image messages to OpenCV format
        self.bridge = CvBridge()

    def listener_callback(self, msg: DetectionResults):

        compressed_image = np.array(msg.image, dtype=np.uint8)

        # Decompress the image using OpenCV
        cv_image = cv2.imdecode(compressed_image, cv2.IMREAD_COLOR)


        # Check if image was successfully decompressed
        if cv_image is None:
            self.get_logger().error("Failed to decompress image")
            return

        # Loop over all detections in the message
        num_detections = len(msg.box_labels)
        for i in range(num_detections):
            # Get the label, confidence, and bounding box
            label = msg.box_labels[i]
            confidence = msg.box_confidences[i]
            xmin = int(msg.box_coordinates[i * 4 + 0])
            ymin = int(msg.box_coordinates[i * 4 + 1])
            xmax = int(msg.box_coordinates[i * 4 + 2])
            ymax = int(msg.box_coordinates[i * 4 + 3])

            # Draw the bounding box
            cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw the label and confidence
            label_text = f'{label}: {confidence:.2f}'
            cv2.putText(cv_image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            self.detection_pub.publish(self.bridge.cv2_to_compressed_imgmsg(cv_image, "jpg"))



def main(args=None):
    rclpy.init(args=args)
    node = ImageProcessorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
