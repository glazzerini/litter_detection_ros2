import rclpy
from rclpy.node import Node
from custom_msgs.msg import DetectionResults
import numpy as np
import cv2


class ImageProcessorNode(Node):
    def __init__(self):
        super().__init__('image_processor_node')

        # Subscribers
        self.detection_data_sub = self.create_subscription(DetectionResults, '/detection_data', self.listener_callback, 1)

        # Counter to save images with unique names
        self.image_counter = 0

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
            cv2.rectangle(cv_image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

            # Draw the label and confidence
            label_text = f'{label}: {confidence:.2f}'
            cv2.putText(cv_image, label_text, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Use the counter to create a unique filename
        filename = f'/home/guidolazzerini/ros_data/litter_detection/frames/bts/prova_{self.image_counter}.png'
        self.image_counter += 1  # Increment the counter for the next image

        # Save the image with the unique filename
        # cv2.imwrite(filename, cv_image)

        cv2.imshow('Image', cv_image)
        cv2.waitKey(1)


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
