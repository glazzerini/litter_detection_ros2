#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
from ultralytics import YOLO
from cv_bridge import CvBridge
from custom_msgs.msg import DetectionResults
from std_msgs.msg import Header


class LitterDetector(Node):
    def __init__(self):
        super().__init__('litter_detector')

        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter('detection.model_type', 'PT')
        self.declare_parameter('detection.confidence', 0.5)
        self.declare_parameter('detection.device', 'cpu')
        self.declare_parameter('pkg_path', '')

        self.model_type = self.get_parameter('detection.model_type').get_parameter_value().string_value
        self.confidence = self.get_parameter('detection.confidence').get_parameter_value().double_value
        self.device = self.get_parameter('detection.device').get_parameter_value().string_value
        self.pkg_path = self.get_parameter('pkg_path').get_parameter_value().string_value

        model_paths = {
            "PT": '/models/best.pt',
            "OV": '/models/best_openvino_model',
            "ONNX": '/models/best.onnx'
        }
        if self.model_type not in model_paths:
            self.get_logger().error("Model type not recognised")
        else:
            self.model_path = self.pkg_path + model_paths.get(self.model_type, '/models/best.pt')

        self.get_logger().info(self.model_path)
        self.model = YOLO(self.model_path, task="detect")
        self.get_logger().info(f"---- INSTANCIATED MODEL OF TYPE {self.model_type} ----")

        # Subscribers
        self.camera_sub = self.create_subscription(CompressedImage, '/camera_usb/compressed', self.camera_callback, 1)

        # Publishers
        self.detection_pub = self.create_publisher(CompressedImage, '/detection', 1)
        self.detection_data_pub = self.create_publisher(DetectionResults, '/detection_data', 1)

    def predict(self, chosen_model, img, classes=[], conf=0.8, device='cpu'):
        if classes:
            results = chosen_model.predict(img, classes=classes, conf=conf, device=device)
        else:
            results = chosen_model.predict(img, conf=conf, device=device)

        return results

    def predict_and_detect(self, chosen_model, img, classes=[], conf=0.8, device='cpu'):
        results = self.predict(chosen_model, img, classes, conf, device=device)

        for result in results:
            for box in result.boxes:
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), (255, 0, 0), 2)
                cv2.putText(img, f"{result.names[int(box.cls[0])] + str(round(float(box.conf), 2))}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 1)
        return img, results

    def camera_callback(self, msg):
        start_time = self.get_clock().now()

        # Convert image msg to OpenCV and make a copy because the "img" numpy array WRITABLE flag is set to False and cannot be changed
        img = self.bridge.compressed_imgmsg_to_cv2(msg, "passthrough").copy()
        result_img, results = self.predict_and_detect(self.model, img, classes=[], conf=self.confidence, device=self.device)
        
        end_time = self.get_clock().now()
        dt = end_time - start_time
        self.get_logger().info("Total inference time = {:.2f} ms".format(dt.nanoseconds / 1e6))

        detection_results_msg = DetectionResults()
        detection_results_msg.header = Header()
        detection_results_msg.header.stamp = self.get_clock().now().to_msg()
        detection_results_msg.preprocess_time = [result.speed['preprocess'] for result in results]
        detection_results_msg.inference_time = [result.speed['inference'] for result in results]
        detection_results_msg.postprocess_time = [result.speed['postprocess'] for result in results]

        box_labels = []
        box_confidences = []
        box_coordinates = []

        for result in results:
            for box in result.boxes:
                box_labels.append(result.names[int(box.cls[0])])
                box_confidences.append(float(box.conf))
                box_coordinates.extend([
                    float(box.xyxy[0][0]),
                    float(box.xyxy[0][1]),
                    float(box.xyxy[0][2]),
                    float(box.xyxy[0][3])
                ])

        detection_results_msg.box_labels = box_labels
        detection_results_msg.box_confidences = box_confidences
        detection_results_msg.box_coordinates = box_coordinates

        self.detection_pub.publish(self.bridge.cv2_to_compressed_imgmsg(result_img, "jpg"))
        self.detection_data_pub.publish(detection_results_msg)

    def run(self):
        rclpy.spin(self)


def main(args=None):
    rclpy.init(args=args)
    detector = LitterDetector()
    try:
        detector.run()
    except KeyboardInterrupt:
        pass
    finally:
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

