import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_name = 'litter_detection'  

    # Parameters to be passed to the node
    model_type = LaunchConfiguration('model_type', default='PT') # PT (PyTorch), OV (OpenVino), ONNX
    confidence = LaunchConfiguration('confidence', default=0.5)
    device = LaunchConfiguration('device', default='cpu')

    # Get path to package
    pkg_path = get_package_share_directory(pkg_name)

    return LaunchDescription([
        Node(
            package=pkg_name,
            executable='litter_detection_node',
            name='litter_detector',
            output='screen',
            parameters=[
                {'detection.model_type': model_type},
                {'detection.confidence': confidence},
                {'detection.device': device},
                {'pkg_path': pkg_path}
            ]
        )
    ])

