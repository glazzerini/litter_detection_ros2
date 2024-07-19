import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_name = 'litter_detection_ros2'  

    # Parameters to be passed to the node
    model_type = LaunchConfiguration('model_type', default='ONNX') # PT (PyTorch), OV (OpenVino), ONNX
    confidence = LaunchConfiguration('confidence', default=0.78)
    device = LaunchConfiguration('device', default='cpu')

    # Get path to package
    pkg_share_path = get_package_share_directory(pkg_name)

    # Get the workspace root by going up three levels from the share directory
    workspace_root = os.path.abspath(os.path.join(pkg_share_path, '../../../..'))

    # Construct the source directory path
    pkg_path = os.path.join(workspace_root, 'src', pkg_name)
    
    
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

