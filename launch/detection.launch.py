import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    pkg_name = 'litter_detection_ros2'  

    # Parameters to be passed to the node
    model_train = LaunchConfiguration('model_train', default='all_train_640') # wc_train_640, flow_fw_train_640, flow_fw_jarun_train_640, all_train_640 
    model_type = LaunchConfiguration('model_type', default='OV') # PT (PyTorch), OV (OpenVino), ONNX
    confidence = LaunchConfiguration('confidence', default=0.5)
    device = LaunchConfiguration('device', default='cpu')
    avg_inference_window_size = LaunchConfiguration('avg_inference_window_size', default=200)

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
                {'detection.model_train': model_train},
                {'detection.model_type': model_type},
                {'detection.confidence': confidence},
                {'detection.device': device},
                {'detection.avg_inference_window_size': avg_inference_window_size},
                {'pkg_path': pkg_path}
            ]
        )
    ])

