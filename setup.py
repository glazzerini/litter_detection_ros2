from setuptools import find_packages, setup
from glob import glob

package_name = 'litter_detection_ros2'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/detection.launch.py']), 
        ('share/' + package_name + '/launch/cfg', ['launch/cfg/detection.yaml']),
    ],
    install_requires=[
        'setuptools',
        'launch',
        'launch_ros',
        ],
    zip_safe=True,
    maintainer='stdops',
    maintainer_email='stdops@pladypos.cres',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'litter_detection_node = litter_detection_ros2.litter_detection_node:main'
        ],
    },
)
