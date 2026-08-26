from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'laser_triangulation'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),

        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='Computer vision package for BlueROV2 laser tracking',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'laser_perception_node = laser_triangulation.laser_perception_node:main',
            'laser_kinematics_node = laser_triangulation.laser_kinematics_node:main',
            'laser_kinematics_node_simulation = laser_triangulation.laser_kinematics_node_simulation:main',
            'laser_perception_node_centroid = laser_triangulation.laser_perception_node_centroid:main',
            'laser_perception_node_gauss = laser_triangulation.laser_perception_node_gauss:main'
        ],
    },
)
