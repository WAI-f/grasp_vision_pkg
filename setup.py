from glob import glob

from setuptools import find_packages, setup

package_name = 'grasp_vision_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='15071194757@163.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_subscriber = grasp_vision_pkg.camera_subscriber:main',
            'grasp_pose_estimator = grasp_vision_pkg.grasp_pose_estimator:main',
            'sam3_onnx_segmenter = grasp_vision_pkg.sam3_onnx_segmenter:main',
        ],
    },
)
