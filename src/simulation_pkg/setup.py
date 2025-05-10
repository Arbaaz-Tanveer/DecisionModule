from setuptools import setup
import os
from glob import glob

package_name = 'simulation_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='arbaaz',
    maintainer_email='your.email@example.com',
    description='Decision making package for RoboCup MSL',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bot_simulation = simulation_pkg.robot_sim:main',
            'joystick_controller = simulation_pkg.joystick_controller:main',
            'linear_controller = simulation_pkg.linear_controller:main',
            'decision = simulation_pkg.decision_making:main',
            'kinematic_controller = simulation_pkg.kinematic_controller:main',
            'multi_robot_kinematic_controller = simulation_pkg.multi_robot_kinematic_controller:main' 
        ],
    },
)
