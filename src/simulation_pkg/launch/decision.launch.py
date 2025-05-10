from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace



def generate_launch_description():
    return LaunchDescription([
        # Package 1
        GroupAction(
            actions=[
                # PushRosNamespace('simulation_pkg'),
                Node(
                    package='simulation_pkg',
                    executable='bot_simulation',
                    name='bot_simulation',
                    output='screen'
                ),
                Node(
                    package='simulation_pkg',
                    executable='multi_robot_kinematic_controller',
                    name='multi_robot_kinematic_controller',
                    output='screen'
                ),

                # Node(
                #     package='simulation_pkg',
                #     executable='node2',
                #     name='node2',
                #     output='screen'
                # )
            ]
        ),

        GroupAction(
            actions=[
                # PushRosNamespace('o1'),
                Node(
                    package='o1',
                    executable='main1',
                    name='main1',
                    output='screen'
                )
            ]
        ),
        # Package 2
        GroupAction(
            actions=[
                # PushRosNamespace('o2'),
                Node(
                    package='o2',
                    executable='main2',
                    name='main2',
                    output='screen'
                )
            ]
        ),
        # Package 3
        GroupAction(
            actions=[
                # PushRosNamespace('o3'),
                Node(
                    package='o3',
                    executable='main3',
                    name='main3',
                    output='screen'
                )
            ]
        ),
        # Package 4
        GroupAction(
            actions=[
                # PushRosNamespace('o4'),
                Node(
                    package='o4',
                    executable='main4',
                    name='main4',
                    output='screen'
                )
            ]
        ),
        # Package 5
        GroupAction(
            actions=[
                # PushRosNamespace('o5'),
                Node(
                    package='o5',
                    executable='main5',
                    name='main5',
                    output='screen'
                )
            ]
        ),
        
        
    ])



# def generate_launch_description():
#     return LaunchDescription([
#         # Package 1
#         GroupAction(
#             actions=[
#                 # PushRosNamespace('simulation_pkg'),
#                 Node(
#                     package='simulation_pkg',
#                     executable='bot_simulation',
#                     name='bot_simulation',
#                     output='screen'
#                 ),
#                 # Node(
#                 #     package='simulation_pkg',
#                 #     executable='node2',
#                 #     name='node2',
#                 #     output='screen'
#                 # )
#             ]
#         ),

#         GroupAction(
#             actions=[
#                 # PushRosNamespace('o1'),
#                 Node(
#                     package='o1',
#                     executable='main1',
#                     name='main1',
#                     output='screen'
#                 ),
#                 Node(
#                     package='o1',
#                     executable='controller1',
#                     name='controller1',
#                     output='screen'
#                 )
#             ]
#         ),
#         # Package 2
#         GroupAction(
#             actions=[
#                 # PushRosNamespace('o2'),
#                 Node(
#                     package='o2',
#                     executable='main2',
#                     name='main2',
#                     output='screen'
#                 ),
#                 Node(
#                     package='o2',
#                     executable='controller2',
#                     name='controller2',
#                     output='screen'
#                 )
#             ]
#         ),
#         # Package 3
#         GroupAction(
#             actions=[
#                 # PushRosNamespace('o3'),
#                 Node(
#                     package='o3',
#                     executable='main3',
#                     name='main3',
#                     output='screen'
#                 ),
#                 Node(
#                     package='o3',
#                     executable='controller3',
#                     name='controller3',
#                     output='screen'
#                 )
#             ]
#         ),
#         # Package 4
#         GroupAction(
#             actions=[
#                 # PushRosNamespace('o4'),
#                 Node(
#                     package='o4',
#                     executable='main4',
#                     name='main4',
#                     output='screen'
#                 ),
#                 Node(
#                     package='o4',
#                     executable='controller4',
#                     name='controller4',
#                     output='screen'
#                 )
#             ]
#         ),
#         # Package 5
#         GroupAction(
#             actions=[
#                 # PushRosNamespace('o5'),
#                 Node(
#                     package='o5',
#                     executable='main5',
#                     name='main5',
#                     output='screen'
#                 ),
#                 Node(
#                     package='o5',
#                     executable='controller5',
#                     name='controller5',
#                     output='screen'
#                 )
#             ]
#         ),
        
        
#     ])
