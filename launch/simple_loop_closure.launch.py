import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    ld = LaunchDescription()
    package_dir = get_package_share_directory('simple_loop_closure')
    default_params_file = os.path.join(package_dir, 'config', 'simple_loop_closure.yaml')
    default_odometry_topic="/lm/odom_node/odom"
    default_cloud_topic = "/lm/odom_node/pointcloud/deskewed"

    arg_loop_closure_params_file = LaunchConfiguration('loop_closure_params_file')
    arg_odometry_topic = LaunchConfiguration('odometry_topic')
    arg_cloud_topic = LaunchConfiguration('cloud_topic')

    ld.add_action(DeclareLaunchArgument('loop_closure_params_file', default_value=str(default_params_file),
                                        description='name or path to the parameters file to use.'))
    ld.add_action(DeclareLaunchArgument('odometry_topic', default_value=default_odometry_topic))
    ld.add_action(DeclareLaunchArgument('cloud_topic', default_value=default_cloud_topic))

    remappings=[
        ('/odometry', arg_odometry_topic),
        ('/cloud', arg_cloud_topic)
    ]

    node_loop_closure = Node(
        package='simple_loop_closure',
        # executable='test_lc_node',
        executable='simple_loop_closure_node',
        name='simple_loop_closure',
        output='both',
        parameters=[arg_loop_closure_params_file],
        remappings=remappings,
        # respawn=True,
        # respawn_delay=3
    )
    ld.add_action(node_loop_closure)

    return ld