# simple_loop_closure
## Overview
This repository is a ROS 2 implementation of Loop Closure for LiDAR SLAM. Currently (as on 15 Oct 2024) it is tested on ROS2 Humble, Ubuntu 22.04 


## Build
Please install [GTSAM4.x](https://gtsam.org/get_started/).

Or
```
# Add PPA
* sudo add-apt-repository ppa:borglab/gtsam-develop
* sudo apt update  # not necessary since Bionic
# Install:
* sudo apt install libgtsam-dev libgtsam-unstable-dev
```
```
* cd ros2_ws/src/

* git clone --recursive-submodules -b ros2 https://github.com/ashBabu/simple_loop_closure.git
```

## Execution
Example: This is [simple_loop_closure.launch.py](launch/simple_loop_closure.launch.py) is currently configured to run [Direct Lidar Inertial Odometry](https://github.com/vectr-ucla/direct_lidar_inertial_odometry). Make modifications as needed
~~~
* Launch your ros2 simulation environment
* ros2 launch simple_loop_closure simple_loop_closure.launch.py
~~~
## Saving Point Cloud Maps
Please put the destination directory in the "save_req" topic and publish.
~~~
ros2 topic pub /save_req std_msgs/msg/String "{data: /home/username/}"
~~~
This will publish a `map.pcd` at the specified directory and also create another directory `/home/username/frames/` where each frames are also saved.

## Parameters
Please refer to [Parameters.md](Parameters.md).

## Requirements for LiDAR Odometry
- LiDAR odometry must publish `nav_msgs::msg::Odometry` format odometry and a paired LiDAR pointcloud of type `sensor_msgs::msg::PointCloud2`.
- The timestamps of the two topics must be almost the same time.