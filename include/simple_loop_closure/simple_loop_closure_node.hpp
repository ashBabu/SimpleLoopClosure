#pragma once

#define PCL_NO_PRECOMPILE

// #include <ros/ros.h>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/color_rgba.hpp>
// #include <std_msgs/String.h>
#include <std_msgs/msg/string.hpp>
// #include <nav_msgs/Odometry.h>
// #include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
// #include <eigen_conversions/eigen_msg.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <Eigen/Dense>

#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <fstream>
#include <stdexcept>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/io/pcd_io.h>

#include <nanoflann.hpp>

#include <gtsam/geometry/Rot3.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/Marginals.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/ISAM2.h>

#include <vector>
#include <deque>
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <memory>

class SimpleLoopClosureNode : public rclcpp::Node
{
private: 
    typedef pcl::PointXYZI PointType;
    typedef pcl::PointCloud<PointType> PointCloudType;
    typedef std::lock_guard<std::mutex> MtxLockGuard;
    typedef std::shared_ptr<Eigen::Affine3d> Affine3dPtr;
    typedef Eigen::Matrix<double, Eigen::Dynamic, 3> KDTreeMatrix;
    typedef nanoflann::KDTreeEigenMatrixAdaptor<KDTreeMatrix, 3, nanoflann::metric_L2_Simple> KDTree;
    typedef std::vector<nanoflann::ResultItem<long int, double>> NanoFlannSearchResult;
    typedef std::pair<int, int> LoopEdgeID;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::PointCloud2, nav_msgs::msg::Odometry> SyncPolicy;
    typedef message_filters::Synchronizer<SyncPolicy> Sync;

    // ros::Publisher pub_map_cloud_;
    // ros::Publisher pub_vis_pose_graph_;
    // ros::Publisher pub_pgo_odometry_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_map_cloud_{nullptr};
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_vis_pose_graph_{nullptr};
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_pgo_odometry_{nullptr};


    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_cloud_;
    message_filters::Subscriber<nav_msgs::msg::Odometry> sub_odom_;
    std::shared_ptr<Sync> synchronizer_;

    // ros::Subscriber sub_save_req_;
    // rclcpp::Subscription<sensor_msgs::msg::LaserScan>::SharedPtr sub_save_req_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr sub_save_req_;

    std_msgs::msg::ColorRGBA odom_edge_color_;
    std_msgs::msg::ColorRGBA loop_edge_color_;
    std_msgs::msg::ColorRGBA node_color_;
    double edge_scale_;
    double node_scale_;

    std::mutex mtx_buf_;
    std::deque<PointCloudType::Ptr> keyframes_cloud_;
    std::deque<Affine3dPtr> keyframes_odom_;
    std::deque<PointCloudType::Ptr> keyframes_cloud_copied_;
    std::deque<Affine3dPtr> keyframes_odom_copied_;
    std::deque<double> trajectory_dist_;
    std::deque<double> trajectory_dist_copied_;
    std::string odom_frame_id_;

    size_t added_odom_id_;
    size_t searched_loop_id_;

    gtsam::ISAM2 isam2_;
    std::mutex mtx_res_;
    gtsam::Values optimization_result_;
    std::deque<LoopEdgeID> loop_edges_;

    gtsam::SharedNoiseModel prior_noise_;
    gtsam::SharedNoiseModel odom_noise_;
    gtsam::SharedNoiseModel const_loop_edge_noise_;

    pcl::IterativeClosestPoint<PointType, PointType> icp_;
    pcl::VoxelGrid<PointType> vg_target_;
    pcl::VoxelGrid<PointType> vg_source_;
    pcl::VoxelGrid<PointType> vg_map_;

    bool mapped_cloud_;
    double time_stamp_tolerance_;
    double keyframe_dist_th_;
    double keyframe_angular_dist_th_;
    double loop_search_time_diff_th_;
    double loop_search_dist_diff_th_;
    double loop_search_angular_dist_th_;
    int loop_search_frame_interval_;
    double search_radius_;
    int target_frame_num_;
    double target_voxel_leaf_size_;
    double source_voxel_leaf_size_;
    double vis_map_voxel_leaf_size_;
    double fitness_score_th_;
    int vis_map_cloud_frame_interval_;

    bool stop_loop_closure_thread_;
    bool stop_visualize_thread_;

    std::thread save_thread_;
    bool saving_;
    std::string save_directory_;

    void initialize();

    bool saveEachFrames();

    void saveThread();

    void saveRequestCallback(const std_msgs::msg::String::ConstSharedPtr &directory);

    void publishPoseGraphOptimizedOdometry(const Eigen::Affine3d &affine_curr, const nav_msgs::msg::Odometry &odom_msg);

    void pointCloudAndOdometryCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr &cloud_msg, const nav_msgs::msg::Odometry::ConstSharedPtr &odom_msg);

    pcl::PointCloud<pcl::PointXYZI>::Ptr constructPointCloudMap(const int interval = 0);
    // PointCloudType::Ptr constructPointCloudMap(const int interval = 0);

    void publishMapCloud(const PointCloudType::Ptr &map_cloud);

    void constructVisualizationOdometryEdges(const int &res_size, const std_msgs::msg::Header &header, visualization_msgs::msg::Marker &marker_msg);

    void constructVisualizationLoopEdges(const int &res_size, const std_msgs::msg::Header &header, visualization_msgs::msg::Marker &marker_msg);

    void constructVisualizationNodes(const int &res_size, const std_msgs::msg::Header &header, visualization_msgs::msg::Marker &marker_msg);

    void publishVisualizationGraph();

    void visualizeThread();

    void copyKeyFrames();

    void getLastOptimizedPose(Eigen::Affine3d &pose, int &id);

    bool constructOdometryGraph(gtsam::NonlinearFactorGraph &graph, gtsam::Values &init_estimate);

    bool constructKDTreeMatrix(KDTreeMatrix &kdtree_mat);

    int searchTarget(const KDTree &kdtree, const int &id_query, const Eigen::Affine3d &pose_query, const rclcpp::Time &stamp_query);

    pcl::PointCloud<pcl::PointXYZI>::Ptr constructTargetCloud(const int target_id);

    pcl::PointCloud<pcl::PointXYZI>::Ptr constructSourceCloud(const int source_id);

    bool tryRegistration(const Eigen::Affine3d &init_pose, const PointCloudType::Ptr &source_cloud, const PointCloudType::Ptr &target_cloud, Eigen::Affine3d &result, double &score);

    bool constructLoopEdge(gtsam::NonlinearFactorGraph &graph);

    bool updateISAM2(const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &init_estimate);

    void loopCloseThread();

public:
    typedef std::shared_ptr<SimpleLoopClosureNode> Ptr;

    SimpleLoopClosureNode();

    ~SimpleLoopClosureNode() = default;

    void spin();
};