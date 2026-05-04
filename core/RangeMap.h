//
// Created by albus on 2026/2/3.
//

#ifndef SPLINE_FITTING_RANGEMAP_H
#define SPLINE_FITTING_RANGEMAP_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sys/stat.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <opencv4/opencv2/opencv.hpp>
#include <Eigen/Eigenvalues>


// OpenMP 用于加速
#include <omp.h>

struct RangePixel {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double range = 0.0;
    bool valid = false;
    std::pair<double, double> curvature;
    bool is_visted = false;
};

struct SegmentationResult {
    std::vector<int> label_map;             // 全图标签
    std::vector<std::vector<int>> clusters; // 每个面的索引集合
};

struct VoxelGrid {
    double resolution;  // 体素大小，如 0.5m
    std::unordered_map<std::string, std::vector<int>> voxel_map;  // key: "x_y_z", value: 点索引列表

    VoxelGrid(double res = 0.5) : resolution(res) {}

    std::string getKey(double x, double y, double z) const {
        int vx = static_cast<int>(std::floor(x / resolution));
        int vy = static_cast<int>(std::floor(y / resolution));
        int vz = static_cast<int>(std::floor(z / resolution));
        return std::to_string(vx) + "_" + std::to_string(vy) + "_" + std::to_string(vz);
    }

    void addPoint(const RangePixel& px, int index) {
        if (!px.valid) return;
        std::string key = getKey(px.x, px.y, px.z);
        voxel_map[key].push_back(index);
    }
};


class RangeImageProcessor {
public:
    const int H_SCANS = 64;
    const int W_COLS = 1500;
    const float FOV_UP = 2.0f;
    const float FOV_DOWN = -24.8f;
    double alpha_vert_rad_;
    double alpha_horiz_rad_;
    std::vector<RangePixel> range_image_;
    pcl::PointCloud<pcl::PointXYZ> ouyt;
    RangeImageProcessor() {
        range_image_.resize(H_SCANS * W_COLS);
        alpha_vert_rad_ = (FOV_UP * M_PI / 180.0f - FOV_DOWN * M_PI / 180.0f)/(H_SCANS - 1);
        alpha_horiz_rad_ = (2.0 * M_PI) / W_COLS;

    }
    void saveRangeImageBin(const std::string& filename);
    // -----------------------------------------------------------------
    // 2. 核心函数: PointCloud -> RangeImage
    // -----------------------------------------------------------------
    void generateRangeImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    bool getPoint(int u, int v, Eigen::Vector3d& out_point) const {
        // 处理 V 方向 (水平) 的周期性
        v = v % W_COLS;
        v += (v < 0) * W_COLS;

        bool u_valid = (unsigned)u < (unsigned)H_SCANS;

        int idx = u * W_COLS + v;
        const auto& px = range_image_[idx];
        out_point << px.x, px.y, px.z;
        return u_valid & px.valid;
    }
    int wrapCol(int v) const {
        int nv = (v + W_COLS) % W_COLS;
        return nv;
    }

    bool computePixelCurvature(int u, int v, std::pair<double, double>& curvature);
    SegmentationResult segmentRangeImage(double theta_deg, double max_h_curvature, double max_v_curvature, double max_dist, int min_cluster_size);
    void saveClustersToTxt(const SegmentationResult& result, const std::string& folder_path);
    bool findValidNeighborPt(int u, int v, const Eigen::Vector3d& center_pt, Eigen::Vector3d& neighbor_pt, bool is_vertical = false, int dir = 1) const;
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> generateClusterClouds(const SegmentationResult& result);
};


#endif //SPLINE_FITTING_RANGEMAP_H