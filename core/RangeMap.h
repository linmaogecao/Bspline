//
// Created by albus on 2026/2/3.
//

#ifndef SPLINE_FITTING_RANGEMAP_H
#define SPLINE_FITTING_RANGEMAP_H

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <algorithm>

// PCL Headers
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>

// OpenMP 用于加速
#include <omp.h>

struct RangePixel {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double range = 0.0;
    bool valid = false;
};

class RangeImageProcessor {
public:
    const int H_SCANS = 64;
    const int W_COLS = 1800;
    const float FOV_UP = 2.0f;
    const float FOV_DOWN = -24.8f;

    std::vector<RangePixel> range_image_;

    RangeImageProcessor() {
        range_image_.resize(H_SCANS * W_COLS);
    }

    // -----------------------------------------------------------------
    // 2. 核心函数: PointCloud -> RangeImage
    // -----------------------------------------------------------------
    void generateRangeImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud);
    bool getPoint(int u, int v, Eigen::Vector3d& out_point) {
        // 处理 V 方向 (水平) 的周期性
        if (v < 0) v += W_COLS;
        if (v >= W_COLS) v -= W_COLS;

        // U 方向越界则无效
        if (u < 0 || u >= H_SCANS) return false;

        int idx = u * W_COLS + v;
        const auto& px = range_image_[idx];

        if (px.valid) {
            out_point << px.x, px.y, px.z;
            return true;
        }
        return false;
    }
};


#endif //SPLINE_FITTING_RANGEMAP_H