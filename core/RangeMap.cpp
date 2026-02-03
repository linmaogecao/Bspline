//
// Created by albus on 2026/2/3.
//

#include "RangeMap.h"

void RangeImageProcessor::generateRangeImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    std::fill(range_image_.begin(), range_image_.end(), RangePixel());

    float fov_up_rad = FOV_UP * M_PI / 180.0f;
    float fov_down_rad = FOV_DOWN * M_PI / 180.0f;
    float fov_total_rad = std::abs(fov_up_rad - fov_down_rad);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& pt = cloud->points[i];

        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;
        double range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if (range < 1.0 || range > 120.0) continue;

        double angle_vert = std::asin(pt.z / range);

        // 归一化到 [0, 1]
        double row_ratio = (angle_vert - fov_down_rad) / fov_total_rad;
        int row = std::round(row_ratio * (H_SCANS - 1));

        double angle_horiz = std::atan2(pt.y, pt.x);
        int col = std::round((angle_horiz + M_PI) / (2.0 * M_PI) * W_COLS);
        if (col >= W_COLS) col -= W_COLS;
        if (col < 0) col += W_COLS;

        if (row >= 0 && row < H_SCANS && col >= 0 && col < W_COLS) {
            int idx = row * W_COLS + col;
            RangePixel& px = range_image_[idx];
            if (!px.valid || range < px.range) {
                px.x = pt.x;
                px.y = pt.y;
                px.z = pt.z;
                px.range = range;
                px.valid = true;
            }
        }
    }
}
