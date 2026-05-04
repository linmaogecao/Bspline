//
// Created by albus on 2026/2/3.
//

#include "RangeMap.h"

struct QueueItem {
    int index;
    Eigen::Vector3d accumulated_direction;  // Smoothed direction over last few steps
    int step_count;
};

void RangeImageProcessor::generateRangeImage(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud)
{
    std::fill(range_image_.begin(), range_image_.end(), RangePixel());
    int cnt = 0;
    int cnt_1 = 0;
    int cnt_2 = 0;
    float fov_up_rad = FOV_UP * M_PI / 180.0f;
    float fov_down_rad = FOV_DOWN * M_PI / 180.0f;
    float fov_total_rad = std::abs(fov_up_rad - fov_down_rad);
    //#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < cloud->size(); ++i) {
        const auto& pt = cloud->points[i];
        if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z))
            continue;
        double range = std::sqrt(pt.x * pt.x + pt.y * pt.y + pt.z * pt.z);
        if (range < 1.0 || range > 50.0) {
            cnt_1++;
            continue;
        }

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
            cnt_2++;
            if (!px.valid || range < px.range) {
                px.x = pt.x;
                px.y = pt.y;
                px.z = pt.z;
                px.range = range;
                px.valid = true;
            }
        }

    }
    ouyt.reserve(cloud->size()); // 预分配内存优化
    for (const auto& px : range_image_) {
        if (px.valid) {
            pcl::PointXYZ p;
            p.x = px.x;
            p.y = px.y;
            p.z = px.z;
            ouyt.push_back(p);
            cnt++;
        }
    }
    saveRangeImageBin("test1.bin");
    std::cout<<"init cnt "<<cnt<<" "<<cnt_1<<" "<<cnt_2<<std::endl;
}

void RangeImageProcessor::saveRangeImageBin(const std::string& filename) {
    std::ofstream out(filename, std::ios::binary);
    if (!out) {
        std::cerr << "无法打开文件 " << filename << " 进行写入!\n";
        return;
    }

    // 1. 写入头信息：高 (H_SCANS) 和 宽 (W_COLS)
    int h = H_SCANS;
    int w = W_COLS;
    out.write(reinterpret_cast<const char*>(&h), sizeof(int));
    out.write(reinterpret_cast<const char*>(&w), sizeof(int));

    // 2. 逐个像素写入数据
    for (int i = 0; i < H_SCANS * W_COLS; ++i) {
        const auto& px = range_image_[i];

        // 提取出基本数据类型，直接取地址写入，不搞任何花里胡哨的数组
        int valid = px.valid ? 1 : 0;
        double r = px.range;
        double x = px.x;
        double y = px.y;
        double z = px.z;

        // 逐个变量写入二进制文件
        out.write(reinterpret_cast<const char*>(&valid), sizeof(int));
        out.write(reinterpret_cast<const char*>(&r), sizeof(double));
        out.write(reinterpret_cast<const char*>(&x), sizeof(double));
        out.write(reinterpret_cast<const char*>(&y), sizeof(double));
        out.write(reinterpret_cast<const char*>(&z), sizeof(double));
    }

    out.close();
    std::cout << ">>> Range Image 已成功保存至: " << filename << " <<<\n";
}

bool RangeImageProcessor::computePixelCurvature(int u, int v, std::pair<double, double>& curvature) {
    Eigen::Vector3d center_pt;
    curvature.first = -1.0;
    curvature.second = -1.0;

    bool valid_any = false; //confirm  one direction has curvrate
    if (!getPoint(u, v, center_pt)) return false;

    Eigen::Vector3d left_pt, right_pt;
    bool is_left_valid = findValidNeighborPt(u,v,center_pt,left_pt,false,1);
    bool is_right_valid =findValidNeighborPt(u,v,center_pt,right_pt,false,-1);

    if (is_left_valid && is_right_valid) {
        Eigen::Vector3d v_left = left_pt - center_pt;
        Eigen::Vector3d v_right = right_pt - center_pt;
        double norm_l = v_left.norm();
        double norm_r = v_right.norm();
        if (norm_l > 1e-3 && norm_r > 1e-3) {
            // 3. 计算 Cosine 值
            double dot_product = v_left.dot(v_right);
            double cos_theta = dot_product / (norm_l * norm_r);

            // 钳制数值防止 acos 越界 (比如计算误差导致 1.000001)
            if (cos_theta > 1.0) cos_theta = 1.0;
            if (cos_theta < -1.0) cos_theta = -1.0;

            // 4. 计算角度 (弧度 -> 度)
            double angle_rad = std::acos(cos_theta);
            double angle_deg = angle_rad * 180.0 / M_PI;
            // if (angle_deg < 130 && (center_pt.x() > 2.1 || center_pt.x() < 1.9 || center_pt.y() >2.1|| center_pt.y() <1.9)) {
            //     std::cout << "center: "<<center_pt.transpose()<<std::endl;
            //     std::cout << "left_pt: "<<left_pt.transpose()<<std::endl;
            //     std::cout << "right_pt: "<<right_pt.transpose()<<std::endl;
            //     std::cout << "v_left: "<<v_left.transpose()<<std::endl;
            //     std::cout << "v_right: "<<v_right.transpose()<<std::endl;
            //     std::cout << "cos_theta: "<<angle_deg<<std::endl;
            // }
            curvature.first = angle_deg;

            valid_any = true;
        }
    }

    // Eigen::Vector3d up_pt, down_pt;
    // bool is_up_valid = findValidNeighborPt(u,v,center_pt,up_pt,true,1);
    // bool is_down_valid =findValidNeighborPt(u,v,center_pt,down_pt,true,-1);
    // if (is_up_valid && is_down_valid)
    // {
    //     Eigen::Vector3d v_up = up_pt - center_pt;
    //     Eigen::Vector3d v_down = down_pt - center_pt;
    //
    //     double norm_u = v_up.norm();
    //     double norm_d = v_down.norm();
    //     if (norm_u > 1e-3 && norm_d > 1e-3) {
    //         double dot_product = v_up.dot(v_down);
    //         double cos_theta = dot_product / (norm_u * norm_d);
    //
    //         if (cos_theta > 1.0) cos_theta = 1.0;
    //         if (cos_theta < -1.0) cos_theta = -1.0;
    //
    //         double angle_deg = std::acos(cos_theta) * 180.0 / M_PI;
    //
    //         curvature.second = angle_deg;
    //         valid_any = true;
    //     }
    // }
    return valid_any;
}
SegmentationResult RangeImageProcessor::segmentRangeImage(double theta_deg, double max_h_curvature, double max_v_curvature, double max_dist, int min_cluster_size) {
    SegmentationResult result;
    int pixel_num = H_SCANS * W_COLS;
    result.label_map.assign(pixel_num, 0);

    // 1. 初始化 Valid
    std::vector<bool> pixel_valid(pixel_num, false);
    double theta_rad = theta_deg * M_PI / 180.0;
    int test_cnt = 0;
    for (int i = 0; i < H_SCANS; ++i) {
        for (int j = 0; j < W_COLS; ++j) {
            int idx = i * W_COLS + j;
            if (range_image_[idx].valid) {
                pixel_valid[idx] = true;
                test_cnt++;
            }
        }
    }
    std::cout<<"test_cnt: "<<test_cnt<<std::endl;
    int current_label = 0;
    int dir_u[8] = {-1, 1, 0, 0, -1, -1, 1, 1};
    int dir_v[8] = {0, 0, -1, 1, -1, 1, -1, 1};
    int cluster_cnt = 0;
    for (int index = 0; index < pixel_num; ++index) {
        if (!pixel_valid[index]) continue;
        if (result.label_map[index] != 0) continue;
        // if (pixel_curvate[index].first < max_h_curvature) {
        //     if (range_image_[index].y > -10 && range_image_[index].y < -6 && range_image_[index].x >-5 && range_image_[index].x <6) {
        //         std::cout << "range_image_[cur_index] " << range_image_[index].x << " " << range_image_[index].y << " " << range_image_[index].z<<std::endl;
        //
        //         std::cout << "reject self by curvarate : " << pixel_curvate[index].first << " > " << max_h_curvature << std::endl;
        //     }
        //     continue;
        // }

        current_label++;
        std::vector<int> current_cluster;
        std::deque<int> q;

        // 种子入队
        result.label_map[index] = current_label;
        current_cluster.push_back(index);
        q.push_back(index);
        int last_neighour_idx = -1;
        while (!q.empty()) {
            int cur_index = q.front(); q.pop_front();
            
            int u = cur_index / W_COLS;
            int v = cur_index % W_COLS;
            double crange = range_image_[cur_index].range;
            for (int k = 0; k < 8; ++k) {
                int found_neighbor_idx = -1;
                int found_step = 0;

                // 根据方向设定最大跳跃步长 (垂直方向跳跃少一点，水平/对角线跳跃可以多一点)
                int max_step = (k < 2) ? 2 : 5;

                for (int step = 1; step <= max_step; ++step) {
                    int neighbour_u = u + dir_u[k] * step;
                    int neighbour_v = wrapCol(v + dir_v[k] * step); // 确保水平方向能 wrap around

                    if (neighbour_u < 0 || neighbour_u >= H_SCANS) break; // 超出上下边界

                    int temp_idx = neighbour_u * W_COLS + neighbour_v;

                    if (!pixel_valid[temp_idx]) continue; // 核心：如果是无效点(黑洞)，继续往远处看

                    // 找到了当前方向上*第一个*有效点！
                    if (result.label_map[temp_idx] == 0) {
                        found_neighbor_idx = temp_idx;
                        found_step = step;
                    }
                    break; // 极其重要：无论这个点是否满足后续条件，视线已经被挡住，立刻停止在这条射线上的搜索！
                }

                if (found_neighbor_idx == -1) continue;
                int n_u = found_neighbor_idx / W_COLS;
                int n_v = found_neighbor_idx % W_COLS;
                // if (range_image_[cur_index].y > -8 && range_image_[cur_index].y < -5 && range_image_[cur_index].x >15.8 && range_image_[cur_index].x <17.3) {
                //     std::cout << "\n[BFS] 当前(u:" << u << ", v:" << v << ") -> 探寻方向k:" << k
                //               << " 步长:" << found_step << " -> 邻居(u:" << n_u << ", v:" << n_v << ")\n";
                //     std::cout << "  - 当前点 3D: (" << range_image_[cur_index].x << ", " << range_image_[cur_index].y << ", " << range_image_[cur_index].z << ")\n";
                //     std::cout << "  - 邻居点 3D: (" << range_image_[found_neighbor_idx].x << ", " << range_image_[found_neighbor_idx].y << ", " << range_image_[found_neighbor_idx].z << ")\n";
                //     }

                double nrange = range_image_[found_neighbor_idx].range;
                // Adaptive threshold: larger for farther points, smaller for closer points
                double avg_range = (crange + nrange) * 0.5;
                double adaptive_max_dist = max_dist + avg_range * 0.02;
                double dx = range_image_[cur_index].x - range_image_[found_neighbor_idx].x;
                double dy = range_image_[cur_index].y - range_image_[found_neighbor_idx].y;
                double dz = range_image_[cur_index].z - range_image_[found_neighbor_idx].z;
                double min_euclidean_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (min_euclidean_dist > adaptive_max_dist) {
                    if (range_image_[found_neighbor_idx].y > -6 && range_image_[found_neighbor_idx].y < -5.5 && range_image_[found_neighbor_idx].x >22 && range_image_[found_neighbor_idx].x <22.2 ) {
                        std::cout << "range_image_[cur_index] " << range_image_[cur_index].x << " " << range_image_[cur_index].y << " " << range_image_[cur_index].z << std::endl;
                        std::cout << "range_image_[found_neighbor_idx] " << range_image_[found_neighbor_idx].x << " " << range_image_[found_neighbor_idx].y << " " << range_image_[found_neighbor_idx].z << std::endl;
                        std::cout << "REJECTD by Euclidean Dist: " << min_euclidean_dist << " > " << adaptive_max_dist << std::endl;
                        std::cout<< "u "<<(cur_index / W_COLS) << "v "<<(cur_index % W_COLS) << std::endl;
                        std::cout<< "n_u "<<(found_neighbor_idx / W_COLS) << "n_v "<<(found_neighbor_idx % W_COLS) << std::endl;
                    }
                    continue;
                }

                if (last_neighour_idx > 0) {
                    double dx_last = range_image_[last_neighour_idx].x - range_image_[found_neighbor_idx].x;
                    double dy_last = range_image_[last_neighour_idx].y - range_image_[found_neighbor_idx].y;
                    double dz_last = range_image_[last_neighour_idx].z - range_image_[found_neighbor_idx].z;

                    double min_euclidean_dist_last = std::sqrt(dx_last*dx_last + dy_last*dy_last + dz_last*dz_last);
                    if (min_euclidean_dist_last > 1.3 * adaptive_max_dist) {
                        continue;
                    }
                    if (range_image_[found_neighbor_idx].y > -6 && range_image_[found_neighbor_idx].y < -5.5 && range_image_[found_neighbor_idx].x >21.6 && range_image_[found_neighbor_idx].x <22.2) {
                        std::cout << "range_image_[last _neighou_idx]" << range_image_[last_neighour_idx].x << " " << range_image_[last_neighour_idx].y << " " << range_image_[last_neighour_idx].z << std::endl;
                        std::cout << "range_image_[cur_index] " << range_image_[cur_index].x << " " << range_image_[cur_index].y << " " << range_image_[cur_index].z << std::endl;
                        std::cout << "range_image_[found_neighbor_idx] " << range_image_[found_neighbor_idx].x << " " << range_image_[found_neighbor_idx].y << " " << range_image_[found_neighbor_idx].z << std::endl;
                        std::cout << "ACC by min_euclidean_dist_last Dist: " << min_euclidean_dist_last << " < " << 1.3 * adaptive_max_dist << std::endl;
                        std::cout<< "u "<<(cur_index / W_COLS) << "v "<<(cur_index % W_COLS) << std::endl;
                        std::cout<< "n_u "<<(found_neighbor_idx / W_COLS) << "n_v "<<(found_neighbor_idx % W_COLS) << std::endl;
                        std::cout <<"last_u "<<(last_neighour_idx / W_COLS) << "last_v "<<(last_neighour_idx % W_COLS) << std::endl;
                        std::cout << "clust" << cluster_cnt << std::endl;
                    }
                }
                last_neighour_idx = cur_index;

                double d1 = std::max(crange, nrange);
                double d2 = std::min(crange, nrange);
                double alpha_rad_base = (k < 2) ? alpha_vert_rad_ : alpha_horiz_rad_;
                double alpha_rad = alpha_rad_base * found_step;

                double denom = d1 - d2 * std::cos(alpha_rad);
                if (std::abs(denom) >= 1e-9) {
                    double belta = std::atan2(d2 * std::sin(alpha_rad), denom);
                    theta_rad = k<2 ? 30* M_PI / 180.0 : 2.5 * M_PI / 180.0;
                    if (belta > theta_rad) {
                        if (range_image_[found_neighbor_idx].y > -6 && range_image_[found_neighbor_idx].y < -5.5 && range_image_[found_neighbor_idx].x >21.6 && range_image_[found_neighbor_idx].x <22.2) {
                            std::cout << "range_image_[cur_index] " << range_image_[cur_index].x << " " << range_image_[cur_index].y << " " << range_image_[cur_index].z << std::endl;
                            std::cout << "range_image_[found_neighbor_idx] " << range_image_[found_neighbor_idx].x << " " << range_image_[found_neighbor_idx].y << " " << range_image_[found_neighbor_idx].z << std::endl;
                            std::cout << "ACC by Euclidean Dist: " << min_euclidean_dist << " < " << adaptive_max_dist << std::endl;
                            std::cout << "ACC by theta_rad : " << belta << " > " << theta_rad << std::endl;
                            std::cout<< "u "<<(cur_index / W_COLS) << "v "<<(cur_index % W_COLS) << std::endl;
                            std::cout<< "n_u "<<(found_neighbor_idx / W_COLS) << "n_v "<<(found_neighbor_idx % W_COLS) << std::endl;
                            std::cout << "clust" << cluster_cnt << std::endl;
                        }
                        result.label_map[found_neighbor_idx] = current_label;
                        current_cluster.push_back(found_neighbor_idx);
                        q.push_back(found_neighbor_idx);
                    }
                    else {
                        if (range_image_[found_neighbor_idx].y > -6 && range_image_[found_neighbor_idx].y < -5.5 && range_image_[found_neighbor_idx].x >21.6 && range_image_[found_neighbor_idx].x <22.2) {
                            std::cout << "range_image_[cur_index] " << range_image_[cur_index].x << " " << range_image_[cur_index].y << " " << range_image_[cur_index].z << std::endl;
                            std::cout << "range_image_[found_neighbor_idx] " << range_image_[found_neighbor_idx].x << " " << range_image_[found_neighbor_idx].y << " " << range_image_[found_neighbor_idx].z << std::endl;
                            std::cout << "ACC by theta_rad : " << belta << " > " << theta_rad << std::endl;
                            std::cout << "clust" << cluster_cnt << std::endl;
                            std::cout<< "u "<<(cur_index / W_COLS) << "v "<<(cur_index % W_COLS) << std::endl;
                            std::cout<< "n_u "<<(found_neighbor_idx / W_COLS) << "n_v "<<(found_neighbor_idx % W_COLS) << std::endl;
                        }
                    }
                }
            }
        }

        if (current_cluster.size() > min_cluster_size) {
            result.clusters.push_back(std::move(current_cluster));
            cluster_cnt++;
        }
    }
    return result;
}


void RangeImageProcessor::saveClustersToTxt(const SegmentationResult& result, const std::string& folder_path) {
    if (result.clusters.empty()) {
        std::cout << "No clusters to save!" << std::endl;
        return;
    }

    std::cout << "Saving " << result.clusters.size() << " clusters to " << folder_path << " ..." << std::endl;

    // 遍历每一个聚类
    for (size_t i = 0; i < result.clusters.size(); ++i) {
        const auto& cluster_indices = result.clusters[i];

        // 1. 构造文件名: folder/cluster_0.txt
        // 注意：请确保 folder_path 文件夹已经存在，否则 ofstream 会打开失败
        std::string filename = folder_path + "/cluster_" + std::to_string(i) + ".txt";

        std::ofstream outfile(filename);
        if (!outfile.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            continue;
        }

        // 2. 设置输出精度 (保留4位小数)
        outfile << std::fixed << std::setprecision(4);

        // 3. 遍历聚类中的每一个索引，从 range_image_ 中取 XYZ
        for (int idx : cluster_indices) {
            // 你的 range_image_ 是成员变量，直接访问
            const auto& px = range_image_[idx];

            // 这里不需要再 check valid 了，因为分割时已经过滤过了
            outfile << px.x << " " << px.y << " " << px.z << "\n";
        }

        outfile.close();
    }

    std::cout << "All clusters saved." << std::endl;
}

bool RangeImageProcessor::findValidNeighborPt(int u, int v, const Eigen::Vector3d& center_pt, Eigen::Vector3d &neighbor_pt, bool is_vertical, int dir) const {
    double best_dist = std::numeric_limits<double>::max();
    bool found = false;
    
    // Adaptive distance threshold based on range to center
    double center_range = center_pt.norm();
    double max_neighbor_dist = 0.1 + center_range * 0.03;  // Closer points need smaller threshold
    
    if (!is_vertical) {
        // Horizontal direction: search in a small window
        for (int du = -1; du <= 1; ++du) {
            int curr_u = u + du;
            if (curr_u < 0 || curr_u >= H_SCANS) continue;
            
            for (int i = 1; i <= 5; ++i) {
                int curr_v = wrapCol(v + dir * i);
                Eigen::Vector3d temp_pt;
                
                if (getPoint(curr_u, curr_v, temp_pt)) {
                    double dist = (temp_pt - center_pt).norm();
                    
                    // Must be within reasonable distance AND closer than previous best
                    if (dist < max_neighbor_dist && dist < best_dist && dist > 1e-6) {
                        best_dist = dist;
                        neighbor_pt = temp_pt;
                        found = true;
                    }
                }
            }
        }
    } else {
        // Vertical direction: search in a window around expected position
        for (int i = 1; i <= 5; ++i) {
            int curr_u = u + dir * i;
            if (curr_u < 0 || curr_u >= H_SCANS) continue;
            
            for (int dv = -10; dv <= 10; ++dv) {
                int curr_v = wrapCol(v + dv);
                Eigen::Vector3d temp_pt;
                
                if (getPoint(curr_u, curr_v, temp_pt)) {
                    double dist = (temp_pt - center_pt).norm();
                    
                    if (dist < max_neighbor_dist && dist < best_dist && dist > 1e-6) {
                        best_dist = dist;
                        neighbor_pt = temp_pt;
                        found = true;
                    }
                }
            }
            
            // If we found a good neighbor in this row, don't search further rows
            if (found) break;
        }
    }
    
    return found;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> RangeImageProcessor::generateClusterClouds(const SegmentationResult& result) {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_list;

    if (result.clusters.empty()) {
        return cloud_list;
    }

    // 遍历每一个聚类
    for (size_t i = 0; i < result.clusters.size(); ++i) {
        const auto& cluster_indices = result.clusters[i];

        // 创建一个新的点云对象
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster(new pcl::PointCloud<pcl::PointXYZ>);

        // 预分配内存优化
        current_cluster->reserve(cluster_indices.size());

        // 遍历索引填充点
        for (int idx : cluster_indices) {
            const auto& px = range_image_[idx];
            current_cluster->push_back(pcl::PointXYZ(px.x, px.y, px.z));
        }

        // 设置点云属性
        current_cluster->width = current_cluster->points.size();
        current_cluster->height = 1;
        current_cluster->is_dense = true;

        // 加入列表
        cloud_list.push_back(current_cluster);
    }

    return cloud_list;
}