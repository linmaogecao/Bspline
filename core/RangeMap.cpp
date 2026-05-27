//
// Created by albus on 2026/2/3.
//

#include "RangeMap.h"
#include <sstream>

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
        if (range < MIN_RANGE || range > MAX_RANGE || pt.z < MIN_Z) {
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
            if (!px.valid || range < px.range)
            {
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
    //saveRangeImageBin("test1.bin");
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

SegmentationResult RangeImageProcessor::segmentRangeImage(double theta_deg, double normal_angle_deg, double max_v_curvature, double max_dist, int min_cluster_size) {
    SegmentationResult result;
    const int pixel_num = H_SCANS * W_COLS;
    result.label_map.assign(pixel_num, 0);

    const double theta_rad = theta_deg * M_PI / 180.0;
    const double normal_cos_thresh = std::cos(normal_angle_deg * M_PI / 180.0);

    // ---------- 1. 标记有效像素 ----------
    std::vector<bool> pixel_valid(pixel_num, false);
    for (int idx = 0; idx < pixel_num; ++idx) {
        if (range_image_[idx].valid) pixel_valid[idx] = true;
    }

    // ---------- 2. 预计算法向量 ----------
    // 以当前点 P 为顶点, 取左邻居 P_left = P(u, v-1) 和下邻居 P_down = P(u+1, v)
    // n = (P_left - P) x (P_down - P), 然后归一化
    // 朝向: 让法向量大致指向传感器原点 O (即 -P 方向)
    std::vector<Eigen::Vector3d> normals(pixel_num, Eigen::Vector3d::Zero());
    std::vector<bool> normal_valid(pixel_num, false);
    for (int u = 0; u < H_SCANS; ++u) {
        for (int v = 0; v < W_COLS; ++v) {
            int idx = u * W_COLS + v;
            if (!pixel_valid[idx]) continue;

            Eigen::Vector3d P, P_left, P_down;
            if (!getPoint(u, v, P)) continue;
            if (!getPoint(u, v - 1, P_left)) continue;          // 左邻居 (列周期已处理)
            if (u + 1 >= H_SCANS) continue;                      // 下邻居超界
            if (!getPoint(u + 1, v, P_down)) continue;

            Eigen::Vector3d e1 = P_left - P;
            Eigen::Vector3d e2 = P_down - P;
            Eigen::Vector3d n = e1.cross(e2);
            double len = n.norm();
            if (len < 1e-6) continue;
            n /= len;
            // 朝向传感器 (O = 原点, 视线方向是 -P)
            if (n.dot(-P) < 0) n = -n;
            normals[idx] = n;
            normal_valid[idx] = true;
        }
    }

    // ---------- 3. BFS 分割 ----------
    int current_label = 0;
    const int dir_u[4] = {-1, 1, 0, 0};
    const int dir_v[4] = { 0, 0,-1, 1};

    for (int seed = 0; seed < pixel_num; ++seed) {
        if (!pixel_valid[seed]) continue;
        if (result.label_map[seed] != 0) continue;

        ++current_label;
        std::vector<int> current_cluster;
        std::deque<int> q;
        result.label_map[seed] = current_label;
        current_cluster.push_back(seed);
        q.push_back(seed);

        while (!q.empty()) {
            int cur_index = q.front(); q.pop_front();
            int u = cur_index / W_COLS;
            int v = cur_index % W_COLS;
            const auto& cur_px = range_image_[cur_index];
            double crange = cur_px.range;//当前点到雷达距离

            for (int k = 0; k < 4; ++k) {
                int nu = u + dir_u[k];
                int nv = wrapCol(v + dir_v[k]);
                if (nu < 0 || nu >= H_SCANS) continue;

                int n_idx = nu * W_COLS + nv;
                if (!pixel_valid[n_idx]) continue;
                if (result.label_map[n_idx] != 0) continue;

                const auto& n_px = range_image_[n_idx];
                double nrange = n_px.range;//选中点到雷达的距离

                // ---- 条件 1: 欧氏距离粗筛 (基于角分辨率自适应) ----
                // 同一 scan 相邻两束光在距离 r 处的自然点间距 ≈ r * alpha_rad
                // k<2 为垂直方向邻居, k>=2 为水平方向邻居, 二者角分辨率不同
                double avg_range = (crange + nrange) * 0.5;
                double angle_res = (k < 2) ? alpha_vert_rad_ : alpha_horiz_rad_;
                // 容差系数 2.5: 允许倾斜面使点间距最多扩大到自然间距的 2.5 倍
                // 同时保留 max_dist 作为近距离的绝对下限, 防止阈值过小
                double adaptive_max_dist = std::max(max_dist, avg_range * angle_res * 2.5);
                double dx = cur_px.x - n_px.x;
                double dy = cur_px.y - n_px.y;
                double dz = cur_px.z - n_px.z;
                double euc_dist = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (euc_dist > adaptive_max_dist) continue;

                // ---- 条件 2: β 角度判断 (论文公式) ----
                // β = atan2(d2 * sin α, d1 - d2 * cos α)
                double d1 = std::max(crange, nrange);
                double d2 = std::min(crange, nrange);
                double alpha = (k < 2) ? alpha_vert_rad_ : alpha_horiz_rad_;
                double denom = d1 - d2 * std::cos(alpha);
                if (std::abs(denom) < 1e-9) continue;
                double beta = std::atan2(d2 * std::sin(alpha), denom);
                if (beta < theta_rad) continue;

                // // ---- 条件 3: 法向量一致性 ----
                // if (normal_valid[cur_index] && normal_valid[n_idx]) {
                //     double cos_n = normals[cur_index].dot(normals[n_idx]);
                //     if (cos_n < normal_cos_thresh) continue;
                // }

                // ---- 接受 ----
                result.label_map[n_idx] = current_label;
                current_cluster.push_back(n_idx);
                q.push_back(n_idx);
            }
        }

        if ((int)current_cluster.size() > min_cluster_size) {
            result.clusters.push_back(std::move(current_cluster));
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

pcl::PointCloud<pcl::PointXYZ>::Ptr
RangeImageProcessor::generateOneClusterCloud(const SegmentationResult& result, int cluster_id) const {
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    if (cluster_id < 0 || cluster_id >= (int)result.clusters.size()) return cloud;
    const auto& indices = result.clusters[cluster_id];
    cloud->reserve(indices.size());
    for (int idx : indices) {
        if (idx < 0 || idx >= (int)range_image_.size()) continue;
        const auto& px = range_image_[idx];
        if (!px.valid) continue;
        cloud->push_back(pcl::PointXYZ(px.x, px.y, px.z));
    }
    cloud->width  = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

std::vector<VoxelKey>
RangeImageProcessor::getClusterOccupiedVoxels(const VoxelizedClusters& vc, int cluster_id) const {
    std::vector<VoxelKey> out;
    auto it = vc.cluster_to_subclusters.find(cluster_id);
    if (it == vc.cluster_to_subclusters.end()) return out;
    out.reserve(it->second.size());
    for (int sub_id : it->second) {
        if (sub_id < 0 || sub_id >= (int)vc.sub_clusters.size()) continue;
        out.push_back(vc.sub_clusters[sub_id].voxel_key);
    }
    return out;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> RangeImageProcessor::generateClusterClouds(const SegmentationResult& result) {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cloud_list;

    if (result.clusters.empty()) {
        return cloud_list;
    }
    for (size_t i = 0; i < result.clusters.size(); ++i) {
        const auto& cluster_indices = result.clusters[i];
        // 创建一个新的点云对象
        pcl::PointCloud<pcl::PointXYZ>::Ptr current_cluster(new pcl::PointCloud<pcl::PointXYZ>);

        // 预分配内存优化
        current_cluster->reserve(cluster_indices.size());

        // 遍历索引填充点
        int bool1 = false;
        int bool2 = false;
        int bool3 = false;
        for (int idx : cluster_indices) {
            const auto& px = range_image_[idx];
            current_cluster->push_back(pcl::PointXYZ(px.x, px.y, px.z));
            if (px.x > -4 && px.x < -2 && px.y > 11 && px.y < 12 ) {
                //std::cout<<"px.x "<<px.x<<" px.y "<<px.y<<std::endl;
                bool1 = true;
            }

            if (px.x > 0 && px.x < 2 && px.y > 11 && px.y < 12 ) {
                bool2 = true;
            }
            if (px.x > 2 && px.x < 4 && px.y > 11 && px.y < 12 ) {
                bool3 = true;
            }

        }
        if (bool1 && bool2 && bool3) {
            std::cout << "Saved ------" << i << std::endl;
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

// ============================================================================
// 体素化分割结果
// ============================================================================
VoxelizedClusters RangeImageProcessor::voxelizeClusters(const SegmentationResult& result,
                                                        double voxel_size,
                                                        int min_subcluster_size) const {
    VoxelizedClusters vc;
    vc.voxel_size = voxel_size;
    if (voxel_size <= 1e-6) return vc;
    const double inv_size = 1.0 / voxel_size;

    auto keyFromPoint = [&](double x, double y, double z) {
        VoxelKey k;
        k.x = static_cast<int>(std::floor(x * inv_size));
        k.y = static_cast<int>(std::floor(y * inv_size));
        k.z = static_cast<int>(std::floor(z * inv_size));
        return k;
    };

    // 先按 cluster 遍历, cluster 内按 voxel 分桶
    for (size_t cid = 0; cid < result.clusters.size(); ++cid) {
        const auto& cluster_indices = result.clusters[cid];
        if (cluster_indices.empty()) continue;

        std::unordered_map<VoxelKey, std::vector<int>, VoxelKeyHash> bucket;
        bucket.reserve(cluster_indices.size() / 4 + 1);

        for (int idx : cluster_indices) {
            const auto& px = range_image_[idx];
            if (!px.valid) continue;
            VoxelKey k = keyFromPoint(px.x, px.y, px.z);
            bucket[k].push_back(idx);
        }

        // 每个 (cluster_id, voxel_key) 组合产生一个 SubCluster
        for (auto& kv : bucket) {
            if ((int)kv.second.size() < min_subcluster_size) continue;

            SubCluster sc;
            //cid 聚类的id
            sc.cluster_id = static_cast<int>(cid);
            //kvfirst 体素id的key
            sc.voxel_key = kv.first;
            //体素包含的range image id'
            sc.indices = std::move(kv.second);

            int sub_idx = static_cast<int>(vc.sub_clusters.size());
            vc.sub_clusters.push_back(std::move(sc));
            vc.voxels[kv.first].sub_cluster_ids.push_back(sub_idx);
            vc.cluster_to_subclusters[static_cast<int>(cid)].push_back(sub_idx);
        }
    }

    std::cout << "[Voxelize] voxel_size=" << voxel_size
              << "  voxels=" << vc.voxels.size()
              << "  sub_clusters=" << vc.sub_clusters.size()
              << "  source_clusters=" << vc.cluster_to_subclusters.size()
              << std::endl;
    return vc;
}

void RangeImageProcessor::saveVoxelizedClustersToTxt(const VoxelizedClusters& vc,
                                                     const std::string& folder_path,
                                                     std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>& clouds) const {

    if (vc.sub_clusters.empty()) {
        std::cout << "No sub-clusters to save!" << std::endl;
        return;
    }
    int cnt = 0;
    for (const auto& sc : vc.sub_clusters) {
        auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        std::ostringstream fn;
        fn << folder_path << "/cluster_" << cnt << ".txt";
        cnt++;
        std::ofstream out(fn.str());
        if (!out.is_open()) {
            std::cerr << "Cannot open " << fn.str() << std::endl;
            continue;
        }
        out << std::fixed << std::setprecision(4);
        int bool1 = false;
        int bool2 = false;
        for (int idx : sc.indices) {
            const auto& px = range_image_[idx];
            out << px.x << " " << px.y << " " << px.z << "\n";
            cloud->push_back(pcl::PointXYZ(px.x, px.y, px.z));
            if (px.x > -9 && px.x < -8.5 && px.y > -6.5 && px.y < -6.25 ) {
                //std::cout<<"px.x "<<px.x<<" px.y "<<px.y<<std::endl;
                bool1 = true;
            }

            if (px.x > -8.75 && px.x < -8.5 && px.y > -7 && px.y < -6.6 ) {
                bool2 = true;
            }
        }
        if (bool1 && bool2) {
            std::cout << "Saved " << cnt << std::endl;
        }
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;
        clouds.push_back(cloud);
    }
    std::cout << "Saved " << vc.sub_clusters.size() << " sub-clusters to "
              << folder_path << std::endl;
}

std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr>
RangeImageProcessor::generateSubClusterClouds(const VoxelizedClusters& vc) const {
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds;
    clouds.reserve(vc.sub_clusters.size());
    for (const auto& sc : vc.sub_clusters) {
        auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
        cloud->reserve(sc.indices.size());
        for (int idx : sc.indices) {
            const auto& px = range_image_[idx];
            cloud->push_back(pcl::PointXYZ(px.x, px.y, px.z));
        }
        cloud->width = cloud->points.size();
        cloud->height = 1;
        cloud->is_dense = true;
        clouds.push_back(cloud);
    }
    return clouds;
}