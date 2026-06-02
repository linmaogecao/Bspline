//
// Created by 叶卓杨 on 2021/5/8.
//

#include "core/BSpline.h"
#include "core/Spline_curve_fitting.h"
#include "readWrite.h"
#include <RangeMap.h>
#include <iostream>
#include <array>
#include <cmath>
#include <Eigen/Eigenvalues>


namespace {

// ------------------------------------------------------------------
// PCA + 2D 栅格化 + Moore-Neighbor tracing 提取有序边界
//
// 输入: 一个 cluster 的点云 (近似共面)
// 输出: 沿外轮廓顺序排列的 3D 点 (闭合)
//
// 参数:
//   cell_size : 2D 栅格分辨率, 例如 0.1 m
//   pad_cells : 栅格外围 padding (>=1, 保证最外圈是空, 边界一定存在邻居为空)
// ------------------------------------------------------------------
std::vector<Eigen::Vector3d> extractOrderedBoundary(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    double cell_size = 0.1,
    int pad_cells = 1)
{
    std::vector<Eigen::Vector3d> out;
    if (!cloud || cloud->points.size() < 3) return out;

    const int N = static_cast<int>(cloud->points.size());

    // ---------- 1. PCA: 得到平面坐标系 (u_axis, v_axis, n_axis) ----------
    Eigen::Matrix<double, 3, Eigen::Dynamic> P(3, N);
    for (int i = 0; i < N; ++i) {
        P(0, i) = cloud->points[i].x;
        P(1, i) = cloud->points[i].y;
        P(2, i) = cloud->points[i].z;
    }
    Eigen::Vector3d centroid = P.rowwise().mean();
    Eigen::Matrix<double, 3, Eigen::Dynamic> Q = P.colwise() - centroid;
    Eigen::Matrix3d cov = (Q * Q.transpose()) / static_cast<double>(N);

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(cov);
    // eigenvalues() 升序: [小, 中, 大]
    Eigen::Vector3d u_axis = es.eigenvectors().col(2).normalized(); // 最大方差
    Eigen::Vector3d v_axis = es.eigenvectors().col(1).normalized();
    Eigen::Vector3d n_axis = es.eigenvectors().col(0).normalized(); // 法向

    // ---------- 2. 投影到 (u, v) ----------
    std::vector<double> us(N), vs(N);
    double u_min =  std::numeric_limits<double>::infinity();
    double u_max = -std::numeric_limits<double>::infinity();
    double v_min =  std::numeric_limits<double>::infinity();
    double v_max = -std::numeric_limits<double>::infinity();
    double h_sum = 0.0;
    for (int i = 0; i < N; ++i) {
        Eigen::Vector3d d = Q.col(i);
        us[i] = d.dot(u_axis);
        vs[i] = d.dot(v_axis);
        h_sum += d.dot(n_axis);
        u_min = std::min(u_min, us[i]);
        u_max = std::max(u_max, us[i]);
        v_min = std::min(v_min, vs[i]);
        v_max = std::max(v_max, vs[i]);
    }
    const double h_avg = h_sum / static_cast<double>(N); // 平面到质心的偏移, 一般 ~0

    // ---------- 3. 栅格化 ----------
    const int W = static_cast<int>(std::ceil((u_max - u_min) / cell_size)) + 2 * pad_cells;
    const int H = static_cast<int>(std::ceil((v_max - v_min) / cell_size)) + 2 * pad_cells;
    if (W < 3 || H < 3) return out;

    std::vector<unsigned char> grid(static_cast<size_t>(W) * H, 0);
    auto idx_of = [&](int x, int y) { return static_cast<size_t>(y) * W + x; };
    auto cell_of = [&](double u, double v) {
        int cx = static_cast<int>(std::floor((u - u_min) / cell_size)) + pad_cells;
        int cy = static_cast<int>(std::floor((v - v_min) / cell_size)) + pad_cells;
        return std::pair<int,int>(cx, cy);
    };
    for (int i = 0; i < N; ++i) {
        auto [cx, cy] = cell_of(us[i], vs[i]);
        if (cx >= 0 && cx < W && cy >= 0 && cy < H) grid[idx_of(cx, cy)] = 1;
    }

    // ---------- 4. Moore-Neighbor tracing (Jacob 停止条件) ----------
    // 8 邻域顺时针: E, SE, S, SW, W, NW, N, NE
    const int dx[8] = { 1, 1, 0, -1, -1, -1,  0,  1};
    const int dy[8] = { 0, 1, 1,  1,  0, -1, -1, -1};

    // 起点: 自下而上、自左而右找第一个 occupied 格 (确保它一定在轮廓上)
    int sx = -1, sy = -1;
    for (int y = 0; y < H && sy < 0; ++y) {
        for (int x = 0; x < W; ++x) {
            if (grid[idx_of(x, y)]) { sx = x; sy = y; break; }
        }
    }
    if (sx < 0) return out;

    std::vector<std::pair<int,int>> contour;
    contour.reserve(static_cast<size_t>(2 * (W + H)));
    contour.emplace_back(sx, sy);

    // 起点的"进入方向" backtrack: 我们从西边过来 (因为是从下往上、左到右扫到的)
    int backtrack = 4; // W 方向
    int cur_x = sx, cur_y = sy;
    bool entered_start_again = false;

    const int max_iter = W * H * 8;
    for (int iter = 0; iter < max_iter; ++iter) {
        // 从 backtrack 的下一个方向 (顺时针) 开始扫 8 邻居
        int start_dir = (backtrack + 1) % 8;
        int found_dir = -1;
        for (int k = 0; k < 8; ++k) {
            int dir = (start_dir + k) % 8;
            int nx = cur_x + dx[dir];
            int ny = cur_y + dy[dir];
            if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
            if (grid[idx_of(nx, ny)]) {
                found_dir = dir;
                // 新 backtrack: 旧位置相对新位置的方向 = dir 的反向
                backtrack = (dir + 4) % 8;
                cur_x = nx;
                cur_y = ny;
                break;
            }
        }
        if (found_dir < 0) break; // 孤立像素

        // 简化的 Jacob 停止条件: 回到起点 (大多数 cluster 形状下足够鲁棒)
        if (cur_x == sx && cur_y == sy) {
            if (!entered_start_again) {
                entered_start_again = true;
                continue;
            } else {
                break;
            }
        }
        contour.emplace_back(cur_x, cur_y);
    }

    // ---------- 5. 边界格子中心 -> 3D ----------
    out.reserve(contour.size());
    for (const auto& [cx, cy] : contour) {
        double u_c = u_min + (cx - pad_cells + 0.5) * cell_size;
        double v_c = v_min + (cy - pad_cells + 0.5) * cell_size;
        Eigen::Vector3d p3 = centroid + u_c * u_axis + v_c * v_axis + h_avg * n_axis;
        out.push_back(p3);
    }
    return out;
}

} // namespace


/*
 * Main function for Spline curve fitting
 *
 */
int main(int argc, char *argv[]){

    char inpf[200],*input,*index_num;
    argc--;
    argv++;					//Skip program name arg

    if(argc<1)
    {
        cout<<"Input file:"<<endl;
        //cin>>inpf;
        strcpy(inpf,"01.txt");
        input = inpf;
    }
    else input    = argv[0];
    if (argc == 2)
        index_num = argv[1];
    string inFileName( input );
    string outFileName1 = "01_controls.txt";
    string outFileName2 = "01_spline.txt";
    string outFileName3 = "hull.txt";


    BSplineSurface surface(3,3,8,8,0.05);


    std::vector<Vector3d> points;
    std::vector<Vector3d> points2;
    //readWrite::readData( inFileName, points );

    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    Eigen::Matrix<double, 3, Eigen::Dynamic> tmp_point_expend;
    if (!readKitti("/home/albus/dataset/kitti/data_odometry_velodyne/dataset/sequences/", "00", 19, *cloud,tmp_point_expend))
    {
        std::cout << "No more PCD file!" << std::endl;
        return false;
    }
    // for (auto &p : points) {
    //     cloud->points.emplace_back(p.x(), p.y(), p.z());
    // }
    std::cout<<"read"<<cloud->size()<<endl;
    points.reserve(cloud->points.size());
     for (const auto& vec : cloud->points) {
         // Eigen(double) -> PCL(float) 会自动隐式转换
         points.emplace_back(vec.x, vec.y, vec.z);
     }
    readWrite::writeDate(inFileName, points,true);

    //surface.apply(cloud, 50,1,1,0.5);
    std::cout<<"apply"<<endl;
//	CReadWriteAsc::writeAsc( inFileName, points);
    //readWrite::writeDate( outFileName1, surface.getControls());
    //readWrite::writeDate( outFileName2, surface.getSamples() );

        // 1. 生成一些假数据用于测试 (或者替换为 pcl::io::loadPCDFile)

    // std::cout << "正在生成测试点云..." << std::endl;
    RangeImageProcessor t1;
    t1.generateRangeImage(cloud);
    SegmentationResult result = t1.segmentRangeImage(5,80,60,0.1,30);
    t1.saveClustersToTxt(result, "output_clusters");
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds = t1.generateClusterClouds(result);

    // 体素化: 1m × 1m × 1m, 每个子块至少 5 个点才保留
    VoxelizedClusters vc = t1.voxelizeClusters(result, 3.0, 5);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> sub_clouds;
    t1.saveVoxelizedClustersToTxt(vc, "output_voxels",sub_clouds);
    // std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> sub_clouds = t1.generateSubClusterClouds(vc);
    // for (auto cloud_single : clouds) {
    //     surface.apply(cloud_single, 50,1,1,0.5);
    //     readWrite::writeDate( outFileName1, surface.getControls(),true);
    //     readWrite::writeDate( outFileName2, surface.getSamples(),true );
    // }
    // auto total_start = std::chrono::high_resolution_clock::now();
    // for (int i_1 = 0; i_1 < (int)clouds.size(); ++i_1) {
    //     auto t0 = std::chrono::high_resolution_clock::now();
    //     int ref_num = int(sqrt(clouds[i_1]->size()));
    //     if (ref_num <8) ref_num -=2;
    //     else if (ref_num < 15) ref_num -= 3;
    //     else if (ref_num < 22) ref_num -= 5;
    //
    //     ref_num = ref_num < 5 ? 5 : ref_num;
    //     ref_num = ref_num > 20 ? 20 : ref_num;
    //     BSplineSurface surf_local(3, 3, ref_num, ref_num, 0.25);
    //     auto init_cp = t1.computeInitControlPoints(result, i_1,
    //                                                /*num_u=*/surf_local.controls_num_u, /*num_v=*/surf_local.controls_num_v,
    //                                                /*smooth_window=*/2);
    //     if (!init_cp.empty()) {
    //         surf_local.setExternalInitControls(init_cp);// new object each time, avoid state leak
    //         surf_local.apply(clouds[i_1], 50, 1, 1, 0.05);
    //         auto t1 = std::chrono::high_resolution_clock::now();
    //         double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    //         // std::cout << "cluster " << i_1 << " / " << clouds.size()
    //         //           << "  pts=" << clouds[i_1]->size()
    //         //           << "  time=" << ms << " ms" << std::endl;
    //         readWrite::writeDate("output_result/cluster_" + std::to_string(i_1) + ".txt",
    //                              surf_local.getSamples(), false);
    //     }
    // }
    // auto total_end = std::chrono::high_resolution_clock::now();
    // std::cout << "total: "
    //           << std::chrono::duration<double, std::milli>(total_end - total_start).count()
    //           << " ms" << std::endl;
    int indexx = atoi(index_num);

    // 用 range-image 行列结构初始化控制点，不依赖 PCA 平面投影，支持垂直墙/直角拐角
    {
        auto init_cp = t1.computeInitControlPoints(result, indexx,
                                                   /*num_u=*/surface.controls_num_u, /*num_v=*/surface.controls_num_v,
                                                   /*smooth_window=*/2);
        if (!init_cp.empty()) {
            surface.setExternalInitControls(init_cp);
            readWrite::writeDate("./../build/initial_control.txt", init_cp, false);
        }
        else {
            std::cerr << "警告: 未能提取到初始控制点!" << std::endl;
        }
    }
    surface.apply(clouds[indexx], 10,1,1,0.15);
    readWrite::writeDate(inFileName, clouds[indexx],false);
    readWrite::writeDate( outFileName1, surface.getControls(),false);
    readWrite::writeDate( outFileName2, surface.getSamples(),false );
    //
    //
    // points2 = t1.extractClusterBoundary3D(result, indexx, /*pad=*/1);
    // if (points2.empty()) {
    //     std::cerr << "警告: 未能提取到有序边界点!" << std::endl;
    // } else {
    //     std::cout << "提取到 " << points2.size() << " 个有序边界点" << std::endl;
    // }
    //
    // readWrite::writeDate(outFileName3, points2, false);
    //
    //
    // for (int k = 0; k < result.clusters.size(); k++) {
    //     points2 = t1.extractClusterBoundary3D(result, k, /*pad=*/3);
    //     readWrite::writeDate("output_hull/cluster_" + std::to_string(k) + ".txt", points2, false);
    // }
    // std::cout << "正在写入点云数据..." << std::endl;
    // std::vector<Vector3d> points_out2;
    // points_out2.resize(5000);
    //
    // for (auto p : t1.ouyt) {
    //     Vector3d vec(p.x,p.y,p.z);
    //     points_out2.push_back(vec);
    // }
    //
    // readWrite::writeDate( outFileName2, points_out2 );
        // 模拟一个圆柱体
        // for (int ring = 0; ring < 64; ++ring) {
        //     double angle_vert = -24.8 + ring * (26.8 / 63.0);
        //     double r = 10.0; // 半径 10米
        //     double z = r * std::sin(angle_vert * M_PI / 180.0);
        //     double r_xy = r * std::cos(angle_vert * M_PI / 180.0);
        //
        //     for (int i = 0; i < 1800; ++i) {
        //         double angle_hor = i * (360.0 / 1800.0) * M_PI / 180.0;
        //         pcl::PointXYZ pt;
        //         pt.x = r_xy * std::cos(angle_hor);
        //         pt.y = r_xy * std::sin(angle_hor);
        //         pt.z = z;
        //
        //         // 加一点噪声模拟真实雷达的“无序性”
        //         pt.x += (rand() % 100) / 10000.0;
        //
        //         cloud->points.push_back(pt);
        //     }
        // }
        // // 打乱顺序，模拟 KITTI 的无序性
        // std::random_shuffle(cloud->points.begin(), cloud->points.end());

    return 0;

}
