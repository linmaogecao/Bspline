//
// Created by 叶卓杨 on 2021/5/8.
//

#include "core/BSpline.h"
#include "core/Spline_curve_fitting.h"
#include "readWrite.h"
#include <RangeMap.h>
#include <iostream>


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


    BSplineSurface surface(3,3,15,15,0.25);


    std::vector<Vector3d> points;
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
    SegmentationResult result = t1.segmentRangeImage(5,80,60,0.1,10);
    t1.saveClustersToTxt(result, "output_clusters");
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clouds = t1.generateClusterClouds(result);

    // 体素化: 1m × 1m × 1m, 每个子块至少 5 个点才保留
    VoxelizedClusters vc = t1.voxelizeClusters(result, 1.0, 5);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> sub_clouds;
    t1.saveVoxelizedClustersToTxt(vc, "output_voxels",sub_clouds);
    // std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> sub_clouds = t1.generateSubClusterClouds(vc);
    // for (auto cloud_single : clouds) {
    //     surface.apply(cloud_single, 50,1,1,0.5);
    //     readWrite::writeDate( outFileName1, surface.getControls(),true);
    //     readWrite::writeDate( outFileName2, surface.getSamples(),true );
    // }
    int indexx = std::stoi(index_num);
    surface.apply(sub_clouds[indexx], 50,1,1,0.05);
    readWrite::writeDate( outFileName1, surface.getControls(),true);
    readWrite::writeDate( outFileName2, surface.getSamples(),true );



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
