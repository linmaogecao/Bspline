//
// Created by 叶卓杨 on 2021/5/8.
//

#include "core/BSpline.h"
#include "core/Spline_curve_fitting.h"
#include "readWrite.h"

#include <iostream>


/*
 * Main function for Spline curve fitting
 *
 */
int main(int argc, char *argv[]){

    char inpf[200],*input;
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

    string inFileName( input );
    string outFileName1 = "01_controls.txt";
    string outFileName2 = "01_spline.txt";


    BSplineSurface surface(3,3,9,9,0.01);


    std::vector<Vector3d> points;
    readWrite::readData( inFileName, points );
    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();

    cloud->points.reserve(points.size());
    for (const auto& vec : points) {
        // Eigen(double) -> PCL(float) 会自动隐式转换
        cloud->points.emplace_back(vec.x(), vec.y(), vec.z());
    }

    surface.apply(cloud, 50,1,1,0.5);
    std::cout<<"apply"<<endl;
//	CReadWriteAsc::writeAsc( inFileName, points);
    readWrite::writeDate( outFileName1, surface.getControls());
    readWrite::writeDate( outFileName2, surface.getSamples() );



}
