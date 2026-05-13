//
// Created by 叶卓杨 on 2021/5/8.
//

#pragma once


#include <string>
#include <vector>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;



class readWrite {
public:
    static bool readData(const string& filename, vector<Vector3d>& points);


    static bool writeDate(const string& filename,const vector<Vector3d>& points, bool append);



};
