//
// Created by 叶卓杨 on 2021/5/8.
//

#include "readWrite.h"

#include <fstream>
#include <iostream>


bool readWrite::readData(const string &filename, vector<Vector3d>& points) {
    std::ifstream fin(filename.c_str());
    std::vector<double> coefficients;
    std::string line;
    int nRow = 0;
    int nCol = 0;
    if( fin.fail() )
        return false;


    while( std::getline( fin, line ) )
    {
        std::stringstream ss(line);
        double d = 0;
        nCol = 0;
        while( ss >> d)
        {
            coefficients.push_back( d);
            ++nCol;
        }
        ++nRow;
    }
    fin.close();

    if( nCol < 3)
        return false;

    points.resize( nRow);
    int idx = 0;
    for( int i = 0; i!= nRow; ++i )
    {
        points[i] = Vector3d( coefficients[idx+0], coefficients[idx+1], coefficients[idx+2]);
        idx += nCol;
    }
    return true;
}


bool readWrite::writeDate(const string &filename, const vector<Vector3d>& points) {
    ofstream fout( filename.c_str());
    if( fout.fail() )
        return false;

    for( int i = 0; i!= points.size(); ++i )
    {
        fout << points[i].x() << " " << points[i].y() << " " << points[i].z()  << endl;
    }
    fout.close();

    return true;
}
