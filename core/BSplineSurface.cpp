//
// Created by albus on 2026/1/18.
//
#include <fstream>
#include <ANN/ANN.h>

#include "BSpline.h"
#include "BSplineSDMErr.h"
Vector3d BSplineSurface::getPos(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls, int num_cp_v) {
    double tf_u = paraU.second;
    int ki_u = paraU.first;
    Matrix4d matU = ComputeNonUniformBsplineMatrix(ki_u, knotsU);
    double dt_u = knotsU[ki_u+1] - knotsU[ki_u];
    double u = (tf_u - knotsU[ki_u]) / dt_u;
    Vector4d U_pos;

    U_pos << 1.0, u, u * u, u * u * u;
    RowVector4d weights_u = U_pos.transpose() * matU;

    double tf_v = paraV.second;
    int ki_v = paraV.first;
    Matrix4d matV = ComputeNonUniformBsplineMatrix(ki_v, knotsV);
    double dt_v = knotsV[ki_v+1] - knotsV[ki_v];
    double v = (tf_v - knotsV[ki_v]) / dt_v;
    Vector4d V_pos;

    V_pos << 1.0, v, v * v, v * v * v;
    RowVector4d weights_v = V_pos.transpose() * matV;


    Vector3d pos = Vector3d::Zero();
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int global_u_idx = ki_u - 3 + i;
            int global_v_idx = ki_v - 3 + j;

            int flat_index = global_u_idx * num_cp_v + global_v_idx;

            if (flat_index >= 0 && flat_index < controls.size()) {
                double weight = weights_u(i) * weights_v(j);
                pos += weight * controls[flat_index];
            }
        }
    }
    return pos;
}

Vector3d BSplineSurface::getFirstDiff(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v, bool is_diff_u)
{
    double tf_u = paraU.second;
    int ki_u = paraU.first;
    Matrix4d matU = ComputeNonUniformBsplineMatrix(ki_u, knotsU);
    double dt_u = knotsU[ki_u+1] - knotsU[ki_u];
    double u = (dt_u > 1e-9) ? (tf_u - knotsU[ki_u]) / dt_u : 0.0;

    Vector4d vec_u;
    RowVector4d weights_u;

    if (is_diff_u) {
        vec_u << 0.0, 1.0, 2.0 * u, 3.0 * u * u;
        double scale = (dt_u > 1e-9) ? (1.0 / dt_u) : 0.0;
        weights_u = (vec_u.transpose() * matU) * scale;
    } else {
        vec_u << 1.0, u, u * u, u * u * u;
        weights_u = vec_u.transpose() * matU;
    }

    double tf_v = paraV.second;
    int ki_v = paraV.first;
    Matrix4d matV = ComputeNonUniformBsplineMatrix(ki_v, knotsV);
    double dt_v = knotsV[ki_v+1] - knotsV[ki_v];
    double v = (dt_v > 1e-9) ? (tf_v - knotsV[ki_v]) / dt_v : 0.0;

    Vector4d vec_v;
    RowVector4d weights_v;

    if (!is_diff_u) {
        vec_v << 0.0, 1.0, 2.0 * v, 3.0 * v * v;
        double scale = (dt_v > 1e-9) ? (1.0 / dt_v) : 0.0;
        weights_v = (vec_v.transpose() * matV) * scale;
    } else {
        vec_v << 1.0, v, v * v, v * v * v;
        weights_v = vec_v.transpose() * matV;
    }
    Vector3d result = Vector3d::Zero();

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int u_idx_global = ki_u - 3 + i;
            int v_idx_global = ki_v - 3 + j;

            int flat_idx = u_idx_global * num_cp_v + v_idx_global;

            if (flat_idx >= 0 && flat_idx < controls.size()) {
                double w = weights_u(i) * weights_v(j);
                result += w * controls[flat_idx];
            }
        }
    }
    return result;
}

Vector3d BSplineSurface::getSecondDiff(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v, int type) {
    double tf_u = paraU.second;
    int ki_u = paraU.first;
    Matrix4d matU = ComputeNonUniformBsplineMatrix(ki_u, knotsU);
    double dt_u = knotsU[ki_u+1] - knotsU[ki_u];
    double u = (dt_u > 1e-9) ? (tf_u - knotsU[ki_u]) / dt_u : 0.0;

    Vector4d vec_u;
    double scale_u = 1.0;

    double tf_v = paraV.second;
    int ki_v = paraV.first;
    Matrix4d matV = ComputeNonUniformBsplineMatrix(ki_v, knotsV);
    double dt_v = knotsV[ki_v+1] - knotsV[ki_v];
    double v = (dt_v > 1e-9) ? (tf_v - knotsV[ki_v]) / dt_v : 0.0;

    Vector4d vec_v;
    double scale_v = 1.0;

    if (type == 0) {
        vec_u << 0.0, 0.0, 2.0, 6.0 * u;
        scale_u = (dt_u > 1e-9) ? (1.0 / (dt_u * dt_u)) : 0.0;
        vec_v << 1.0, v, v * v, v * v * v;
        scale_v = 1.0;

    } else if (type == 1) {
        vec_u << 1.0, u, u * u, u * u * u;
        scale_u = 1.0;
        vec_v << 0.0, 0.0, 2.0, 6.0 * v;
        scale_v = (dt_v > 1e-9) ? (1.0 / (dt_v * dt_v)) : 0.0;

    } else if (type == 2) {
        vec_u << 0.0, 1.0, 2.0 * u, 3.0 * u * u;
        scale_u = (dt_u > 1e-9) ? (1.0 / dt_u) : 0.0;
        vec_v << 0.0, 1.0, 2.0 * v, 3.0 * v * v;
        scale_v = (dt_v > 1e-9) ? (1.0 / dt_v) : 0.0;
    }

    RowVector4d weights_u = (vec_u.transpose() * matU) * scale_u;
    RowVector4d weights_v = (vec_v.transpose() * matV) * scale_v;

    Vector3d result = Vector3d::Zero();

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int u_idx_global = ki_u - 3 + i;
            int v_idx_global = ki_v - 3 + j;
            int flat_idx = u_idx_global * num_cp_v + v_idx_global;

            if (flat_idx >= 0 && flat_idx < controls.size()) {
                double w = weights_u(i) * weights_v(j);
                result += w * controls[flat_idx];
            }
        }
    }

    return result;
}


SurfaceCurvature BSplineSurface::getCurvature(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v) {

    SurfaceCurvature result;
    Vector3d Su = getFirstDiff(paraU, paraV, knotsU, knotsV, controls, num_cp_v, true);  // true for u
    Vector3d Sv = getFirstDiff(paraU, paraV, knotsU, knotsV, controls, num_cp_v, false); // false for v

    Vector3d Suu = getSecondDiff(paraU, paraV, knotsU, knotsV, controls, num_cp_v, 0); // type 0 = Suu
    Vector3d Svv = getSecondDiff(paraU, paraV, knotsU, knotsV, controls, num_cp_v, 1); // type 1 = Svv
    Vector3d Suv = getSecondDiff(paraU, paraV, knotsU, knotsV, controls, num_cp_v, 2); // type 2 = Suv

    Vector3d normal_raw = Su.cross(Sv);
    double area = normal_raw.norm();

    if (area < 1e-9) {
        result.normal = Vector3d::UnitZ();
        result.tangent1 = Vector3d::UnitX();
        result.tangent2 = Vector3d::UnitY();
        result.E=1; result.G=1; result.F=0;
        result.L=0; result.M=0; result.N=0;
        result.k1=0; result.k2=0; result.K=0; result.H=0;
        return result;
    }
    result.normal = normal_raw / area;

    result.E = Su.dot(Su);
    result.F = Su.dot(Sv);
    result.G = Sv.dot(Sv);

    result.L = Suu.dot(result.normal);
    result.M = Suv.dot(result.normal);
    result.N = Svv.dot(result.normal);

    double det_I = result.E * result.G - result.F * result.F; // EG - F^2
    if (std::abs(det_I) < 1e-9) {
        result.K = 0; result.H = 0; result.k1 = 0; result.k2 = 0;
        return result;
    }

    double det_II = result.L * result.N - result.M * result.M; // LN - M^2

    result.K = det_II / det_I; // Gaussian
    result.H = (result.E * result.N + result.G * result.L - 2 * result.F * result.M) / (2 * det_I); // Mean


    double discriminant = result.H * result.H - result.K;
    discriminant = (discriminant < 0) ? 0 : discriminant;
    double root_dis = std::sqrt(discriminant);

    result.k1 = result.H + root_dis;
    result.k2 = result.H - root_dis;

    double A = result.L - result.k1 * result.E;
    double B = result.M - result.k1 * result.F;

    Vector3d T1;
    if (std::abs(A) < 1e-8 && std::abs(B) < 1e-8) {
        T1 = Su.normalized();
    } else {
        T1 = (-B * Su + A * Sv).normalized();
    }

    // T2 垂直于 T1 和 N
    Vector3d T2 = result.normal.cross(T1).normalized();

    result.tangent1 = T1;
    result.tangent2 = T2;

    return result;
}

double BSplineSurface::findFootPrint(const vector<Vector3d> &givepoints, vector<pair<Parameter, Parameter>> &footPrints) {
    footPrints.clear();
    footPrints.resize( givepoints.size());

    int iKNei = 1;
    int iDim = 3;
    size_t iNPts = positions.size();
    double eps = 0;

    ANNpointArray dataPts = annAllocPts(iNPts, iDim); // allocate data points; // data points
    ANNpoint queryPt = annAllocPt(iDim);  // allocate query point

    ANNidxArray nnIdx = new ANNidx[iKNei]; // allocate near neigh indices
    ANNdistArray dists = new ANNdist[iKNei]; // allocate near neighbor dists

    for( int i = 0; i!= iNPts; ++i) {
        dataPts[i][0] = positions[i].x();
        dataPts[i][1] = positions[i].y();
        dataPts[i][2] = positions                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    [i].z();
    }
    ANNkd_tree* kdTree = new ANNkd_tree( // build search structure
            dataPts, // the data points
            iNPts, // number of points
            iDim);

    double squareSum = 0.0;
    for( int i = 0 ;i!= (int)givepoints.size(); ++i) {
        queryPt[0] = givepoints[i].x();
        queryPt[1] = givepoints[i].y();
        queryPt[2] = givepoints[i].z();
        kdTree->annkSearch( // search
                queryPt, // query point
                iKNei, // number of near neighbors
                nnIdx, // nearest neighbors (returned)
                dists, // distance (returned)
                eps); // error bound
        squareSum += dists[0];

        footPrints[i] =  getPara(nnIdx[0]) ;
    }

    delete[] nnIdx;
    delete[] dists;
    delete kdTree;
    annDeallocPts(dataPts);
    annClose(); // done with ANN

    return squareSum;

}

std::pair<BSplineSurface::Parameter, BSplineSurface::Parameter> BSplineSurface:: getPara(int index) {
    if (index < 0 || index >= sampling_paras_.size()) {
        std::cout<<"nnIdx[0]"<<index<<sampling_paras_.size()<<endl;
        return {Parameter(0, 0.0), Parameter(0, 0.0)};
    }
    return sampling_paras_[index];
}

void BSplineSurface::pclToEigenVector(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud,
    std::vector<Vector3d> &out_vec) {
    if (!cloud || cloud->empty()) {
        return;
    }
    out_vec.clear();
    out_vec.reserve(cloud->size());

    for (const auto& pt : cloud->points) {
        out_vec.emplace_back(pt.x, pt.y, pt.z);
    }
}

void BSplineSurface::initControlPoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, vector<Vector3d> &controlPs, int num_u,
                                      int num_v) {
    Eigen::Vector4f min_pt_4f, max_pt_4f;
    pcl::getMinMax3D(*cloud, min_pt_4f, max_pt_4f);
    Vector3d min_pt(min_pt_4f[0], min_pt_4f[1], min_pt_4f[2]);
    Vector3d max_pt(max_pt_4f[0], max_pt_4f[1], max_pt_4f[2]);
    max_x = max_pt(0); min_x = min_pt(0);
    max_y = max_pt(1); min_y = min_pt(1);
    max_z = max_pt(2); min_z = min_pt(2);
    std::cout <<max_x<<" "<<max_y<<" "<<max_z<<min_x<<" "<<min_y<<" "<<min_z<<endl;
    Vector3d range = max_pt - min_pt;
    Vector3d margin = range * 0.15;
    min_pt -= margin;
    max_pt += margin;
    range = max_pt - min_pt;
    int axis_u, axis_v, axis_h;
    // if (range.x() <= range.y() && range.x() <= range.z()) {
    //     axis_h = 0; axis_u = 1; axis_v = 2; // u=y, v=z
    // }
    // else if (range.y() <= range.x() && range.y() <= range.z()) {
    //     axis_h = 1; axis_u = 0; axis_v = 2; // u=x, v=z
    // }
    // else {
    //     axis_h = 2; axis_u = 0; axis_v = 1; // u=x, v=y
    // }
    axis_h = 2; axis_u = 0; axis_v = 1; // u=x, v=y
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    double u_step = range[axis_u] / (num_u - 1);
    double v_step = range[axis_v] / (num_v - 1);
    double h_query = min_pt[axis_h];
    for (int i = 0; i < num_u; ++i) {
        for (int j = 0; j < num_v; ++j) {
            double cur_u = min_pt[axis_u] + i * u_step;
            double cur_v = min_pt[axis_v] + j * v_step;

            Vector3d pos;
            pos(axis_u) = cur_u;
            pos(axis_v) = cur_v;
            pcl::PointXYZ searchPoint;
            searchPoint.x = (axis_u == 0) ? cur_u : ((axis_v == 0) ? cur_v : 0);
            searchPoint.y = (axis_u == 1) ? cur_u : ((axis_v == 1) ? cur_v : 0);
            searchPoint.z = (axis_u == 2) ? cur_u : ((axis_v == 2) ? cur_v : 0);
            if(axis_h == 0) searchPoint.x = h_query;
            else if(axis_h == 1) searchPoint.y = h_query;
            else searchPoint.z = h_query;
            std::vector<int> pointIdxNKNSearch(1);
            std::vector<float> pointNKNSquaredDistance(1);
            if (kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                // 找到了最近点，只取它的高度！
                pcl::PointXYZ nearest_pt = cloud->points[pointIdxNKNSearch[0]];
                if (axis_h == 0) pos(axis_h) = nearest_pt.x;
                else if (axis_h == 1) pos(axis_h) = nearest_pt.y;
                else pos(axis_h) = nearest_pt.z;
            } else {
                // 找不到就用平均值
                pos(axis_h) = h_query;
            }
            controlPs[i * num_v + j] = pos;
        }
    }
}
bool BSplineSurface::isPointValid(const Vector3d& p) {
    if (p.x()<(min_x-0.05) || p.x()>(max_x+0.05) || p.y()<(min_y-0.05) || p.y()>(max_y+0.05) || p.z()<(min_z-0.1) || p.z()>(max_z+0.1))
        return false;
    return true;
}

void BSplineSurface::setNewControl(const vector<Vector3d> &controlPs, int num_u, int num_v) {
    clear();
    controls = controlPs;
    controls_num_u = num_u;
    controls_num_v = num_v;

    int start_u = 3;
    int end_u   = controls_num_u ;

    int start_v = 3;
    int end_v   = controls_num_v ;
    for (int i = start_u; i <= end_u; ++i) {
        double dt_u = knots_u[i + 1] - knots_u[i];
        if (dt_u <= 1e-6) continue;
        for (int j = start_v; j <= end_v; ++j) {
            double dt_v = knots_v[j + 1] - knots_v[j];
            if (dt_v <= 1e-6) continue;

            for (double fu = 0.0; fu <= 1.0; fu += interal_) {
                for (double fv = 0.0; fv <= 1.0; fv += interal_) {
                    double global_u = knots_u[i] + fu * dt_u;
                    double global_v = knots_v[j] + fv * dt_v;

                    Parameter paraU(i, global_u);
                    Parameter paraV(j, global_v);

                    Vector3d p = getPos(paraU, paraV, knots_u, knots_v, controls, controls_num_v);
                    if (isPointValid(p))
                        {
                            positions.push_back(p);
                            sampling_paras_.push_back(std::make_pair(paraU, paraV));
                        }

                }
            }
        }
    }

}

void BSplineSurface::setKnotParams(int num_cp_u,int num_cp_v) {
    knots_u.resize(num_cp_u + 4);
    knots_v.resize(num_cp_v + 4);
    double denom_u = (double)(num_cp_u - 3);
    for (int i = 0; i < knots_u.size(); ++i) {
        if (i <= 3) {
            knots_u[i] = 0.0;
        }
        else if (i >= num_cp_u) {
            knots_u[i] = 1.0;
        }
        else {
            knots_u[i] = double(i - 3) / denom_u;
        }
    }
    double denom_v = (double)(num_cp_v - 3);

    for (int i = 0; i < knots_v.size(); ++i) {
        if (i <= 3) {
            knots_v[i] = 0.0;
        }
        else if (i >= num_cp_v) {
            knots_v[i] = 1.0;
        }
        else {
            knots_v[i] = double(i - 3) / denom_v;
        }
    }

}


Eigen::Matrix4d BSplineSurface::ComputeNonUniformBsplineMatrix(int i, const vector<double> &knots) {
    Eigen::Matrix4d M = Eigen::Matrix4d::Zero();
    double t_i   = knots[i];
    double t_im1 = knots[i - 1]; // i-1
    double t_im2 = knots[i - 2]; // i-2
    double t_ip1 = knots[i + 1]; // i+1
    double t_ip2 = knots[i + 2]; // i+2
    double t_ip3 = knots[i + 3]; // i+3

    double dt_i_im1   = t_i - t_im1;
    double dt_ip1_i   = t_ip1 - t_i;
    double dt_ip1_im1 = t_ip1 - t_im1;
    double dt_ip1_im2 = t_ip1 - t_im2;
    double dt_ip2_im1 = t_ip2 - t_im1;
    double dt_ip2_i   = t_ip2 - t_i;
    double dt_ip3_i   = t_ip3 - t_i;

    double m00 = (dt_ip1_i * dt_ip1_i) / (dt_ip1_im1 * dt_ip1_im2);
    double m02 = (dt_i_im1 * dt_i_im1) / (dt_ip2_im1 * dt_ip1_im1);
    double m22 = 3.0 * (dt_ip1_i * dt_ip1_i) / (dt_ip2_im1 * dt_ip1_im1);
    double m33 = (dt_ip1_i * dt_ip1_i) / (dt_ip3_i * dt_ip2_i);
    double m12 = 3.0 * dt_ip1_i * dt_i_im1 / (dt_ip2_im1 * dt_ip1_im1);

    // Row 0
    M(0, 0) = m00;
    M(0, 1) = 1.0 - m00 - m02; // m01 = 1 - m00 - m02
    M(0, 2) = m02;
    M(0, 3) = 0.0;

    // Row 1
    M(1, 0) = -3.0 * m00;
    M(1, 1) = 3.0 * m00 - m12;
    M(1, 2) = m12;
    M(1, 3) = 0.0;

    // Row 2
    M(2, 0) = 3.0 * m00;
    M(2, 1) = -3.0 * m00 - m22;
    M(2, 2) = m22;
    M(2, 3) = 0.0;

    // Row 3
    M(3, 0) = -m00;
    double term_extra = (dt_ip1_i * dt_ip1_i) / (dt_ip2_i * dt_ip2_im1);
    M(3, 2) = -m22 / 3.0 - m33 - term_extra;
    M(3, 1) = m00 - M(3, 2) - m33; // m31 = m00 - m32 - m33
    M(3, 3) = m33;

    return M;
}


double BSplineSurface::apply(
        pcl::PointCloud<pcl::PointXYZ>::Ptr& points,
        int maxIterNum,
        double alpha,
        double gama,
        double eplison )
{

    this->input_cloud_ = points;
    this->input_kdtree_.setInputCloud(points);
    vector<Vector3d> controlPs;
    controlPs.resize(controls_num_u * controls_num_v);

    initControlPoint(points, controlPs,controls_num_u, controls_num_v);
    setNewControl(controlPs,controls_num_u,controls_num_v);
    setKnotParams(controls_num_u, controls_num_v);
    // update the control point
    // compute P"(t)
    // MatrixXd pm = spline_surface->getSIntegralSq();
    // MatrixXd sm = spline_surface->getFIntegralSq();
    // end test

    // find the foot print, will result in error
    double total_error = 1e9;
    double last_error = 1e9;
    vector<Vector3d> givepoints;
    pclToEigenVector(points, givepoints);
    for(int iter = 0; iter < maxIterNum; ++iter) {
        ceres::Problem problem;
        vector<pair<Parameter, Parameter>> parameters;
        double current_sq_dist = findFootPrint(givepoints, parameters);
        std::cout<<last_error<<" "<<current_sq_dist<<endl;
        if (std::abs((last_error - current_sq_dist) < eplison && current_sq_dist <=2) || current_sq_dist <=0.5) {
            std::cout << "Converged at iter " << iter << std::endl;
            break;
        }
        last_error = current_sq_dist;
        for( int i = 0; i< parameters.size(); i++)
        {
            Parameter paraU = parameters[i].first, paraV = parameters[i].second;
            SurfaceCurvature surf_info = getCurvature(paraU, paraV, knots_u, knots_v, controls, controls_num_v);

            surf_info.point = getPos(paraU, paraV, knots_u, knots_v,controls,controls_num_v);
            int span_u = paraU.first;
            int span_v = paraV.first;
            std::vector<double> active_weights;
            std::vector<double*> active_cp_pointers;
            Matrix4d mat_coeff_u = ComputeNonUniformBsplineMatrix(span_u, knots_u);
            Matrix4d mat_coeff_v = ComputeNonUniformBsplineMatrix(span_v, knots_v);
            double dt_u = knots_u[span_u + 1] - knots_u[span_u];
            double u = (dt_u > 1e-9) ? (paraU.second - knots_u[span_u]) / dt_u : 0.0;
            Vector4d U_vec;
            U_vec << 1.0, u, u * u, u * u * u;
            RowVector4d w_u = U_vec.transpose() * mat_coeff_u;
            double dt_v = knots_v[span_v + 1] - knots_v[span_v];
            double v = (dt_v > 1e-9) ? (paraV.second - knots_v[span_v]) / dt_v : 0.0;
            Vector4d V_vec;
            V_vec << 1.0, v, v * v, v * v * v;
            RowVector4d w_v = V_vec.transpose() * mat_coeff_v;
            for (int l = 0; l < 4; ++l) {
                for (int m = 0; m < 4; ++m)
                {
                    int flat_index = (span_u -3 + l) * controls_num_v + span_v - 3 + m;
                    if (flat_index < 0 || flat_index >= controls.size()) {
                        // 如果越界，说明 u,v 算错了，跳过这个无效项，保命要紧
                        continue;
                    }
                    active_cp_pointers.push_back(controls[flat_index].data());

                    active_weights.push_back(w_u(l)*w_v(m));

                }
            }
            ceres::CostFunction* cost_func = new BSplineSDMErr(givepoints[i], surf_info, active_weights);

            problem.AddResidualBlock(cost_func, nullptr, active_cp_pointers);
            // 1. U 方向平滑 (行约束)
            // 遍历每一行，对中间的点加约束

        }
        double smooth_weight = 0.03;
        for (int i = 0; i < controls_num_u; ++i) {
            for (int j = 1; j < controls_num_v - 1; ++j) {
                // 获取连续三个点的索引
                int idx_prev = i * controls_num_v + (j - 1);
                int idx_curr = i * controls_num_v + j;
                int idx_next = i * controls_num_v + (j + 1);

                problem.AddResidualBlock(
                    BSplineSmoothnessErr::Create(smooth_weight),
                    nullptr, // 核函数 (nullptr 表示不开鲁棒核)
                    controls[idx_prev].data(), // P(i, j-1)
                    controls[idx_curr].data(), // P(i, j)
                    controls[idx_next].data()  // P(i, j+1)
                );
            }
        }

        // 2. V 方向平滑 (列约束)
        // 遍历每一列，对中间的点加约束
        for (int j = 0; j < controls_num_v; ++j) {
            for (int i = 1; i < controls_num_u - 1; ++i) {
                // 获取连续三个点的索引 (跨行取点)
                int idx_prev = (i - 1) * controls_num_v + j;
                int idx_curr = i * controls_num_v + j;
                int idx_next = (i + 1) * controls_num_v + j;

                problem.AddResidualBlock(
                    BSplineSmoothnessErr::Create(smooth_weight),
                    nullptr,
                    controls[idx_prev].data(), // P(i-1, j)
                    controls[idx_curr].data(), // P(i, j)
                    controls[idx_next].data()  // P(i+1, j)
                );
            }
        }



        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 5; // 关键点！
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.BriefReport() << std::endl;
        vector<Vector3d> controls_copy = controls; // <--- ✅ 先克隆一份
        setNewControl(controls_copy, controls_num_u, controls_num_v);
    }

    // 在 apply 函数的 return last_error; 之前加入：


    return 1.0;
}
