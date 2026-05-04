//
// Created by albus on 2026/1/26.
//

#ifndef SPLINE_FITTING_BSPLINESDMERR_H
#define SPLINE_FITTING_BSPLINESDMERR_H
#include <ceres/ceres.h>
#include <Eigen/Core>
#include "BSpline.h"
class BSplineSDMErr : public ceres::CostFunction {
public:
    BSplineSDMErr(const Eigen::Vector3d& target_point,
                  const SurfaceCurvature& frame,
                  const std::vector<double>& basis_weights)
        : target_point_(target_point), frame_(frame), weights_(basis_weights)
    {
        // 1. 设置残差维度：3维 (T1方向, T2方向, N方向)
        set_num_residuals(3);

        // 2. 设置参数块维度：根据传入的权重数量决定有多少个控制点
        // 每个控制点都是 3维 (x,y,z)
        for (size_t i = 0; i < weights_.size(); ++i) {
            mutable_parameter_block_sizes()->push_back(3);
        }

        // 3. 预计算公式 (7) 的权重系数 (跟之前一样)
        double dist = (target_point - frame.point).dot(frame.normal);
        double eps = 1e-4;

        auto calc_coeff = [&](double rho) -> double {

            double denom = dist - rho;
            if (std::abs(denom) < eps) return 0.0;
            double val = dist / denom;
            if (val > 100.0) return 10.0;
            return (val > 0) ? std::sqrt(val) : 0.0;
        };

        coeff_t1_ = calc_coeff(1.0/frame.k1);
        coeff_t2_ = calc_coeff(1.0/frame.k2);
        coeff_n_  = 1.0;
    }

    // 核心计算函数
    virtual bool Evaluate(double const* const* parameters,
                          double* residuals,
                          double** jacobians) const override {

        size_t num_cps = weights_.size();

        // --- 1. 计算 P_plus (当前控制点下的曲面点) ---
        Vector3d P_plus = Vector3d::Zero();
        for (size_t i = 0; i < num_cps; ++i) {
            Vector3d cp(parameters[i][0], parameters[i][1], parameters[i][2]);
            P_plus += weights_[i] * cp;
        }
        // --- 2. 计算残差 ---
        Vector3d diff = P_plus - target_point_;

        // 投影到局部坐标系
        double x1 = diff.dot(frame_.tangent1);
        double x2 = diff.dot(frame_.tangent2);
        double x3 = diff.dot(frame_.normal);

        residuals[0] = coeff_t1_ * x1;
        residuals[1] = coeff_t2_ * x2;
        residuals[2] = coeff_n_  * x3;
        // --- 3. 计算雅可比 ---
        if (jacobians) {
            for (size_t i = 0; i < num_cps; ++i) {
                if (jacobians[i]) {
                    double w = weights_[i];
                    jacobians[i][0] = coeff_t1_ * frame_.tangent1.x() * w;
                    jacobians[i][1] = coeff_t1_ * frame_.tangent1.y() * w;
                    jacobians[i][2] = coeff_t1_ * frame_.tangent1.z() * w;

                    // Row 1: 对 Residual[1] (T2方向) 求导
                    jacobians[i][3] = coeff_t2_ * frame_.tangent2.x() * w;
                    jacobians[i][4] = coeff_t2_ * frame_.tangent2.y() * w;
                    jacobians[i][5] = coeff_t2_ * frame_.tangent2.z() * w;

                    // Row 2: 对 Residual[2] (N方向) 求导
                    jacobians[i][6] = coeff_n_ * frame_.normal.x() * w;
                    jacobians[i][7] = coeff_n_ * frame_.normal.y() * w;
                    jacobians[i][8] = coeff_n_ * frame_.normal.z() * w;
                }
            }
        }

        return true;
    }

private:
    Vector3d target_point_;
    SurfaceCurvature frame_;
    std::vector<double> weights_;
    double coeff_t1_, coeff_t2_, coeff_n_;
};

struct BSplineSmoothnessErr {
    BSplineSmoothnessErr(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const p_prev, // 前一个点 P(i-1)
                    const T* const p_curr, // 当前点   P(i)
                    const T* const p_next, // 后一个点 P(i+1)
                    T* residuals) const {
        // 公式： weight * (P_prev - 2*P_curr + P_next)
        // 也就是离散化的二阶导数
        residuals[0] = T(weight_) * (p_prev[0] - T(2.0) * p_curr[0] + p_next[0]);
        residuals[1] = T(weight_) * (p_prev[1] - T(2.0) * p_curr[1] + p_next[1]);
        residuals[2] = T(weight_) * (p_prev[2] - T(2.0) * p_curr[2] + p_next[2]);
        return true;
    }

    // 工厂函数：告诉 Ceres 这是一个 3维残差，输入是 3个 3维向量
    static ceres::CostFunction* Create(double weight) {
        return new ceres::AutoDiffCostFunction<BSplineSmoothnessErr, 3, 3, 3, 3>(
            new BSplineSmoothnessErr(weight));
    }

    double weight_;
};
struct BSplineFirstOrderErr {
    BSplineFirstOrderErr(double weight) : weight_(weight) {}

    template <typename T>
    bool operator()(const T* const p_curr, // 当前点 P(i)
                    const T* const p_next, // 下一点 P(i+1)
                    T* residuals) const {
        // 公式： weight * (P_next - P_curr)
        // 也就是离散化的一阶导数（差分）
        // 物理意义：最小化控制点之间的距离，产生“张力”
        residuals[0] = T(weight_) * (p_next[0] - p_curr[0]);
        residuals[1] = T(weight_) * (p_next[1] - p_curr[1]);
        residuals[2] = T(weight_) * (p_next[2] - p_curr[2]);
        return true;
    }

    // 工厂函数：输入是 2个 3维向量
    static ceres::CostFunction* Create(double weight) {
        return new ceres::AutoDiffCostFunction<BSplineFirstOrderErr, 3, 3, 3>(
            new BSplineFirstOrderErr(weight));
    }

    double weight_;
};

struct BoundaryPenalty {
    BoundaryPenalty(const Vector3d& min_p, const Vector3d& max_p, double w)
        : min_p_(min_p), max_p_(max_p), w_(w) {}

    template <typename T>
    bool operator()(const T* const cp, T* residual) const {
        for (int k = 0; k < 3; ++k) {
            T v = cp[k];
            T minv = T(min_p_[k]);
            T maxv = T(max_p_[k]);
            T r = T(0);
            if (v < minv) r = minv - v;
            else if (v > maxv) r = v - maxv;
            residual[k] = T(w_) * r;
        }
        return true;
    }

    Vector3d min_p_, max_p_;
    double w_;
};


#endif //SPLINE_FITTING_BSPLINESDMERR_H