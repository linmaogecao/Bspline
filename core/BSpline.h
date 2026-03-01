#pragma once


#include <Eigen/Core>
#include <vector>
#include <pcl/common/common.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/centroid.h>
#include <tool.h>
#include <map>


using namespace std;
using namespace Eigen;

class BSplineCurve
{
   public:
       typedef std::pair<int, double> Parameter;

       BSplineCurve(double interal=0.001):interal_(interal){

       }

       ~BSplineCurve(){
           clear();
       }

       size_t nb_control() const {return controls_.size();}

       /*
        * Compute the x,y position of current parameter
        */
       Vector2d getPos(const Parameter& para) const;


       /*
        * Compute the first differential
        */
       Vector2d getFirstDiff(const Parameter& para) const;


       /*
        * Compute the second differential
        */
       Vector2d getSecondDiff(const Parameter& para) const;

       /*
        * Compute the curvature of a given point
        */
       double getCurvature(const Parameter& para) const;


       /*
        * Compute the unit tangent vector
        */
       Vector2d getTangent(const Parameter &para) const;

       /*
        * Compute the unit Normal vector
        */
       Vector2d getNormal(const Parameter &para) const;


       /*
        * Compute the Curvature centor
        */
       Vector2d getCurvCenter(const Parameter &para) const;


       /*
        * Compute the foot print
        */
       double findFootPrint(const vector<Vector2d>& givepoints,vector<Parameter>& footPrints) const;


       /*
        * find the coff vector
        */
       VectorXd getCoffe(const Parameter&para) const ;

       /*
        * set the control points and compute a uniform spatial partition of the data points
        */
        void setNewControl(const vector<Vector2d>& controlPs);

        /*
         * Check if two points are on the same side
         */
        bool checkSameSide(Vector2d p1,Vector2d p2,Vector2d neip);


        MatrixXd getSIntegralSq();

        MatrixXd getFIntegralSq();

        const vector<Vector2d>& getControls() const{return controls_;}

        const vector<Vector2d>& getSamples() const{return positions_;}




   private:

    void clear(){
        controls_.clear();
        positions_.clear();
    }

    Parameter getPara(int index) const;


    bool checkInside(Vector2d p);

    /*
     * Test if a point is Left|On|Right of an infinite line
     */
    int isLeft(Vector2d p0,Vector2d p1, Vector2d p2);


private:

    double interal_;
    std::vector<Vector2d> controls_;
    std::vector<Vector2d> positions_;
};


struct SurfaceCurvature {
    Vector3d point;
    Vector3d normal;
    double E, F, G;
    double L, M, N;
    double K;     // Gaussian
    double H;     // Mean
    Vector3d tangent1; // 主方向 T1
    Vector3d tangent2; // 主方向 T2
    double k1, k2;
};

struct CurvatureCenters {
    Vector3d center1;
    Vector3d center2;
    bool is_planar;
};

class BSplineSurface {
public:
    typedef std::pair<int, double> Parameter;
    struct LocalRange {
        double min_x, max_x;
        double min_y, max_y;
        double min_z, max_z;
        bool has_data = false;
    };
    BSplineSurface(int deg_u,int deg_v,int control_num_u,int control_num_v,double interal=0.01):
        interal_u(interal),interal_v(interal),Deg_u(deg_u),Deg_v(deg_v),controls_num_u(control_num_u),controls_num_v(control_num_v)
    {
        knots_u.resize(deg_u+control_num_u+1);
        knots_v.resize(deg_v+control_num_v+1);
        for (int i = 0; i <= deg_u+control_num_u; i++) {
            if (i <= deg_u)
                knots_u[i] = 0;
            else if (i >= control_num_u)
                knots_u[i] = 1;
            else
                knots_u[i] = (double)(i-deg_u)/(double)(control_num_u-deg_u);
        }
        for (int i = 0; i <= deg_v+control_num_v; i++) {
            if (i <= deg_v)
                knots_v[i] = 0;
            else if (i >= control_num_v)
                knots_v[i] = 1;
            else
                knots_v[i] = (double)(i-deg_v)/(double)(control_num_v-deg_v);
        }
        interal_ = interal;
    }

    ~BSplineSurface(){
        clear();
    }

    // size_t nb_control_u() const {return controls_u.size();}
    // size_t nb_control_v() const {return controls_v.size();}

    /**
     *  calculate the position of current parameter
     *  ref to <<General Matrix Representations for B-Splines>> remap the tf into [ti,ti+1]
     * @param para
     * @param knots intervals
     * @param controls control points set
     * @return
     */
    Vector3d getPos(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v);
    Vector3d getFirstDiff(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v, bool is_diff_u);
    Vector3d getSecondDiff(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v, int type);
    SurfaceCurvature getCurvature(const Parameter& paraU, const Parameter& paraV, const vector<double>& knotsU, const vector<double>& knotsV, const std::vector<Vector3d>& controls,int num_cp_v);
    Vector3d getTangent(const Parameter& para, const vector<double> &knots,const std::vector<Vector3d> &controls);
    Vector3d getNormal(const Parameter& para, const vector<double> &knots,const std::vector<Vector3d> &controls);
    Vector3d getCurvCenter(const Parameter& para, const vector<double> &knots,const std::vector<Vector3d> &controls);
    double findFootPrint(const vector<Vector3d>& givepoints,vector<pair<Parameter, Parameter>>& footPrints);
    void initControlPoint(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,vector<Vector3d>& controlPs,int num_u,int num_v);
    void setNewControl(const vector<Vector3d> &controlPs, int num_u, int num_v,bool isCut = false);
    void setKnotParams(int num_cp_u,int num_cp_v);
    pair<Parameter, Parameter> getPara(int index);
    void pclToEigenVector(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, std::vector<Vector3d>& out_vec);
    double apply(pcl::PointCloud<pcl::PointXYZ>::Ptr& points,int maxIterNum,double alpha,double gama,double eplison);
    const vector<Vector3d>& getControls() const{return controls;}
    bool isPointValid(const Vector3d& p);
    const vector<Vector3d>& getSamples() const{return positions;}
    void buildRangeGrid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, int grid_res);
private:

    void clear(){
        controls.clear();
        positions.clear();
        sampling_paras_.clear();
    }

    /** follow <<General Matrix Representations for B-Splines>> calculate BSpline coeff matrix
     *
     * @param i in which param interal  [ti,ti+1]
     * @param knots knots intervals
     * @return
     */
    Eigen::Matrix4d ComputeNonUniformBsplineMatrix(int i, const vector<double>& knots);

private:
    double interal_u;
    double interal_v;
    std::vector<Vector3d> controls;
    std::vector<Vector3d> positions;
    std::vector<double> knots_u;
    std::vector<double> knots_v;
    int Deg_u;
    int Deg_v;
    int controls_num_u;
    int controls_num_v;
    double interal_;
    double max_x,max_y,max_z;
    double min_x,min_y,min_z;
    pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud_;
    pcl::KdTreeFLANN<pcl::PointXYZ> input_kdtree_;
    vector<pair<Parameter, Parameter>> sampling_paras_;
    std::vector<std::vector<LocalRange>> range_grid_;
    int grid_res_x_ = 0;
    int grid_res_y_ = 0;
    double grid_cell_size_x_ = 0;
    double grid_cell_size_y_ = 0;
    double grid_origin_x_ = 0;
    double grid_origin_y_ = 0;
    int cn1 = 0;
    int cn2 = 0;
    int cn3 = 0;
};
