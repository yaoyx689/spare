#ifndef REGISTRATION_H_
#define REGISTRATION_H_
#include "tools/nanoflann.h"
#include "tools/tools.h"
#include "tools/Types.h"
#include "tools/OmpHelper.h"

class Registration
{
public:
    Registration();
    virtual ~Registration();

    Mesh* src_mesh_;
    Mesh* tar_mesh_;
    int n_src_vertex_;
    int n_tar_vertex_;
    int n_landmark_nodes_;


    struct Closest{
        int src_idx; // vertex index from source model
        int tar_idx; // face index from target model
        Vector3 position;
        Vector3 normal;
        Scalar  min_dist2;
    };
    typedef std::vector<Closest> VPairs;

protected:
    // non-rigid Energy function paras
    VectorX weight_d_;	                // robust weight for alignment "\alpha_i"
    RowMajorSparseMatrix mat_A0_;	    // symmetric coefficient matrix for linear equations
    Matrix3X tar_points_;               // target_mesh  (3,m);
    VectorX vec_b_;                     // rights for linear equations
    VectorX corres_U0_;                 // all correspondence points (3,n);

    KDtree* target_tree;                // correspondence paras

    // Rigid paras
    Affine3 rigid_T_;                   // rigid registration transform matrix

    // Check correspondence points
    VectorX corres_pair_ids_;
    VPairs correspondence_pairs_;
    int current_n_;

    // dynamic welsch parasmeters
    bool init_nu;
    Scalar end_nu;
    Scalar nu;



public:
    // adjusted paras
    RegParas pars_;

public:
    virtual Scalar DoNonRigid() { return 0.0; }
    Scalar DoRigid();
    void InitFromInput(Mesh& src_mesh, Mesh& tar_mesh, RegParas& paras);
    virtual void Initialize(){}

private:
    Eigen::VectorXi  init_geo_pairs;

protected:
    //point to point rigid registration
    template <typename Derived1, typename Derived2, typename Derived3>
    Affine3 point_to_point(Eigen::MatrixBase<Derived1>& X,
        Eigen::MatrixBase<Derived2>& Y, const Eigen::MatrixBase<Derived3>& w);

    // Find correct correspondences
    void InitCorrespondence(VPairs & corres);
    void FindClosestPoints(VPairs & corres);
	void FindClosestPoints(VPairs & corres, VectorX & deformed_v);
    void FindClosestPoints(VPairs & corres, VectorX & deformed_v, std::vector<size_t>& sample_indices);

    // Pruning method
    void SimplePruning(VPairs & corres, bool use_distance, bool use_normal);

    // Use landmark;
    void LandMarkCorres(VPairs & correspondence_pairs);

    
    template<typename Derived1>
    Scalar FindKnearestMed(Eigen::MatrixBase<Derived1>& X, int nk);
};
#endif
