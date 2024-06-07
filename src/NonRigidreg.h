#ifndef QN_WELSCH_H_
#define QN_WELSCH_H_
#include "Registration.h"
#include "tools/nodeSampler.h"


#ifdef USE_PARDISO
#include <Eigen/PardisoSupport>
#endif
// #define DEBUG


class NonRigidreg : public Registration
{
public:
    NonRigidreg();
    ~NonRigidreg();
    virtual Scalar DoNonRigid();
    virtual void Initialize();

private:
    void welsch_weight(VectorX& r, Scalar p);

    void   CalcNodeRotations();
    Scalar SetMeshPoints(Mesh* mesh, const VectorX & target);

    #ifdef USE_PARDISO
    Eigen::PardisoLDLT<RowMajorSparseMatrix, Eigen::Lower> solver_;
    #else
    Eigen::SimplicialLDLT<RowMajorSparseMatrix, Eigen::Lower> solver_;
    #endif

private:
	// Sample paras
    int		num_sample_nodes;               // (r,) the number of sample nodes
    int     num_graph_edges;
    int     num_edges;

	// simplily nodes storage structure
    svr::nodeSampler src_sample_nodes;

	// variable
    VectorX		            X_;	                        // (12r,) transformations of sample nodes

    // alignment term
	VectorX		            nodes_P_;                   // (3n,) all sample nodes' coordinates
    RowMajorSparseMatrix    align_coeff_PV0_;           // (3n, 12r) coefficient matrix F 
    // smooth term
    RowMajorSparseMatrix	reg_coeff_B_;	            // (6|E_G|, 12r) the smooth between nodes, |E_G| is the edges' number of sample node graph;
    VectorX		            reg_right_D_;	            // (6|E_G|,) the different coordinate between xi and xj
    VectorX			        reg_cwise_weights_;         // (6|E_G|) the smooth weight;
    // rotation matrix term
    VectorX		            nodes_R_;	                // (9r,)  "proj(A)"
    RowMajorSparseMatrix	rigid_coeff_L_;	            // (12r, 12r) "H"
    RowMajorSparseMatrix	rigid_coeff_J_;	            // (9r, 12r)  "Y"
    VectorX		            diff_UP_;                   // aux matrix 
	// ARAP term (coarse)
	VectorX                 arap_laplace_weights_;      // (6E,) 
	Matrix3X                local_rotations_;           // (3n,3) "R"
	RowMajorSparseMatrix    arap_coeff_;                // (6E, 12r) "B"
    RowMajorSparseMatrix    arap_coeff_mul_;            // (12r, 12r) "B^T*B"
	VectorX                 arap_right_;                // (6E,) "L"
    // ARAP term (fine) 
	RowMajorSparseMatrix    arap_coeff_fine_;           // (6E, 3n) "B"
    RowMajorSparseMatrix    arap_coeff_mul_fine_;       // (3n, 3n) "B^T*B"
	VectorX                 arap_right_fine_;           // (6E,)  "Y"

    // point clouds 
	Matrix3X                src_points_;                // (3,n)
	Matrix3X                src_normals_;               // (3,n)
	Matrix3X                deformed_normals_;          // (3,n)
	VectorX                 deformed_points_;           // (3n,)
	Matrix3X                target_normals_;            // (3,n)
	RowMajorSparseMatrix    normals_sum_;               // (n,3n) "N" for alignment term 

    // sampling points & vertices relation matrix
    std::vector<size_t>     sampling_indices_;
    std::vector<int>        vertex_sample_indices_;

    // knn-neighbor indices for source points if no faces. 
    Eigen::MatrixXi         src_knn_indices_;
    // bool                    src_has_faces_; 
    int                     align_sampling_num_ = 3000;
	
    // weights of terms during the optimization process 
    Scalar          w_align;      
    Scalar          w_smo;        
    Scalar          optimize_w_align;  
    Scalar          optimize_w_smo;
    Scalar          optimize_w_rot;
	Scalar			optimize_w_arap;

    void InitWelschParam();
	void FullInARAPCoeff();

	void CalcARAPRight();
	void CalcARAPRightFine();

	void InitRotations();
    void CalcLocalRotations(bool isCoarseAlign);
	void CalcDeformedNormals();
	void InitNormalsSum();
    void CalcNormalsSum();

	void PointwiseFineReg(Scalar nu1);
    void GraphCoarseReg(Scalar nu1);

	Scalar CalcEnergy(Scalar& E_align, Scalar& E_reg,
		Scalar& E_rot, Scalar& E_arap, VectorX & reg_weights);

	Scalar CalcEnergyFine(Scalar& E_align, Scalar& E_arap);
    // Aux_tool function
    Scalar CalcEdgelength(Mesh* mesh, int type);

	
};
#endif
