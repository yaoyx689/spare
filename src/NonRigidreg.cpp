#pragma once
#include "NonRigidreg.h"
#include "median.h"
#include "tools/nanoflann.h"

// #define DEBUG

NonRigidreg::NonRigidreg() {
};

NonRigidreg::~NonRigidreg()
{
}

void NonRigidreg::Initialize()
{
	Timer timer;
	
    InitWelschParam();
	
	src_points_.resize(3, n_src_vertex_);
	src_normals_.resize(3, n_src_vertex_);
	corres_U0_.resize(3* n_src_vertex_);

	#pragma omp parallel for
	for (int i = 0; i < n_src_vertex_; i++)
	{
		Vec3 p = src_mesh_->point(src_mesh_->vertex_handle(i));
		src_points_(0, i) = p[0];
		src_points_(1, i) = p[1];
		src_points_(2, i) = p[2];
		Vec3 n = src_mesh_->normal(src_mesh_->vertex_handle(i));
		src_normals_(0, i) = n[0];
		src_normals_(1, i) = n[1];
		src_normals_(2, i) = n[2];
	}
	deformed_normals_ = src_normals_;
	deformed_points_ = Eigen::Map<VectorX>(src_points_.data(), 3*n_src_vertex_);

	int knn_num_neighbor = 6;
	if(!pars_.use_geodesic_dist)
	{
		src_knn_indices_.resize(knn_num_neighbor, n_src_vertex_);
		KDtree* src_tree = new KDtree(src_points_);
		#pragma omp parallel for
		for(int i = 0; i < n_src_vertex_; i++)
		{
			int* out_indices = new int[knn_num_neighbor+1];
        	Scalar *out_dists = new Scalar[knn_num_neighbor+1];
			src_tree->query(src_points_.col(i).data(), knn_num_neighbor+1, out_indices, out_dists);
			for(int j = 0; j < knn_num_neighbor; j++)
			{
				src_knn_indices_(j, i) = out_indices[j+1];
			}
			delete[] out_indices;
			delete[] out_dists;
		}
		delete src_tree;
	}

    Timer::EventID time1, time2;
    time1 = timer.get_time();
	Scalar sample_radius;
	if(pars_.use_coarse_reg)
	{
		if(pars_.use_geodesic_dist)
			sample_radius = src_sample_nodes.SampleAndConstuct(*src_mesh_, pars_.uni_sample_radio,  src_points_); 
		else
			sample_radius = src_sample_nodes.SampleAndConstuctFPS(*src_mesh_, pars_.uni_sample_radio, src_points_, src_knn_indices_, 4, 8);
	}

    time2 = timer.get_time();
	#ifdef DEBUG
	std::cout << "construct deformation graph time = " << timer.elapsed_time(time1, time2) << std::endl;
	#endif

	#ifdef DEBUG
	if(pars_.use_coarse_reg)
	{
        std::string out_node = "test.obj"; // pars_.out_each_step_info;
        src_sample_nodes.print_nodes(*src_mesh_, out_node);//init sample nodes
	}
    #endif

	if(pars_.use_coarse_reg)
	{
		num_sample_nodes = src_sample_nodes.nodeSize();
		pars_.num_sample_nodes = num_sample_nodes;

		X_.resize(12 * num_sample_nodes); X_.setZero();
		align_coeff_PV0_.resize(3 * n_src_vertex_, 12 * num_sample_nodes);
		nodes_P_.resize(n_src_vertex_ * 3);

		nodes_R_.resize(9 * num_sample_nodes); nodes_R_.setZero();
		rigid_coeff_L_.resize(12 * num_sample_nodes, 12 * num_sample_nodes);
		rigid_coeff_J_.resize(12 * num_sample_nodes, 9 * num_sample_nodes);

		std::vector<Triplet> coeffv(4 * num_sample_nodes);
		std::vector<Triplet> coeffL(9 * num_sample_nodes);
		std::vector<Triplet> coeffJ(9 * num_sample_nodes);
		for (int i = 0; i < num_sample_nodes; i++)
		{
			// X_
			X_[12 * i] = 1.0;
			X_[12 * i + 4] = 1.0;
			X_[12 * i + 8] = 1.0;

			// nodes_R_
			nodes_R_[9 * i] = 1.0;
			nodes_R_[9 * i + 4] = 1.0;
			nodes_R_[9 * i + 8] = 1.0;

			for (int j = 0; j < 9; j++)
			{
				// rigid_coeff_L_
				coeffL[9 * i + j] = Triplet(12 * i + j, 12 * i + j, 1.0);
				// rigid_coeff_J_
				coeffJ[9 * i + j] = Triplet(12 * i + j, 9 * i + j, 1.0);
			}
		}
		rigid_coeff_L_.setFromTriplets(coeffL.begin(), coeffL.end());
		rigid_coeff_J_.setFromTriplets(coeffJ.begin(), coeffJ.end());
		
		// 0.02s
		// update coefficient matrices
		src_sample_nodes.initWeight(align_coeff_PV0_, nodes_P_, 
			reg_coeff_B_, reg_right_D_, reg_cwise_weights_);
		
		num_graph_edges = reg_cwise_weights_.rows();
	
	}
	
	// update ARAP coeffs
	FullInARAPCoeff();

	local_rotations_.resize(3, n_src_vertex_ * 3);

	if(pars_.use_geodesic_dist)
	{
		num_edges = src_mesh_->n_halfedges();
		arap_right_.resize(3*src_mesh_->n_halfedges());
		arap_right_fine_.resize(3 * src_mesh_->n_halfedges());
	}
	else{
		num_edges = knn_num_neighbor*n_src_vertex_;
		arap_right_.resize(3*knn_num_neighbor*n_src_vertex_);
		arap_right_fine_.resize(3 * knn_num_neighbor*n_src_vertex_);
	}
	

	
	InitRotations();

	target_normals_.resize(3, n_tar_vertex_);
	for (int i = 0; i < n_tar_vertex_; i++)
	{
		Vec3 n = tar_mesh_->normal(tar_mesh_->vertex_handle(i));
		target_normals_(0, i) = n[0];
		target_normals_(1, i) = n[1];
		target_normals_(2, i) = n[2];
	}	

	
	Timer::EventID begin_sampling, end_sampling;
    begin_sampling = timer.get_time();

	sampling_indices_.clear();

    // start points
    size_t startIndex = 0;
    sampling_indices_.push_back(startIndex);


	// FPS to get sampling points in align term
	VectorX minDistances(n_src_vertex_);
	minDistances.setConstant(std::numeric_limits<Scalar>::max());
	minDistances[startIndex] = 0;

	vertex_sample_indices_.resize(n_src_vertex_, -1);
	vertex_sample_indices_[startIndex] = 0;
	
    // repeat select farthest points
    while (sampling_indices_.size() < align_sampling_num_) {
        // calculate the distance between each point with the sampling points set.         
		#pragma omp parallel for
        for (size_t i = 0; i < n_src_vertex_; ++i) {
			if(i==startIndex)
				continue;
			
			Scalar dist = (src_points_.col(startIndex) - src_points_.col(i)).norm();
			if(dist < minDistances[i])
				minDistances[i] = dist;
        }

        // choose farthest point
		int maxDistanceIndex;
		minDistances.maxCoeff(&maxDistanceIndex);
		minDistances[maxDistanceIndex] = 0;

        // add the farthest point into the sampling points set.
        sampling_indices_.push_back(maxDistanceIndex);
		startIndex= maxDistanceIndex;
		vertex_sample_indices_[startIndex] = sampling_indices_.size()-1;
    }
	
	end_sampling = timer.get_time();
	#ifdef DEBUG
	std::cout << "cur_sample_idx = " << sampling_indices_.size() << " time = " << timer.elapsed_time(begin_sampling, end_sampling) << std::endl;
	#endif
}

void NonRigidreg::InitWelschParam()
{
    // welsch parameters
    weight_d_.resize(n_src_vertex_*3);
	weight_d_.setOnes();

    // Initialize correspondences
    InitCorrespondence(correspondence_pairs_);

    VectorX init_nus(correspondence_pairs_.size());
	#pragma omp parallel for
    for(size_t i = 0; i < correspondence_pairs_.size(); i++)
    {
        Vector3 closet = correspondence_pairs_[i].position;
        init_nus[i] = (src_mesh_->point(src_mesh_->vertex_handle(correspondence_pairs_[i].src_idx))
                    - Vec3(closet[0], closet[1], closet[2])).norm();
    }
    igl::median(init_nus, pars_.Data_nu);

    if(pars_.calc_gt_err&&n_src_vertex_ == n_tar_vertex_)
    {
        VectorX gt_err(n_src_vertex_);
        for(int i = 0; i < n_src_vertex_; i++)
        {
            gt_err[i] = (src_mesh_->point(src_mesh_->vertex_handle(i)) - tar_mesh_->point(tar_mesh_->vertex_handle(i))).norm();
        }
        pars_.init_gt_mean_errs = std::sqrt(gt_err.squaredNorm()/n_src_vertex_);
        pars_.init_gt_max_errs = gt_err.maxCoeff();
    }
}


Scalar NonRigidreg::DoNonRigid()
{
    // Data term parameters
    Scalar nu1 = pars_.Data_initk * pars_.Data_nu;

	if(pars_.use_coarse_reg)
	{
		optimize_w_align = 1.0;
		optimize_w_smo = pars_.w_smo/reg_coeff_B_.rows() * sampling_indices_.size(); 
		optimize_w_rot = pars_.w_rot/num_sample_nodes * sampling_indices_.size();
		optimize_w_arap = pars_.w_arap_coarse/arap_coeff_.rows() * sampling_indices_.size();
		std::cout << "Coarse Stage: optimize w_align = " << optimize_w_align << " | w_smo = " << optimize_w_smo << " | w_rot = " << optimize_w_rot << " | w_arap = " << optimize_w_arap << std::endl;
		GraphCoarseReg(nu1);
	}
	else
	{
		std::cout << "Coarse registration does not apply!" << std::endl;
	}

	
	if(pars_.use_fine_reg)
	{
		optimize_w_align = 1.0;
		optimize_w_arap = pars_.w_arap_fine /arap_coeff_fine_.rows() * n_src_vertex_;
		std::cout << "Fine Stage: optimize w_align = " << optimize_w_align << " | w_arap = " << optimize_w_arap << std::endl;
		PointwiseFineReg(nu1);
	}
	else{
		std::cout << "Fine registration does not apply!" << std::endl;
	}

	Scalar gt_err = SetMeshPoints(src_mesh_, deformed_points_);
    return 0;
}

void NonRigidreg::CalcNodeRotations()
{
#pragma omp parallel for
    for (int i = 0; i < num_sample_nodes; i++)
    {
		Matrix33 rot;
        Eigen::JacobiSVD<Matrix33> svd(Eigen::Map<Matrix33>(X_.data()+12*i, 3,3), Eigen::ComputeFullU | Eigen::ComputeFullV);
        if (svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
            Vector3 S = Vector3::Ones(); S(2) = -1.0;
            rot = svd.matrixU()*S.asDiagonal()*svd.matrixV().transpose();
        }
        else {
            rot = svd.matrixU()*svd.matrixV().transpose();
        }
		nodes_R_.segment(9 * i, 9) = Eigen::Map<VectorX>(rot.data(), 9);
    }
}

void NonRigidreg::welsch_weight(VectorX& r, Scalar p) {
#pragma omp parallel for
    for (int i = 0; i<r.rows(); ++i) {
		if(r[i] >= 0)
        	r[i] = std::exp(-r[i] / (2 * p*p));
		else
			r[i] = 0.;
    }
}

Scalar NonRigidreg::SetMeshPoints(Mesh* mesh, const VectorX & target)
{
    VectorX gt_errs(n_src_vertex_);
#pragma omp parallel for
    for (int i = 0; i < n_src_vertex_; i++)
    {
		Vec3 p(target[i * 3], target[i * 3 + 1], target[i * 3 + 2]);
        mesh->set_point(mesh->vertex_handle(i), p);
		Vec3 n(deformed_normals_(0,i), deformed_normals_(1,i),deformed_normals_(2,i));
		mesh->set_normal(mesh->vertex_handle(i), n);
		if (pars_.calc_gt_err)
			gt_errs[i] = (target.segment(3 * i, 3) - tar_points_.col(i)).squaredNorm();
    }
    if(pars_.calc_gt_err)
        return gt_errs.sum()/n_src_vertex_;
    else
        return -1.0;
}


void NonRigidreg::FullInARAPCoeff()
{
	arap_laplace_weights_.resize(n_src_vertex_);
	Timer timer;
	
	if(pars_.use_geodesic_dist)
	{
		for (int i = 0; i < n_src_vertex_; i++)
		{
			int nn = 0;
			OpenMesh::VertexHandle vh = src_mesh_->vertex_handle(i);
			for (auto vv = src_mesh_->vv_begin(vh); vv != src_mesh_->vv_end(vh); vv++)
			{
				nn++;
			}
			arap_laplace_weights_[i] = 1.0 / nn;
		}

		std::vector<Triplet> coeffs;
		std::vector<Triplet> coeffs_fine;
		for (int i = 0; i < src_mesh_->n_halfedges(); i++)
		{
			int src_idx = src_mesh_->from_vertex_handle(src_mesh_->halfedge_handle(i)).idx();
			int tar_idx = src_mesh_->to_vertex_handle(src_mesh_->halfedge_handle(i)).idx();
			Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

			if(pars_.use_coarse_reg)
			{
			for (int k = 0; k < 3; k++)
			{
				for (RowMajorSparseMatrix::InnerIterator it(align_coeff_PV0_, src_idx*3+k); it; ++it)
				{
					coeffs.push_back(Triplet(i*3+k, it.col(), w*it.value()));
				}
				for (RowMajorSparseMatrix::InnerIterator it(align_coeff_PV0_, tar_idx*3+k); it; ++it)
				{
					coeffs.push_back(Triplet(i*3+k, it.col(), -w*it.value()));
				}
			}
			}

			coeffs_fine.push_back(Triplet(i * 3, src_idx * 3, w));
			coeffs_fine.push_back(Triplet(i * 3 + 1, src_idx * 3 + 1, w));
			coeffs_fine.push_back(Triplet(i * 3 + 2, src_idx * 3 + 2, w));
			coeffs_fine.push_back(Triplet(i * 3, tar_idx * 3, -w));
			coeffs_fine.push_back(Triplet(i * 3 + 1, tar_idx * 3 + 1, -w));
			coeffs_fine.push_back(Triplet(i * 3 + 2, tar_idx * 3 + 2, -w));
		}
		
		if(pars_.use_coarse_reg)
		{
			arap_coeff_.resize(src_mesh_->n_halfedges()*3, num_sample_nodes * 12);
			arap_coeff_.setFromTriplets(coeffs.begin(), coeffs.end());
			arap_coeff_mul_ = arap_coeff_.transpose() * arap_coeff_;
		}

		arap_coeff_fine_.resize(src_mesh_->n_halfedges() * 3, n_src_vertex_ * 3);
		arap_coeff_fine_.setFromTriplets(coeffs_fine.begin(), coeffs_fine.end());
		arap_coeff_mul_fine_ = arap_coeff_fine_.transpose() * arap_coeff_fine_;
	}
	else
	{
		int nn = src_knn_indices_.rows();
		for (int i = 0; i < n_src_vertex_; i++)
		{
			arap_laplace_weights_[i] = 1.0 / nn;
		}

		std::vector<Triplet> coeffs;
		std::vector<Triplet> coeffs_fine;
		for(int src_idx = 0; src_idx < n_src_vertex_; src_idx++)
		{
			for(int j = 0; j < nn; j++)
			{
				int i = src_idx*nn+j;
				int tar_idx = src_knn_indices_(j, src_idx);
				Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

				if(pars_.use_coarse_reg)
				{
				for (int k = 0; k < 3; k++)
				{
					for (RowMajorSparseMatrix::InnerIterator it(align_coeff_PV0_, src_idx*3+k); it; ++it)
					{
						coeffs.push_back(Triplet(i*3+k, it.col(), w*it.value()));
					}
					for (RowMajorSparseMatrix::InnerIterator it(align_coeff_PV0_, tar_idx*3+k); it; ++it)
					{
						coeffs.push_back(Triplet(i*3+k, it.col(), -w*it.value()));
					}
				}
				}

				coeffs_fine.push_back(Triplet(i * 3, src_idx * 3, w));
				coeffs_fine.push_back(Triplet(i * 3 + 1, src_idx * 3 + 1, w));
				coeffs_fine.push_back(Triplet(i * 3 + 2, src_idx * 3 + 2, w));
				coeffs_fine.push_back(Triplet(i * 3, tar_idx * 3, -w));
				coeffs_fine.push_back(Triplet(i * 3 + 1, tar_idx * 3 + 1, -w));
				coeffs_fine.push_back(Triplet(i * 3 + 2, tar_idx * 3 + 2, -w));
			}
		}
		if(pars_.use_coarse_reg)
		{
			arap_coeff_.resize(n_src_vertex_*nn*3, num_sample_nodes * 12);
			arap_coeff_.setFromTriplets(coeffs.begin(), coeffs.end());
			arap_coeff_mul_ = arap_coeff_.transpose() * arap_coeff_;
		}

		arap_coeff_fine_.resize(n_src_vertex_*nn * 3, n_src_vertex_ * 3);
		arap_coeff_fine_.setFromTriplets(coeffs_fine.begin(), coeffs_fine.end());
		arap_coeff_mul_fine_ = arap_coeff_fine_.transpose() * arap_coeff_fine_;
	}
}


void NonRigidreg::CalcARAPRight()
{
	if(pars_.use_geodesic_dist)
	{
		#pragma omp parallel for
		for (int i = 0; i < src_mesh_->n_halfedges(); i++)
		{
			int src_idx = src_mesh_->from_vertex_handle(src_mesh_->halfedge_handle(i)).idx();
			int tar_idx = src_mesh_->to_vertex_handle(src_mesh_->halfedge_handle(i)).idx();

			Vector3 vij = local_rotations_.block(0, 3 * src_idx,3, 3) * (src_points_.col(src_idx) - src_points_.col(tar_idx));

			Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

			
			arap_right_[i * 3] = w*(vij[0] - nodes_P_[src_idx * 3] + nodes_P_[tar_idx * 3]);
			arap_right_[i * 3 + 1] = w*(vij[1] - nodes_P_[src_idx * 3 + 1] + nodes_P_[tar_idx * 3 + 1]);
			arap_right_[i * 3 + 2] = w*(vij[2] - nodes_P_[src_idx * 3 + 2] + nodes_P_[tar_idx * 3 + 2]);
		}
	}
	else
	{
		int nn = src_knn_indices_.rows();
		#pragma omp parallel for
		for (int src_idx = 0; src_idx < n_src_vertex_; src_idx++)
		{
			for (int j = 0; j < nn; j++)
			{
			int i = src_idx*nn + j;
			int tar_idx = src_knn_indices_(j, src_idx);

			Vector3 vij = local_rotations_.block(0, 3 * src_idx,3, 3) * (src_points_.col(src_idx) - src_points_.col(tar_idx));

			Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

			
			arap_right_[i * 3] = w*(vij[0] - nodes_P_[src_idx * 3] + nodes_P_[tar_idx * 3]);
			arap_right_[i * 3 + 1] = w*(vij[1] - nodes_P_[src_idx * 3 + 1] + nodes_P_[tar_idx * 3 + 1]);
			arap_right_[i * 3 + 2] = w*(vij[2] - nodes_P_[src_idx * 3 + 2] + nodes_P_[tar_idx * 3 + 2]);
			}
		}
	}
}

void NonRigidreg::CalcARAPRightFine()
{
	if(pars_.use_geodesic_dist)
	{
		#pragma omp parallel for
		for (int i = 0; i < src_mesh_->n_halfedges(); i++)
		{
			int src_idx = src_mesh_->from_vertex_handle(src_mesh_->halfedge_handle(i)).idx();
			int tar_idx = src_mesh_->to_vertex_handle(src_mesh_->halfedge_handle(i)).idx();

			Vector3 vij = local_rotations_.block(0, 3 * src_idx, 3, 3) * (src_points_.col(src_idx) - src_points_.col(tar_idx));

			Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

			arap_right_fine_[i * 3] = w*(vij[0]);
			arap_right_fine_[i * 3 + 1] = w*(vij[1]);
			arap_right_fine_[i * 3 + 2] = w*(vij[2]);
		}
	}
	else
	{
		int nn = src_knn_indices_.rows();
		#pragma omp parallel for
		for(int src_idx = 0; src_idx < n_src_vertex_; src_idx++)
		{
			for(int j = 0; j < nn; j++)
			{
				int tar_idx = src_knn_indices_(j, src_idx);
				int i = src_idx*nn+j;

				Vector3 vij = local_rotations_.block(0, 3 * src_idx, 3, 3) * (src_points_.col(src_idx) - src_points_.col(tar_idx));

				Scalar w = sqrtf(arap_laplace_weights_[src_idx]);

				arap_right_fine_[i * 3] = w*(vij[0]);
				arap_right_fine_[i * 3 + 1] = w*(vij[1]);
				arap_right_fine_[i * 3 + 2] = w*(vij[2]);
			}
		}
	}
}

void NonRigidreg::InitRotations()
{
	local_rotations_.resize(3, n_src_vertex_ * 3);
	local_rotations_.setZero();
#pragma omp parallel for
	for (int i = 0; i < n_src_vertex_; i++)
	{
		local_rotations_(0, i * 3) = 1;
		local_rotations_(1, i * 3 + 1) = 1;
		local_rotations_(2, i * 3 + 2) = 1;
	}
}


void NonRigidreg::CalcLocalRotations(bool isCoarseAlign)
{
	
	#pragma omp parallel for
	for (int i = 0; i < n_src_vertex_; i++)
	{
		Matrix33 sum;
		sum.setZero();

		int nn = 0;
		if(pars_.use_geodesic_dist)
		{
			OpenMesh::VertexHandle vh = src_mesh_->vertex_handle(i);
			for (auto vv = src_mesh_->vv_begin(vh); vv != src_mesh_->vv_end(vh); vv++)
			{
				int neighbor_idx = vv->idx();
				Vector3 dv = src_points_.col(i) - src_points_.col(neighbor_idx);
				Vector3 new_dv = deformed_points_.segment(3 * i, 3) - deformed_points_.segment(3 * neighbor_idx, 3);
				sum += dv * new_dv.transpose();
				nn++;
			}
		}
		else
		{
			nn = src_knn_indices_.rows();
			for(int j = 0; j < nn; j++)
			{
				int neighbor_idx = src_knn_indices_(j, i);
				Vector3 dv = src_points_.col(i) - src_points_.col(neighbor_idx);
				Vector3 new_dv = deformed_points_.segment(3 * i, 3) - deformed_points_.segment(3 * neighbor_idx, 3);
				sum += dv * new_dv.transpose();
			}
		}
		

		sum*= 1.0*optimize_w_arap/nn;
		
		if(!isCoarseAlign)
		{
			int tar_idx = correspondence_pairs_[i].tar_idx;
			Vector3 d = deformed_points_.segment(3 * i, 3) - tar_points_.col(tar_idx);
			Scalar c = (target_normals_.col(tar_idx) + deformed_normals_.col(i)).dot(d);
			Scalar d_norm2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
			Vector3 h = deformed_normals_.col(i) - c*d/d_norm2;

			Scalar w = optimize_w_align*d_norm2*weight_d_[i];
			sum += w * src_normals_.col(i) * h.transpose();
		}
		else if(vertex_sample_indices_[i] >= 0)
		{
			//Vector3 Rv = rotations_.block(3 * i, 0, 3, 3) * src_normals.col(i);
			// R^k n_s - d * (n_t + R^k n_s )*d / ||d||^2
			// deformed_normals = R^k n_s
			// d = v_i - u_j
			int tar_idx = correspondence_pairs_[vertex_sample_indices_[i]].tar_idx;
			Vector3 d = deformed_points_.segment(3 * i, 3) - tar_points_.col(tar_idx);
			Scalar c = (target_normals_.col(tar_idx) + deformed_normals_.col(i)).dot(d);
			Scalar d_norm2 = d[0]*d[0] + d[1]*d[1] + d[2]*d[2];
			Vector3 h = deformed_normals_.col(i) - c*d/d_norm2;

			Scalar w = optimize_w_align*d_norm2*weight_d_[vertex_sample_indices_[i]];
			sum += w * src_normals_.col(i) * h.transpose();
		}
		

		Eigen::JacobiSVD<Matrix33> svd(sum, Eigen::ComputeFullU | Eigen::ComputeFullV);

		if (svd.matrixU().determinant()*svd.matrixV().determinant() < 0.0) {
			Vector3 S = Vector3::Ones(); S(2) = -1.0;
			sum = svd.matrixV()*S.asDiagonal()*svd.matrixU().transpose();
		}
		else {
			sum = svd.matrixV()*svd.matrixU().transpose();
		}
		for (int s = 0; s < 3; s++)
		{
			for (int t = 0; t < 3; t++)
			{
				local_rotations_(s, 3 * i + t) = sum(s, t);
			}
		}
	}
}

void NonRigidreg::CalcDeformedNormals()
{
#pragma omp parallel for
	for (int i = 0; i < n_src_vertex_; i++)
	{
		deformed_normals_.col(i) = local_rotations_.block(0, i * 3, 3, 3) * src_normals_.col(i);
	}
}

void NonRigidreg::CalcNormalsSum()
{
	#pragma omp parallel for
	for(int i = 0; i < correspondence_pairs_.size(); i++)
    {
		int sidx = correspondence_pairs_[i].src_idx;
		int tidx = correspondence_pairs_[i].tar_idx;
		int j = 0;
        for (RowMajorSparseMatrix::InnerIterator it(normals_sum_, i); it; ++it)
        {
			it.valueRef() = deformed_normals_(j, sidx) + target_normals_(j, tidx);
			j++;
        }
    }
}

void NonRigidreg::InitNormalsSum()
{
	std::vector<Triplet> coeffs(3 * correspondence_pairs_.size());
	normals_sum_.resize(correspondence_pairs_.size(), 3*n_src_vertex_);
	normals_sum_.setZero();

#pragma omp parallel for
	for(int i = 0; i < correspondence_pairs_.size(); i++)
	{
		int sidx = correspondence_pairs_[i].src_idx;
		int tidx = correspondence_pairs_[i].tar_idx;
		coeffs[i * 3] = Triplet(i, 3 * sidx, deformed_normals_(0, sidx) + target_normals_(0, tidx));
		coeffs[i * 3 + 1] = Triplet(i, 3 * sidx + 1, deformed_normals_(1, sidx) + target_normals_(1, tidx));
		coeffs[i * 3 + 2] = Triplet(i, 3 * sidx + 2, deformed_normals_(2, sidx) + target_normals_(2, tidx));
	}
	normals_sum_.setFromTriplets(coeffs.begin(), coeffs.end());
}

void NonRigidreg::PointwiseFineReg(Scalar nu1)
{
	Scalar energy=-1., align_err=-1., arap_err=-1.;

	VectorX prevV = VectorX::Zero(n_src_vertex_ * 3);

	bool run_once = true;

	Timer time;
	Timer::EventID begin_time, run_time;

	// Smooth term parameters
	w_align = optimize_w_align; 
	w_smo = optimize_w_smo; 

	if (pars_.data_use_robust_weight)
	{
		w_align = optimize_w_align *(2.0*nu1*nu1);
	}

	pars_.each_energys.push_back(0.0);
	pars_.each_gt_max_errs.push_back(pars_.init_gt_max_errs);
	pars_.each_gt_mean_errs.push_back(pars_.init_gt_mean_errs);
	pars_.each_iters.push_back(0);
	pars_.each_times.push_back(pars_.non_rigid_init_time);
	pars_.each_term_energy.push_back(Vector4(0, 0, 0, 0));

	Scalar gt_err = -1;

	double construct_mat_time = 0.0;
	double solve_eq_time = 0.0;

	begin_time = time.get_time();

	int out_iter = 0;
	while (out_iter < pars_.max_outer_iters)
	{
		// Find clost points
		FindClosestPoints(correspondence_pairs_, deformed_points_);
		// according correspondence_pairs to update corres_U0_;
		corres_U0_.setZero();
		weight_d_.resize(n_src_vertex_);

		for (size_t i = 0; i < correspondence_pairs_.size(); i++)
		{
			corres_U0_.segment(i * 3, 3) = correspondence_pairs_[i].position;
			weight_d_[i] = correspondence_pairs_[i].min_dist2;
			int tar_idx = correspondence_pairs_[i].tar_idx;
			if(deformed_normals_.col(i).dot(target_normals_.col(tar_idx))<0)
				weight_d_[i] = -1;
		}

		// update weight
		if (pars_.data_use_robust_weight)
		{
			welsch_weight(weight_d_, nu1);
		}
		else
			weight_d_.setOnes();

		// int welsch_iter;
		int total_inner_iters = 0;

		if (run_once == true && pars_.use_landmark == true)
		{
			weight_d_.setOnes();
		}


		// update V,U and D
		#ifdef DEBUG
		Timer::EventID construct_mat_begin = time.get_time();
		#endif
		// construct matrix A0 and pre-decompose
		if (pars_.use_symm_ppl)
		{
			if(out_iter==0)
				InitNormalsSum();
			else
				CalcNormalsSum();
			
			RowMajorSparseMatrix normals_sum_mul = normals_sum_.transpose() * weight_d_.asDiagonal()* normals_sum_;
			mat_A0_ = optimize_w_align * normals_sum_mul
				+ optimize_w_arap * arap_coeff_mul_fine_;
			
			CalcARAPRightFine();

			vec_b_ = optimize_w_align * normals_sum_mul * corres_U0_
				+ optimize_w_arap * arap_coeff_fine_.transpose() * arap_right_fine_;
			
		}
		else
		{
			RowMajorSparseMatrix diag_weights;
			diag_weights.resize(n_src_vertex_ * 3, n_src_vertex_ * 3);
			std::vector<Triplet> coeffs_diag_weights(n_src_vertex_ * 3);

			VectorX weight_corres_U(n_src_vertex_*3);
			
			for(int i = 0; i < n_src_vertex_; i++)
			{
				coeffs_diag_weights[i*3] = Triplet(i*3, i*3, weight_d_[i]);
				coeffs_diag_weights[i*3+1] = Triplet(i*3+1, i*3+1, weight_d_[i]);
				coeffs_diag_weights[i*3+2] = Triplet(i*3+2, i*3+2, weight_d_[i]);

				weight_corres_U[i*3] = weight_d_[i] * corres_U0_[i*3];
				weight_corres_U[i*3+1] = weight_d_[i] * corres_U0_[i*3+1];
				weight_corres_U[i*3+2] = weight_d_[i] * corres_U0_[i*3+2];
			}
			diag_weights.setFromTriplets(coeffs_diag_weights.begin(), coeffs_diag_weights.end());

			mat_A0_ = optimize_w_align * diag_weights
				+ optimize_w_arap * arap_coeff_mul_fine_;
			
			CalcARAPRightFine();
			vec_b_ = optimize_w_align * weight_corres_U
				+ optimize_w_arap * arap_coeff_fine_.transpose() * arap_right_fine_;
		}

		
		#ifdef DEBUG
		Timer::EventID construct_mat_end = time.get_time();
		construct_mat_time += time.elapsed_time(construct_mat_begin, construct_mat_end);
		#endif
		
		if (run_once)
		{
			solver_.analyzePattern(mat_A0_);
			run_once = false;
		}
		solver_.factorize(mat_A0_);
		deformed_points_ = solver_.solve(vec_b_);

		run_time = time.get_time();
		double eps_time = time.elapsed_time(begin_time, run_time);
		pars_.each_times.push_back(eps_time);

		#ifdef DEBUG
		solve_eq_time += time.elapsed_time(construct_mat_end, run_time);
		energy = CalcEnergyFine(align_err, arap_err);
		#endif

		
		CalcLocalRotations(false);
		CalcDeformedNormals();
		
		#ifdef DEBUG
		if (n_src_vertex_ == n_tar_vertex_)
			gt_err = (deformed_points_ - Eigen::Map<VectorX>(tar_points_.data(), 3 * n_src_vertex_)).squaredNorm();
		#endif

		// save results
		pars_.each_gt_mean_errs.push_back(gt_err);
		pars_.each_gt_max_errs.push_back(0);
		pars_.each_energys.push_back(energy);
		pars_.each_iters.push_back(total_inner_iters);
		pars_.each_term_energy.push_back(Vector4(align_err, 0, 0, arap_err));

		if((deformed_points_ - prevV).norm()/sqrtf(n_src_vertex_) < pars_.stop_fine)
		{
			break;
		}
		prevV = deformed_points_;
		out_iter++;
	}

	#ifdef DEBUG
	std::cout << "construct mat time = " << construct_mat_time 
	<< "\nsolve_eq time = " << solve_eq_time  << " iters = " << out_iter << std::endl;
	#endif
}

void NonRigidreg::GraphCoarseReg(Scalar nu1)
{
	Scalar energy=0., align_err=0., reg_err=0., rot_err=0., arap_err=0.;
	VectorX prevV = VectorX::Zero(n_src_vertex_ * 3);

	// welsch_sweight
 	bool run_once = true;

	Timer time;
	Timer::EventID begin_time, run_time;
	pars_.each_energys.clear();
	pars_.each_gt_mean_errs.clear();
	pars_.each_gt_max_errs.clear();
	pars_.each_times.clear();
	pars_.each_iters.clear();
	pars_.each_term_energy.clear();

	w_align = optimize_w_align; 
	w_smo = optimize_w_smo; 

	if (pars_.data_use_robust_weight)
	{
		w_align = optimize_w_align *(2.0*nu1*nu1);
	}

	pars_.each_energys.push_back(0.0);
	pars_.each_gt_max_errs.push_back(pars_.init_gt_max_errs);
	pars_.each_gt_mean_errs.push_back(pars_.init_gt_mean_errs);
	pars_.each_iters.push_back(0);
	pars_.each_times.push_back(pars_.non_rigid_init_time);
	pars_.each_term_energy.push_back(Vector4(0, 0, 0, 0));

	Scalar gt_err;

	VectorX prev_X = X_;

	begin_time = time.get_time();
	
	#ifdef DEBUG
	double find_cp_time = 0.0;
	double construct_mat_time = 0.0;
	double solve_eq_time = 0.0;
	double calc_energy_time = 0.0;
	double robust_weight_time = 0.0;
	double update_r_time = 0.0;
	#endif


		
	RowMajorSparseMatrix A_fixed_coeff = optimize_w_smo * reg_coeff_B_.transpose() * reg_cwise_weights_.asDiagonal() * reg_coeff_B_  + optimize_w_rot * rigid_coeff_L_ + optimize_w_arap * arap_coeff_mul_;
	int out_iter = 0;
	while (out_iter < pars_.max_outer_iters)
	{
		#ifdef DEBUG
		Timer::EventID inner_start_time = time.get_time();
		#endif
		
		correspondence_pairs_.clear();
		FindClosestPoints(correspondence_pairs_, deformed_points_, sampling_indices_);
		corres_U0_.setZero();
		weight_d_.resize(correspondence_pairs_.size());
		weight_d_.setConstant(-1);

#pragma omp parallel for
		for (size_t i = 0; i < correspondence_pairs_.size(); i++)
		{
			corres_U0_.segment(correspondence_pairs_[i].src_idx * 3, 3) = correspondence_pairs_[i].position;
			weight_d_[i] = correspondence_pairs_[i].min_dist2;
			if(deformed_normals_.col(correspondence_pairs_[i].src_idx).dot(target_normals_.col(correspondence_pairs_[i].tar_idx))<0)
				weight_d_[i] = -1;
		}


		#ifdef DEBUG
		Timer::EventID end_find_cp = time.get_time();
		double eps_time1 = time.elapsed_time(inner_start_time, end_find_cp);
		find_cp_time += eps_time1;
		#endif

		// update weight
		if (pars_.data_use_robust_weight)
		{
			welsch_weight(weight_d_, nu1);
		}
		else
		{
			weight_d_.setOnes();
		}
		
		// int welsch_iter;
		int total_inner_iters = 0;

		if (run_once == true && pars_.use_landmark == true)
		{
			weight_d_.setOnes();
		}


		#ifdef DEBUG
			Timer::EventID end_robust_weight = time.get_time();
			eps_time1 = time.elapsed_time(end_find_cp, end_robust_weight);
			robust_weight_time += eps_time1;
			#endif

		if(pars_.use_symm_ppl)
		{			
			// 0.5s (0.01s / 50 iters)
			if(out_iter==0)					
				InitNormalsSum();
			else
				CalcNormalsSum();
			
			// 1e-3
			diff_UP_ = (corres_U0_ - nodes_P_);

			// 0.37s 
			RowMajorSparseMatrix weight_NPV = normals_sum_ * align_coeff_PV0_;

			mat_A0_ = optimize_w_align * weight_NPV.transpose() * weight_d_.asDiagonal() *  weight_NPV + A_fixed_coeff; 

			CalcARAPRight();
			vec_b_ = optimize_w_align * weight_NPV.transpose() * weight_d_.asDiagonal() * normals_sum_ * diff_UP_ + optimize_w_smo * reg_coeff_B_.transpose() * reg_cwise_weights_.asDiagonal() * reg_right_D_ + optimize_w_rot * rigid_coeff_J_ * nodes_R_ + optimize_w_arap * arap_coeff_.transpose() * arap_right_;

		}
		else
		{
			VectorX weight_d3(3*n_src_vertex_);
			weight_d3.setZero();
			for(int i = 0; i < n_src_vertex_; i++)
			{
				int idx = vertex_sample_indices_[i];
				if(idx>=0)
					weight_d3[i*3] = weight_d3[i*3+1] = weight_d3[i*3+2] = weight_d_[idx];
			}

			diff_UP_ = (corres_U0_ - nodes_P_);

			mat_A0_ = optimize_w_align *align_coeff_PV0_.transpose() * weight_d3.asDiagonal() * align_coeff_PV0_  + A_fixed_coeff; 
			
			CalcARAPRight();
			vec_b_ = optimize_w_align * align_coeff_PV0_.transpose() * weight_d3.asDiagonal() * diff_UP_ + optimize_w_smo * reg_coeff_B_.transpose() * reg_cwise_weights_.asDiagonal() * reg_right_D_ + optimize_w_rot * rigid_coeff_J_ * nodes_R_ + optimize_w_arap * arap_coeff_.transpose() * arap_right_;


		}

		#ifdef DEBUG
		Timer::EventID end_construct_eq = time.get_time();
		eps_time1 = time.elapsed_time(end_robust_weight, end_construct_eq);
		construct_mat_time += eps_time1;	
		#endif

		if (run_once)
		{
			solver_.analyzePattern(mat_A0_);
			run_once = false;
		}
		solver_.factorize(mat_A0_);
		X_ = solver_.solve(vec_b_);		

		run_time = time.get_time();
		double eps_time = time.elapsed_time(begin_time, run_time);
		pars_.each_times.push_back(eps_time);

		#ifdef DEBUG
		eps_time1 = time.elapsed_time(end_construct_eq, run_time);
		solve_eq_time += eps_time1;
		#endif


		#ifdef DEBUG
		energy = CalcEnergy(align_err, reg_err, rot_err, arap_err, reg_cwise_weights_);
		#endif 

		deformed_points_ = align_coeff_PV0_ * X_ + nodes_P_;

		
		#ifdef DEBUG
		Timer::EventID end_calc_energy = time.get_time();
		eps_time1 = time.elapsed_time(run_time, end_calc_energy);
		calc_energy_time += eps_time1;
		#endif

		CalcLocalRotations(true);

		#ifdef DEBUG
		Timer::EventID end_update_r = time.get_time();
		eps_time1 = time.elapsed_time(end_calc_energy, end_update_r);
		update_r_time += eps_time1;
		#endif

		CalcNodeRotations();
		CalcDeformedNormals();

		if (n_src_vertex_ == n_tar_vertex_)
			gt_err = (deformed_points_ - Eigen::Map<VectorX>(tar_points_.data(), 3 * n_src_vertex_)).squaredNorm();

		// save results
		pars_.each_gt_mean_errs.push_back(gt_err);
		pars_.each_gt_max_errs.push_back(0);
		pars_.each_energys.push_back(energy);
		pars_.each_iters.push_back(total_inner_iters);
		pars_.each_term_energy.push_back(Vector4(align_err, reg_err, rot_err, arap_err));

		if((deformed_points_ - prevV).norm()/sqrtf(n_src_vertex_) < pars_.stop_coarse)
		{
			break;
		}
		prevV = deformed_points_;
		out_iter++;

		#ifdef DEBUG
		Timer::EventID end_find_cp2 = time.get_time();
		eps_time1 = time.elapsed_time(end_calc_energy, end_find_cp2);
		find_cp_time += eps_time1;
		#endif
	}

	#ifdef DEBUG
	std::cout << "find cp time = " << find_cp_time
	 << "\nconstruct_mat_timem = " << construct_mat_time
	 << "\nsolve_eq_time = " << solve_eq_time
	 << "\ncalc_energy_time = " << calc_energy_time
	 << "\nrobust_weight_time = " << robust_weight_time
	 << "\nupdate_r_time = " << update_r_time
	 << "\nacculate_iter = " << out_iter << std::endl;
	#endif
}

Scalar NonRigidreg::CalcEnergy(Scalar& E_align, Scalar& E_reg,
	Scalar& E_rot, Scalar& E_arap, VectorX & reg_weights)
{
	if(pars_.use_symm_ppl)
		E_align = (normals_sum_ * (align_coeff_PV0_ * X_ - diff_UP_)).squaredNorm();
	else
		E_align = ((align_coeff_PV0_ * X_ - diff_UP_)).squaredNorm();
	
	E_reg = (reg_coeff_B_ * X_ - reg_right_D_).squaredNorm();
	E_arap = (arap_coeff_ * X_ - arap_right_).squaredNorm();
	E_rot = (rigid_coeff_J_.transpose() * X_ - nodes_R_).squaredNorm();
	
	Scalar energy = w_align * E_align 
		+ w_smo * E_reg 
		+ optimize_w_arap * E_arap 
		+ optimize_w_rot * E_rot;
	return energy;
}

Scalar NonRigidreg::CalcEnergyFine(Scalar & E_align, Scalar & E_arap)
{
	if(pars_.use_symm_ppl)
		E_align = (normals_sum_ * (deformed_points_ - corres_U0_)).squaredNorm();
	else
		E_align = ((deformed_points_ - corres_U0_)).squaredNorm();
	E_arap = (arap_coeff_fine_ * deformed_points_  - arap_right_fine_).squaredNorm();

	Scalar energy = w_align * E_align
		+ optimize_w_arap * E_arap;
	return energy;
}

// *type: 0 :median, 1: average
Scalar NonRigidreg::CalcEdgelength(Mesh* mesh, int type)
{
    Scalar med;
    if(mesh->n_faces() > 0)
    {
        VectorX edges_length(mesh->n_edges());
        for(size_t i = 0; i < mesh->n_edges();i++)
        {
            OpenMesh::VertexHandle vi = mesh->from_vertex_handle(mesh->halfedge_handle(mesh->edge_handle(i),0));
            OpenMesh::VertexHandle vj = mesh->to_vertex_handle(mesh->halfedge_handle(mesh->edge_handle(i),0));
            edges_length[i] = (mesh->point(vi) - mesh->point(vj)).norm();
        }
        if (type == 0)
            igl::median(edges_length, med);
        else
            med = edges_length.mean();
    }
    else
    {
        // source is mesh, target may be point cloud.
		int nn = src_knn_indices_.rows();
		VectorX edges_length(n_src_vertex_*nn);
        for(int src_idx = 0; src_idx < n_src_vertex_; src_idx++)
		{
			for(int j = 0; j < nn; j++)
			{
				int tar_idx = src_knn_indices_(j, src_idx);
				Scalar dist = (src_points_.col(src_idx) - src_points_.col(tar_idx)).norm();
				edges_length[src_idx*nn+j] = dist;
			}
		}
        if (type == 0)
            igl::median(edges_length, med);
        else
            med = edges_length.mean();
    }
    return med;
}