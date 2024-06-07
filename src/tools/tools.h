#ifndef TOOL_H_
#define TOOL_H_
#include "Types.h"

enum CorresType {CLOSEST, LANDMARK};
enum PruningType {SIMPLE, NONE};

struct RegParas
{
    int		max_outer_iters;    // nonrigid max iters
    Scalar	w_smo;              // smoothness weight 
    Scalar	w_rot;               // rotation matrix weight 
	Scalar  w_arap_coarse;      // ARAP weight for coarse alignment
    Scalar  w_arap_fine;        // ARAP weight for fine alignment 
    bool	use_normal_reject;  // use normal reject or not
    bool	use_distance_reject;// use distance reject or not
    Scalar	normal_threshold;
    Scalar	distance_threshold;
    int     rigid_iters;         // rigid registration max iterations
    bool	use_landmark;
    bool    use_fixedvex;        // Set the point which haven't correspondences
    bool    calc_gt_err;         // calculate ground truth error (DEBUG)
    bool    data_use_robust_weight;  // use robust weights for alignment term or not

	bool	use_symm_ppl; // use the symmetric point-to-plane distance or point-to-point distance.

    std::vector<int> landmark_src;
    std::vector<int> landmark_tar;
    std::vector<int> fixed_vertices;

    Scalar  Data_nu;
    Scalar  Data_initk;
    Scalar  Data_endk;
    Scalar  stop_coarse;
    Scalar  stop_fine;

	// Sample para
    Scalar  uni_sample_radio;       // uniform sample radio
    bool    use_geodesic_dist;
    bool    print_each_step_info;   // debug output : each step nodes, correspondences


    // output path
    std::string out_gt_file;
    std::string out_each_step_info;
    int         num_sample_nodes;

    std::vector<Scalar> each_times;
    std::vector<Scalar> each_gt_mean_errs;
    std::vector<Scalar> each_gt_max_errs;
    std::vector<Scalar> each_energys;
    std::vector<Scalar> each_iters;
    std::vector<Vector4> each_term_energy;
    Scalar  non_rigid_init_time;
    Scalar  init_gt_mean_errs;
    Scalar  init_gt_max_errs;

    bool    use_coarse_reg;
    bool    use_fine_reg;

    Scalar  mesh_scale;

    RegParas() // default
    {
        max_outer_iters = 50;
        w_smo = 0.01;  // smooth
        w_rot = 1e-4;   // orth
		w_arap_coarse = 500; // 10;
        w_arap_fine = 200;
        use_normal_reject = false;
        use_distance_reject = false;
        normal_threshold = M_PI / 3;
        distance_threshold = 0.05;
        rigid_iters = 0;
        use_landmark = false;
        use_fixedvex = false;
        calc_gt_err = false;
        data_use_robust_weight = true;

		use_symm_ppl = true;

        Data_nu = 0.0;
        Data_initk = 1;
        Data_endk = 1.0/sqrt(3);
        stop_coarse = 1e-3;
        stop_fine = 1e-4;

        // Sample para
        uni_sample_radio = 5;
        use_geodesic_dist = true;
        print_each_step_info = false;

        non_rigid_init_time = .0;
        init_gt_mean_errs = .0;

        use_coarse_reg = true;
        use_fine_reg = true;
    }

    public:
    void print_params(std::string outf)
    {
        std::ofstream out(outf);
        
    
    out << "setting:\n" <<std::endl;

    // setting
    out << "use_coarse_reg: " << use_coarse_reg << std::endl;
    out << "use_fine_reg: " << use_fine_reg << std::endl;
    out << "use_symm_ppl: " << use_symm_ppl<< std::endl;
    out << "data_use_robust_weight: " << data_use_robust_weight<< std::endl;  // use robust welsch function as energy function or just use L2-norm
    
    out <<"w_smo: "<<w_smo<< std::endl;
    out << "w_rot: "<< w_rot<< std::endl;
	out << "w_arap_coarse: " << w_arap_coarse<< std::endl;
    out << "w_arap_fine: " <<  w_arap_fine<< std::endl;

    out << "uni_sample_radio: " << uni_sample_radio<< std::endl;   
    out << "use_geodesic_dist: " << use_geodesic_dist << std::endl; 
    out << "print_each_step_info: " << print_each_step_info<< std::endl;   
    out << "out_gt_file" << out_gt_file<< std::endl;
    out << "out_each_step_info" << out_each_step_info<< std::endl;

    out << "\n\noutput:\n" <<std::endl;

    // output
    out << "mesh_scale: " << mesh_scale << std::endl;
    out << "num_sample_nodes: " << num_sample_nodes<< std::endl;
    out << "non_rigid_init_time: " << non_rigid_init_time<< std::endl;
    out << "init_gt_mean_errs: " << init_gt_mean_errs<< std::endl;
    out <<"init_gt_max_errs: " << init_gt_max_errs<< std::endl;
    out << "Data_nu: " << Data_nu<< std::endl;

    out << "\n\ndefault:\n" <<std::endl;

    // default
    out << "max_outer_iters: " << max_outer_iters << std::endl;
    out << "stop_coarse: " <<stop_coarse<< std::endl;
    out << "stop_fine: " << stop_fine<< std::endl;
    
    out << "Data_initk: " << Data_initk<< std::endl;
    out << "Data_endk: " << Data_endk<< std::endl;
    out << "calc_gt_err: " << calc_gt_err<< std::endl; 
    out << "use_normal_reject: " <<	use_normal_reject<< std::endl;  // use normal reject or not
    out <<	"use_distance_reject: " << use_distance_reject<< std::endl;
    out << "normal_threshold: " <<	normal_threshold<< std::endl;
    out << "distance_threshold: " <<	distance_threshold<< std::endl;
    
    out << "\n\nuseless:\n" <<std::endl;
    
    // useless 
    out << "use_landmark: " << use_landmark<< std::endl;
    out << "use_fixedvex: " << use_fixedvex<< std::endl;  
    out <<  "rigid_iters: " << rigid_iters << std::endl; 
    out.close();
    }
};

// normalize mesh
Scalar mesh_scaling(Mesh& src_mesh, Mesh& tar_mesh);
// Convert Mesh to libigl format to calculate geodesic distance
void Mesh2VF(Mesh & mesh, MatrixXX& V, Eigen::MatrixXi& F);
Vec3 Eigen2Vec(Vector3 s);
Vector3 Vec2Eigen(Vec3 s);
// read landmark points into landmark_src and landmark_tar if they exist
bool read_landmark(const char* filename, std::vector<int>& landmark_src, std::vector<int>& landmark_tar);
// read fixed points into vertices_list if they exist
bool read_fixedvex(const char* filename, std::vector<int>& vertices_list);

#ifdef __linux__
bool my_mkdir(std::string file_path);
#endif

#endif
