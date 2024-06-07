#include "tools/io_mesh.h"
#include "tools/OmpHelper.h"
#include "NonRigidreg.h"


int main(int argc, char **argv)
{
    Mesh src_mesh;
    Mesh tar_mesh;
    std::string src_file;
    std::string tar_file;
    std::string out_file, outpath;
    std::string landmark_file;
    RegParas paras;
    Scalar input_w_smo = 0.01;
	Scalar input_w_rot = 1e-4; 
    Scalar input_radius = 10;
	Scalar input_w_arap_coarse = 500;
    Scalar input_w_arap_fine = 200;
    bool normalize = true; 

    if(argc==4)
    {
        src_file = argv[1];
        tar_file = argv[2];
        outpath = argv[3];  
    }
    
    else if(argc==10)
    {
        src_file = argv[1];
        tar_file = argv[2];
        outpath = argv[3];
        input_radius = std::stod(argv[4]);
        input_w_smo = std::stod(argv[5]);
        input_w_rot = std::stod(argv[6]);
		input_w_arap_coarse = std::stod(argv[7]);
        input_w_arap_fine = std::stod(argv[8]);
        normalize = bool(std::stoi(argv[9]));
    }
    
    else
    {
        std::cout << "Usage: <srcFile> <tarFile> <outPath>\n" << std::endl;
        std::cout << "    or <srcFile> <tarFile> <outPath> <radius> <w_smo> <w_rot> <w_arap_c> <w_arap_f> <use_normalize>" << std::endl;
        exit(0);
    }

    paras.stop_coarse = 1e-3;
    paras.stop_fine = 1e-4;
    paras.max_outer_iters = 30;
	paras.use_symm_ppl = true;
    paras.use_fine_reg = true;
    paras.use_coarse_reg = true;
    paras.use_geodesic_dist = true;

    // Setting paras
    paras.w_smo = input_w_smo;
    paras.w_rot = input_w_rot;
	paras.w_arap_coarse = input_w_arap_coarse;
    paras.w_arap_fine = input_w_arap_fine;

    paras.rigid_iters = 0;
    paras.calc_gt_err = true;
    paras.uni_sample_radio = input_radius;

    paras.print_each_step_info = false;
    out_file = outpath + "_res.ply";
    std::string out_info = outpath + "_params.txt"; 
    paras.out_each_step_info = outpath; 

    read_data(src_file, src_mesh);
    read_data(tar_file, tar_mesh);
    if(src_mesh.n_vertices()==0 || tar_mesh.n_vertices()==0)
        exit(0);

    if(src_mesh.n_vertices() != tar_mesh.n_vertices())
        paras.calc_gt_err = false;

    if(src_mesh.n_faces()==0)
        paras.use_geodesic_dist = false;

    if(paras.use_landmark)
        read_landmark(landmark_file.c_str(), paras.landmark_src, paras.landmark_tar);
	
    double scale = 1;
    if(normalize)
        scale = mesh_scaling(src_mesh, tar_mesh);

    paras.mesh_scale = scale;
    NonRigidreg* reg;
    reg = new NonRigidreg;

    Timer time;
    std::cout << "registration to initial... (mesh scale: " << scale << ")" << std::endl;
    Timer::EventID time1 = time.get_time();
    reg->InitFromInput(src_mesh, tar_mesh, paras);
    // non-rigid initialize
    reg->Initialize();
    Timer::EventID time2 = time.get_time();
    reg->pars_.non_rigid_init_time = time.elapsed_time(time1, time2);
    std::cout << "non-rigid registration... (graph node number: " << reg->pars_.num_sample_nodes << ")" << std::endl;

    reg->DoNonRigid();

    Timer::EventID time3 = time.get_time();

    std::cout << "Registration done!\ninitialize time : "
              << time.elapsed_time(time1, time2) << " s \tnon-rigid reg running time = " << time.elapsed_time(time2, time3) << " s" << std::endl;
    write_data(out_file.c_str(), src_mesh, scale);

    std::cout<< "write the result to " << out_file << "\n" << std::endl;

    
    reg->pars_.print_params(out_info);

    delete reg;
    return 0;
}