# SPARE
This repository includes the source code of the paper [SPARE: Symmetrized Point-to-Plane Distance for Robust Non-Rigid Registration](https://arxiv.org/abs/2405.20188).

Authors: [Yuxin Yao](https://yaoyx689.github.io/), [Bailin Deng](http://www.bdeng.me/), [Junhui Hou](https://sites.google.com/site/junhuihoushomepage) and [Juyong Zhang](http://staff.ustc.edu.cn/~juyong/).

### <a href="https://arxiv.org/abs/2405.20188" target="_blank">Paper</a> | <a href="https://drive.google.com/file/d/1ms8ZI5wAM5MewnFlD6Xhlv5hf_MT6-kx/view?usp=sharing" target="_blank">Video</a> | <a href="" target="_blank">Data (coming soon)</a>

- Non-rigid registration of two surfaces.
  ![demo](images/demo.png)
- Non-rigid registration of a motion sequence frame by frame using a template surface.
  ![demo](images/tracking_video.gif)

This code is protected under patent. It can be only used for research purposes. If you are interested in business purposes/for-profit use, please contact Juyong Zhang (the corresponding author, email: juyong@ustc.edu.cn).

### TODO

- [x] Release code. 
- [ ] Release processed data and evaluation code.
- [ ] Release GPU-accelerated version. 


## Dependencies
1. [Eigen-3.4.0](http://eigen.tuxfamily.org/index.php?title=Main_Page)
2. [OpenMesh-8.1](https://www.graphics.rwth-aachen.de/software/openmesh/)
3. (Optional for Linux) [MKL](https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/onemkl.html). It is used to speed up the LDLT solver(https://eigen.tuxfamily.org/dox-devel/group__PardisoSupport__Module.html). If it is not installed, Eigen's SimplicialLDLT will be used instead. 

They can be installed in the system path. If they are not installed in the system path, you need to change the install path in the `CMakeLists.txt`. 

## Compilation
The code is compiled using [CMake](https://cmake.org/) and tested on Ubuntu 20.04 (gcc9.4.0). 

Run `cd scripts & ./compile.sh` and an executable `spare` will be generated.


## Usage 
The program is run with following input parameters:
```
$ ./spare <srcFile> <tarFile> <outPath> 
or 
$ ./spare <srcFile> <tarFile> <outPath> <radius> <w_reg> <w_rot> <w_arap_c> <w_arap_f> <use_normalize>
```
<details>
  <summary> Details (click to expand) </summary>

1. `<srcFile>`: an input file storing the source mesh;

2. `<tarFile>`: an input file storing the target mesh or point cloud; 

3. `<outPath>`: an output file storing the path of registered source mesh; 

4. `<radius>`: the sampling radius of deformation graph. 

5. `<w_smo>`: the weight parameter of `smooth term` during the coarse stage.

6. `<w_rot>`: the weight parameter of `rotation matrix term` during the coarse stage.

7. `<w_arap_c>`: the weight parameter of `ARAP term` during the coarse stage.

8. `<w_arap_f>`: the weight parameter of `ARAP term` during the fine stage.

9. `<use_normalize>`: if it's set `1`, the source surface and the target surface will be scaled 
with the same scaling factor, such that they are
contained in a bounding box with a unit diagonal length during the registration process. The deformed surface will return to the original size. If the surfaces are the normalized, set it to `0`. 
</details>

## Demo 
Running `cd scripts & ./run_demo.sh` can execute the test example.

## Tools 
- Calculate normals for point cloud </summary> 
  - Install dependences
   ```
    pip install openmesh 
    pip install pymeshlab 
  ```
  - Change the file path in `estimate_normal.py` and run 
  ```
  cd useful_tools 
  python estimate_normal.py
  ``` 


### Notes
This code supports non-rigid registration with meshes or point clouds. When the input is a point cloud, the normal is required. When the source surface is represented as a point cloud, the deformation graph will be constructed by the farthest point sampling (FPS) based on Euclidean distance. 

### Citation 
If you find our code or paper helps, please consider citing:
```
@article{yao2024spare,
  author    = {Yao, Yuxin and Deng, Bailin and Hou, Junhui and Zhang, Juyong},
  title     = {SPARE: Symmetrized Point-to-Plane Distance for Robust Non-Rigid Registration},
  journal   = {Arxiv},
  year      = {2024},
}
```