cmake_minimum_required(VERSION 2.8)
project(spare)

#--- CMake configuration
set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

#--- Compiler configuration
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")


#--- OpenMP (cannot work in clang)
find_package(OpenMP QUIET)
if(OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()


# Specify whether Pardiso should be used
set(USEPARDISO ON CACHE BOOL "Use Pardiso")

#--- Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
# include_directories("libs/eigen-3.4.0")

# enable Intel-MKL (only for Intel compilers)
# NB requires correct specification of BLA_VENDOR, e.g.
#   export BLA_VENDOR=Intel10_64lp_seq
if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  find_package(BLAS)
  include_directories(${BLAS_INCLUDE_DIRS})
  set(PROJECT_LIBS ${BLAS_LIBRARIES})
  add_definitions(-DEIGEN_USE_MKL_ALL)
endif()

#--- Nanoflann
find_package(NanoFlann REQUIRED)
include_directories(${NANOFLANN_INCLUDE_DIR})

#--- Link openmesh 
# set(OpenMesh_DIR ./libs/OpenMesh-8.1/install/share/OpenMesh/cmake/)
find_package(OpenMesh REQUIRED)

#--- Build example
include_directories("./src/")
file(GLOB SOURCES "./src/*.cpp" "./src/geodesic/*.cpp" "./src/tools/*.cpp")
file(GLOB HEADERS "./src/*.h" "./src/geodesic/*.h" "./src/tools/*.h")
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

#--- openmesh libs
target_link_libraries(${PROJECT_NAME}
    OpenMeshCore OpenMeshTools)

if(APPLE OR UNIX)
    #--- Deploy data folder link
    execute_process(COMMAND ln -f -s ${CMAKE_SOURCE_DIR}/data WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    #--- "Dummy" target to have data appear in QtCreator
    add_custom_target(data SOURCES ${CMAKE_SOURCE_DIR}/data)
else()
    file(COPY ${CMAKE_SOURCE_DIR}/data DESTINATION ${PROJECT_BINARY_DIR})
endif()


# Set up Pardiso
if(USEPARDISO)
	if (${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
		if(EXISTS /opt/intel/oneapi/mkl/latest/include/mkl_pardiso.h)
            include_directories(${BLAS_INCLUDE_DIRS})
			target_compile_definitions(${PROJECT_NAME} PUBLIC -DUSE_PARDISO)
			target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC "/opt/intel/oneapi/mkl/latest/include")
			target_compile_options(${PROJECT_NAME} PUBLIC "-m64")
			target_link_libraries(${PROJECT_NAME} "-Wl,--start-group /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_intel_lp64.a  /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_gnu_thread.a /opt/intel/oneapi/mkl/latest/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl")
		else()
			message("Cannot find Intel MKL at /opt/intel/oneapi/mkl. Using Eigen LDLT for linear system solves.")
		endif()
	else()
		message("Paridiso only supported on Linux. Using Eigen LDLT for linear system solves.")
	endif()
endif()
