import pymeshlab as ml 
import openmesh as om 

def adjust_normals(points, normals, center):
    dots = ((points-center)*normals).sum(1)
    mean_dots = dots.mean()
    if mean_dots < 0:
        normals = - normals 
    return normals 

def estimate_normals(data_path, out_path):
    ms = ml.MeshSet()
    ms.load_new_mesh(data_path)
    ms.compute_normal_for_point_clouds(k=20)
    points = ms.current_mesh().vertex_matrix()
    normals = ms.current_mesh().vertex_normal_matrix()
    center = points.mean(0)
    normals = adjust_normals(points, normals, center)
    # ms.save_current_mesh(out_path,save_vertex_normal=True)
    # ms.current_mesh().vertex_normal_matrix() = normals 
    mesh = om.TriMesh()
    for i in range(points.shape[0]):
        vh = mesh.add_vertex(points[i])
        mesh.set_normal(vh, normals[i])
    
    om.write_mesh(out_path, mesh, vertex_normal=True)
    return  

data_path = '../data/test3/target.ply'
out_path = '../data/test3/target_n.ply'
estimate_normals(data_path, out_path)