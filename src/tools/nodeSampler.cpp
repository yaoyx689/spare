//#pragma once
#include "nodeSampler.h"
#include "tools.h"
#include "tools/OmpHelper.h"
#include "geodesic/geodesic_algorithm_exact.h"
#include "nanoflann.h"

namespace svr
{
    //	Define helper functions
    static auto square = [](const Scalar argu) { return argu * argu; };
    static auto cube = [](const Scalar argu) { return argu * argu * argu; };
    static auto max = [](const Scalar lhs, const Scalar rhs) { return lhs > rhs ? lhs : rhs; };

    //------------------------------------------------------------------------
    //	Node Sampling based on geodesic distance metric
    //
    //	Note that this member function samples nodes along some axis.
    //	Each node is not covered by any other node. And distance between each
    //	pair of nodes is at least sampling radius.
    //------------------------------------------------------------------------
    // Local geodesic calculation
    Scalar nodeSampler::SampleAndConstuctAxis(Mesh &mesh, Scalar sampleRadiusRatio, sampleAxis axis)
    {
        //	Save numbers of vertex and edge
        m_meshVertexNum = mesh.n_vertices();
        m_meshEdgeNum = mesh.n_edges();
        m_mesh = & mesh;

        //	Calculate average edge length of bound mesh
        for (size_t i = 0; i < m_meshEdgeNum; ++i)
        {
            OpenMesh::EdgeHandle eh = mesh.edge_handle(i);
            Scalar edgeLen = mesh.calc_edge_length(eh);
            m_averageEdgeLen += edgeLen;
        }
        m_averageEdgeLen /= m_meshEdgeNum;

        //	Sampling radius is calculated as averageEdgeLen multiplied by sampleRadiusRatio
        m_sampleRadius = sampleRadiusRatio * m_averageEdgeLen;

        //	Reorder mesh vertex along axis
        std::vector<size_t> vertexReorderedAlongAxis(m_meshVertexNum);
        size_t vertexIdx = 0;
        std::generate(vertexReorderedAlongAxis.begin(), vertexReorderedAlongAxis.end(), [&vertexIdx]() -> size_t { return vertexIdx++; });
        std::sort(vertexReorderedAlongAxis.begin(), vertexReorderedAlongAxis.end(), [&mesh, axis](const size_t &lhs, const size_t &rhs) -> bool {
            size_t lhsIdx = lhs;
            size_t rhsIdx = rhs;
            OpenMesh::VertexHandle vhl = mesh.vertex_handle(lhsIdx);
            OpenMesh::VertexHandle vhr = mesh.vertex_handle(rhsIdx);
            Mesh::Point vl = mesh.point(vhl);
            Mesh::Point vr = mesh.point(vhr);
            return vl[axis] > vr[axis];
        });

        //	Sample nodes using radius of m_sampleRadius
        size_t firstVertexIdx = vertexReorderedAlongAxis[0];
        VertexNodeIdx.resize(m_meshVertexNum);
        VertexNodeIdx.setConstant(-1);
        VertexNodeIdx[firstVertexIdx] = 0;
        size_t cur_node_idx = 0;

        m_vertexGraph.resize(m_meshVertexNum);
        VectorX weight_sum = VectorX::Zero(m_meshVertexNum);

        for (auto &vertexIdx : vertexReorderedAlongAxis)
        {
            if(VertexNodeIdx[vertexIdx] < 0 && m_vertexGraph.at(vertexIdx).empty())
            {
                m_nodeContainer.emplace_back(cur_node_idx, vertexIdx);
                VertexNodeIdx[vertexIdx] = cur_node_idx;

                std::vector<size_t> neighbor_verts;
                geodesic::GeodesicAlgorithmExact geoalg(&mesh, vertexIdx, m_sampleRadius);
                geoalg.propagate(vertexIdx, neighbor_verts);
                for(size_t i = 0; i < neighbor_verts.size(); i++)
                {
                    int neighIdx = neighbor_verts[i];
                    Scalar geodist = mesh.data(mesh.vertex_handle(neighIdx)).geodesic_distance;
                    if(geodist < m_sampleRadius)
                    {
                        Scalar weight = std::pow(1-std::pow(geodist/m_sampleRadius, 2), 3);
                        m_vertexGraph.at(neighIdx).emplace(std::pair<int, Scalar>(cur_node_idx, weight));
                        weight_sum[neighIdx] += weight;
                    }
                }
                cur_node_idx++;
            }
        }

        m_nodeGraph.resize(cur_node_idx);
        for (auto &vertexIdx : vertexReorderedAlongAxis)
        {
            for(auto &node: m_vertexGraph[vertexIdx])
            {
                size_t nodeIdx = node.first;
                for(auto &neighNode: m_vertexGraph[vertexIdx])
                {
                    size_t neighNodeIdx = neighNode.first;
                    if(nodeIdx != neighNodeIdx)
                    {
                        m_nodeGraph.at(nodeIdx).emplace(std::pair<int, Scalar>(neighNodeIdx, 1.0));
                    }
                }
                m_vertexGraph.at(vertexIdx).at(nodeIdx) /= weight_sum[vertexIdx];
            }
        }
        return m_sampleRadius;
    }

    //------------------------------------------------------------------------
    //	Node Sampling based on geodesic distance metric
    //
    //	Note that this member function samples nodes along some axis.
    //	Each node is not covered by any other node. And distance between each
    //	pair of nodes is at least sampling radius.
    //------------------------------------------------------------------------
    // Local geodesic calculation
    Scalar nodeSampler::SampleAndConstuct(Mesh &mesh, Scalar sampleRadiusRatio, const Matrix3X & src_points)
    {
        //	Save numbers of vertex and edge
        m_meshVertexNum = mesh.n_vertices();
        m_meshEdgeNum = mesh.n_edges();
        m_mesh = & mesh;
        

        //	Calculate average edge length of bound mesh
        for (size_t i = 0; i < m_meshEdgeNum; ++i)
        {
            OpenMesh::EdgeHandle eh = mesh.edge_handle(i);
            Scalar edgeLen = mesh.calc_edge_length(eh);
            m_averageEdgeLen += edgeLen;
        }
        m_averageEdgeLen /= m_meshEdgeNum;

        //	Sampling radius is calculated as averageEdgeLen multiplied by sampleRadiusRatio
        m_sampleRadius = sampleRadiusRatio * m_averageEdgeLen;
        
        // de-mean
        Vector3 means = src_points.rowwise().mean();
        Matrix3X demean_src_points = src_points.colwise() - means;
        Matrix33 covariance = demean_src_points * demean_src_points.transpose();

        Eigen::SelfAdjointEigenSolver<Matrix33> eigensolver(covariance);
        Vector3 eigen_values = eigensolver.eigenvalues();
        int max_idx = 2;
        if(eigen_values[0] > eigen_values[max_idx])    max_idx = 0;
        if(eigen_values[1] > eigen_values[max_idx])    max_idx = 1;

        Vector3 main_axis = eigensolver.eigenvectors().col(max_idx);

        VectorX projection = demean_src_points.transpose() * main_axis;
        
        projection_indices.resize(projection.size());
        projection_indices.setLinSpaced(projection.size(), 0, projection.size() - 1);

        auto compare = [&projection](int i, int j) {
            return projection(i) < projection(j);
        };

        std::sort(projection_indices.data(), projection_indices.data() + projection_indices.size(), compare);


        //	Sample nodes using radius of m_sampleRadius
        size_t firstVertexIdx = projection_indices[0];
        VertexNodeIdx.resize(m_meshVertexNum);
        VertexNodeIdx.setConstant(-1);
        VertexNodeIdx[firstVertexIdx] = 0;
        size_t cur_node_idx = 0;

        m_vertexGraph.resize(m_meshVertexNum);
        VectorX weight_sum = VectorX::Zero(m_meshVertexNum);

        for(int idx = 0; idx < projection_indices.size(); idx++)
        {
            int vertexIdx = projection_indices[idx];
            if(VertexNodeIdx[vertexIdx] < 0 && m_vertexGraph.at(vertexIdx).empty())
            {
                m_nodeContainer.emplace_back(cur_node_idx, vertexIdx);
                VertexNodeIdx[vertexIdx] = cur_node_idx;

                std::vector<size_t> neighbor_verts;
                geodesic::GeodesicAlgorithmExact geoalg(&mesh, vertexIdx, m_sampleRadius);
                geoalg.propagate(vertexIdx, neighbor_verts);
                for(size_t i = 0; i < neighbor_verts.size(); i++)
                {
                    int neighIdx = neighbor_verts[i];
                    Scalar geodist = mesh.data(mesh.vertex_handle(neighIdx)).geodesic_distance;
                    if(geodist < m_sampleRadius)
                    {
                        Scalar weight = std::pow(1-std::pow(geodist/m_sampleRadius, 2), 3);
                        m_vertexGraph.at(neighIdx).emplace(std::pair<int, Scalar>(cur_node_idx, weight));
                        weight_sum[neighIdx] += weight;
                    }
                }
                cur_node_idx++;
            }
        }

        m_nodeGraph.resize(cur_node_idx);
        for(int idx = 0; idx < projection_indices.size(); idx++)
        {
            int vertexIdx = projection_indices[idx];
            for(auto &node: m_vertexGraph[vertexIdx])
            {
                size_t nodeIdx = node.first;
                for(auto &neighNode: m_vertexGraph[vertexIdx])
                {
                    size_t neighNodeIdx = neighNode.first;
                    if(nodeIdx != neighNodeIdx)
                    {
                        m_nodeGraph.at(nodeIdx).emplace(std::pair<int, Scalar>(neighNodeIdx, 1.0));
                    }
                }
                m_vertexGraph.at(vertexIdx).at(nodeIdx) /= weight_sum[vertexIdx];
            }
        }

        

        // // check neighbors 
        // for(int nodeIdx = 0; nodeIdx < cur_node_idx; nodeIdx++)
        // {
        //     int num_neighbors =  m_nodeGraph[nodeIdx].size();
        //     // for (auto &eachNeighbor : m_nodeGraph[nodeIdx])
        //     // {
        //     //     num_neighbors++;
        //     // }
        //     if(num_neighbors<2)
        //     {
                
        //         VectorX dists(cur_node_idx);
        //         #pragma omp parallel for
        //         for(int ni = 0; ni < cur_node_idx; ni++)
        //         {
        //             int vidx0 = getNodeVertexIdx(nodeIdx);
        //             int vidx1 = getNodeVertexIdx(ni);
        //             Scalar dist = (src_points.col(vidx0) - src_points.col(vidx1)).squaredNorm();
        //             dists[ni] = dist;
        //         }
        //         dists[nodeIdx] = 1e10;
        //         for(int k = 0; k < 6; k++)
        //         {
        //             int min_idx = -1;
        //             dists.minCoeff(&min_idx);
        //             m_nodeGraph.at(nodeIdx).emplace(std::pair<int, Scalar>(min_idx, 1.0));
        //             dists[min_idx] = 1e10;
        //         }
        //     }
        // }
        // Eigen::VectorXi vnn(cur_node_idx);
        // // check vertex neighbors 
        // for(int vidx = 0; vidx < cur_node_idx; vidx++)
        // {
        //     // std::cout << "vidx.neighbor_node = " << m_vertexGraph.at(vidx).size() << std::endl;
        //     vnn[vidx] = m_nodeGraph.at(vidx).size();
        // }
        // std::cout << "v neighbor max = " << vnn.maxCoeff() << " min = " << vnn.minCoeff() << std::endl;
        return m_sampleRadius;
    }


    Scalar nodeSampler::SampleAndConstuctForSrcPoints(Mesh &mesh, Scalar sampleRadiusRatio, const Matrix3X & src_points, const Eigen::MatrixXi & src_knn_indices)
    {
        //	Save numbers of vertex and edge
        m_meshVertexNum = src_points.cols();
        int knn_num_neighbor = src_knn_indices.rows(); 
        m_meshEdgeNum = m_meshVertexNum*knn_num_neighbor;
        m_mesh = & mesh;

        //	Calculate average edge length of bound mesh
        for (size_t i = 0; i < m_meshVertexNum; ++i)
        {
            for(size_t j = 0; j < knn_num_neighbor; j++)
            {
                Scalar edgeLen = (src_points.col(i) - src_points.col(src_knn_indices(j, i))).norm();
                m_averageEdgeLen += edgeLen;
            }
        }
        m_averageEdgeLen /= m_meshEdgeNum;


        //	Sampling radius is calculated as averageEdgeLen multiplied by sampleRadiusRatio
        m_sampleRadius = sampleRadiusRatio * m_averageEdgeLen;

        // de-mean
        Vector3 means = src_points.rowwise().mean();
        Matrix3X demean_src_points = src_points.colwise() - means;
        Matrix33 covariance = demean_src_points * demean_src_points.transpose();

        Eigen::SelfAdjointEigenSolver<Matrix33> eigensolver(covariance);
        Vector3 eigen_values = eigensolver.eigenvalues();
        int max_idx = 2;
        if(eigen_values[0] > eigen_values[max_idx])    max_idx = 0;
        if(eigen_values[1] > eigen_values[max_idx])    max_idx = 1;

        Vector3 main_axis = eigensolver.eigenvectors().col(max_idx);

        VectorX projection = demean_src_points.transpose() * main_axis;
        
        projection_indices.resize(projection.size());
        projection_indices.setLinSpaced(projection.size(), 0, projection.size() - 1);

        auto compare = [&projection](int i, int j) {
            return projection(i) < projection(j);
        };

        std::sort(projection_indices.data(), projection_indices.data() + projection_indices.size(), compare);


        //	Sample nodes using radius of m_sampleRadius
        size_t firstVertexIdx = projection_indices[0];
        VertexNodeIdx.resize(m_meshVertexNum);
        VertexNodeIdx.setConstant(-1);
        VertexNodeIdx[firstVertexIdx] = 0;
        size_t cur_node_idx = 0;

        m_vertexGraph.resize(m_meshVertexNum);
        VectorX weight_sum = VectorX::Zero(m_meshVertexNum);

        for(int idx = 0; idx < projection_indices.size(); idx++)
        {
            int vertexIdx = projection_indices[idx];
            if(VertexNodeIdx[vertexIdx] < 0 && m_vertexGraph.at(vertexIdx).empty())
            {
                m_nodeContainer.emplace_back(cur_node_idx, vertexIdx);
                VertexNodeIdx[vertexIdx] = cur_node_idx;
                
                std::queue<size_t> neighbor_verts;
                neighbor_verts.push(vertexIdx);
                Eigen::VectorXi visited(m_meshVertexNum);
                visited.setZero();
                visited[vertexIdx] = 1;
                while(!neighbor_verts.empty())
                {
                    size_t vidx = neighbor_verts.front();
                    neighbor_verts.pop();
                    Scalar dist = (src_points.col(vertexIdx) - src_points.col(vidx)).norm();

                    if(dist < m_sampleRadius)
                    {
                        Scalar weight = std::pow(1-std::pow(dist/m_sampleRadius, 2), 3);
                        m_vertexGraph.at(vidx).emplace(std::pair<int, Scalar>(cur_node_idx, weight));
                        weight_sum[vidx] += weight;
                        for(int j = 0; j < knn_num_neighbor; j++)
                        {
                            int neighbor_vidx = src_knn_indices(j, vidx);
                            if(visited[neighbor_vidx]==0)
                            {
                                neighbor_verts.push(neighbor_vidx);
                                visited[neighbor_vidx]= 1;
                            }
                        }
                    }      
                }

                cur_node_idx++;             
            }
        }

        m_nodeGraph.resize(cur_node_idx);
        for(int idx = 0; idx < projection_indices.size(); idx++)
        {
            int vertexIdx = projection_indices[idx];
            for(auto &node: m_vertexGraph[vertexIdx])
            {
                size_t nodeIdx = node.first;
                for(auto &neighNode: m_vertexGraph[vertexIdx])
                {
                    size_t neighNodeIdx = neighNode.first;
                    if(nodeIdx != neighNodeIdx)
                    {
                        m_nodeGraph.at(nodeIdx).emplace(std::pair<int, Scalar>(neighNodeIdx, 1.0));
                    }
                }
                m_vertexGraph.at(vertexIdx).at(nodeIdx) /= weight_sum[vertexIdx];
            }
        }

        // // check neighbors 
        // for(int nodeIdx = 0; nodeIdx < cur_node_idx; nodeIdx++)
        // {
        //     int num_neighbors = 0;
        //     for (auto &eachNeighbor : m_nodeGraph[nodeIdx])
        //     {
        //         num_neighbors++;
        //     }
        //     if(num_neighbors==0)
        //     {
                
        //         VectorX dists(cur_node_idx);
        //         #pragma omp parallel for
        //         for(int ni = 0; ni < cur_node_idx; ni++)
        //         {
        //             int vidx0 = getNodeVertexIdx(nodeIdx);
        //             int vidx1 = getNodeNeighborSize(ni);
        //             Scalar dist = (src_points.col(vidx0) - src_points.col(vidx1)).squaredNorm();
        //             dists[ni] = dist;
        //         }
        //         dists[nodeIdx] = 1e10;
        //         int min_idx = -1;
        //         dists.minCoeff(&min_idx);
        //         m_nodeGraph.at(nodeIdx).emplace(std::pair<int, Scalar>(min_idx, 1.0));
        //     }
        // }
        return m_sampleRadius;
    }


    Scalar nodeSampler::SampleAndConstuctFPS(Mesh &mesh, Scalar sampleRadiusRatio, const Matrix3X & src_points, const Eigen::MatrixXi & src_knn_indices, int num_vn, int num_nn)
    {
        m_meshVertexNum = src_points.cols();
        int knn_num_neighbor = src_knn_indices.rows(); 
        m_meshEdgeNum = m_meshVertexNum*knn_num_neighbor;
        m_mesh = & mesh;

        //	Calculate average edge length of bound mesh
        for (size_t i = 0; i < m_meshVertexNum; ++i)
        {
            for(size_t j = 0; j < knn_num_neighbor; j++)
            {
                Scalar edgeLen = (src_points.col(i) - src_points.col(src_knn_indices(j, i))).norm();
                m_averageEdgeLen += edgeLen;
            }
        }
        m_averageEdgeLen /= m_meshEdgeNum;

        //	Sampling radius is calculated as averageEdgeLen multiplied by sampleRadiusRatio
        m_sampleRadius = sampleRadiusRatio * m_averageEdgeLen;

        // start points
        size_t startIndex = 0;
        // FPS to get sampling points in align term
        VectorX minDistances(m_meshVertexNum);
        minDistances.setConstant(std::numeric_limits<Scalar>::max());
        minDistances[startIndex] = 0;
        m_nodeContainer.emplace_back(startIndex, 0);
    
        Scalar minimal_dist = 1e10;
        int cur_node_idx = 1;
        // repeat select farthest points
        while (minimal_dist > m_sampleRadius) {
            // calculate the distance between each point with the sampling points set.         
            #pragma omp parallel for
            for (size_t i = 0; i < m_meshVertexNum; ++i) {
                if(i==startIndex)
                    continue;
                
                Scalar dist = (src_points.col(startIndex) - src_points.col(i)).norm();
                if(dist < minDistances[i])
                    minDistances[i] = dist;
            }

            // choose farthest point
            int maxDistanceIndex;
            minimal_dist = minDistances.maxCoeff(&maxDistanceIndex);
            minDistances[maxDistanceIndex] = 0;

            // add the farthest point into the sampling points set.
            startIndex = maxDistanceIndex;
            m_nodeContainer.emplace_back(cur_node_idx, startIndex);
            cur_node_idx++;
        }

        Matrix3X node_positions(3, cur_node_idx);
        for(int i = 0; i < cur_node_idx; i++)
        {
            int vidx = getNodeVertexIdx(i);
            node_positions.col(i) = src_points.col(vidx);
        }
        KDtree node_tree(node_positions);

        // For each vertex, find num_vn-closest nodes
        m_vertexGraph.resize(m_meshVertexNum);
        VectorX weight_sum(m_meshVertexNum);
        weight_sum.setZero();
        #pragma omp parallel for 
        for(int vidx = 0; vidx < m_meshVertexNum; vidx++)
        {
            std::vector<int> out_indices(num_vn);
            std::vector<Scalar> out_dists(num_vn);
            node_tree.query(src_points.col(vidx).data(), num_vn, out_indices.data(), out_dists.data());
            for(int k = 0; k < num_vn; k++)
            {
                Scalar weight = std::pow(1-std::pow(out_dists[k]/m_sampleRadius, 2), 3);
                m_vertexGraph.at(vidx).emplace(std::pair<int, Scalar>(out_indices[k], weight));
                weight_sum[vidx] += weight;
            }
        }

        #pragma omp parallel for 
        for(int vidx = 0; vidx < m_meshVertexNum; vidx++)
        {
            for(auto &neighNode: m_vertexGraph[vidx])
            {
                size_t neighNodeIdx = neighNode.first;
                m_vertexGraph.at(vidx).at(neighNodeIdx) /= weight_sum[vidx];
            }
        }


        // For each node, find num_nn-closest nodes
        m_nodeGraph.resize(cur_node_idx);
        #pragma omp parallel for 
        for(int nidx = 0; nidx < cur_node_idx; nidx++)
        {
            std::vector<int> out_indices(num_nn+1);
            std::vector<Scalar> out_dists(num_nn+1);
            int vidx = getNodeVertexIdx(nidx);
            node_tree.query(src_points.col(vidx).data(), num_nn+1, out_indices.data(), out_dists.data());
            for(int k = 0; k < num_nn; k++)
            {
                m_nodeGraph.at(nidx).emplace(std::pair<int, Scalar>(out_indices[k+1], 1.0));
            }
        }        
        return m_sampleRadius;
    }


    


    void nodeSampler::initWeight(RowMajorSparseMatrix& matPV, VectorX & matP, RowMajorSparseMatrix& matB, VectorX& matD, VectorX& smoothw)
    {
        Timer time;
        std::vector<Eigen::Triplet<Scalar>> coeff;
        matP.setZero();
        Eigen::VectorXi nonzero_num = Eigen::VectorXi::Zero(m_mesh->n_vertices());
        // data coeff
        for (size_t vertexIdx = 0; vertexIdx < m_meshVertexNum; ++vertexIdx)
        {
            Mesh::Point vi = m_mesh->point(m_mesh->vertex_handle(vertexIdx));
            for (auto &eachNeighbor : m_vertexGraph[vertexIdx])
            {
                size_t nodeIdx = eachNeighbor.first;
                Scalar weight = m_vertexGraph.at(vertexIdx).at(nodeIdx);
                Mesh::Point pj = m_mesh->point(m_mesh->vertex_handle(getNodeVertexIdx(nodeIdx)));

				for (int k = 0; k < 3; k++)
				{
					coeff.push_back(Eigen::Triplet<Scalar>(3 * vertexIdx + k, nodeIdx * 12 + k, weight * (vi[0] - pj[0])));
					coeff.push_back(Eigen::Triplet<Scalar>(3 * vertexIdx + k, nodeIdx * 12 + k + 3, weight * (vi[1] - pj[1])));
					coeff.push_back(Eigen::Triplet<Scalar>(3 * vertexIdx + k, nodeIdx * 12 + k + 6, weight * (vi[2] - pj[2])));
					coeff.push_back(Eigen::Triplet<Scalar>(3 * vertexIdx + k, nodeIdx * 12 + k + 9, weight * 1.0));
				}
                
				matP[vertexIdx * 3] += weight * pj[0];
				matP[vertexIdx * 3 + 1] += weight * pj[1];
				matP[vertexIdx * 3 + 2] += weight * pj[2];
            }
            nonzero_num[vertexIdx] = m_vertexGraph[vertexIdx].size();
        }
        matPV.setFromTriplets(coeff.begin(), coeff.end());


        // smooth coeff
        coeff.clear();
        int max_edge_num = nodeSize() * (nodeSize()-1);
		matB.resize(max_edge_num * 3, 12 * nodeSize());
		matD.resize(max_edge_num * 3);
        smoothw.resize(max_edge_num*3);
		smoothw.setZero();
        matD.setZero();
        int edge_id = 0;

        for (size_t nodeIdx = 0; nodeIdx < m_nodeContainer.size(); ++nodeIdx)
        {
            size_t vIdx0 = getNodeVertexIdx(nodeIdx);
            Mesh::VertexHandle vh0 = m_mesh->vertex_handle(vIdx0);
            Mesh::Point v0 = m_mesh->point(vh0);

            for (auto &eachNeighbor : m_nodeGraph[nodeIdx])
            {
                size_t neighborIdx = eachNeighbor.first;
                size_t vIdx1 = getNodeVertexIdx(neighborIdx);
                Mesh::Point v1 = m_mesh->point(m_mesh->vertex_handle(vIdx1));
                Mesh::Point dv = v0 - v1;
                int k = edge_id;

				for (int t = 0; t < 3; t++)
				{
					coeff.push_back(Eigen::Triplet<Scalar>(k * 3 + t, neighborIdx * 12 + t, dv[0]));
					coeff.push_back(Eigen::Triplet<Scalar>(k * 3 + t, neighborIdx * 12 + t + 3, dv[1]));
					coeff.push_back(Eigen::Triplet<Scalar>(k * 3 + t, neighborIdx * 12 + t + 6, dv[2]));
					coeff.push_back(Eigen::Triplet<Scalar>(k * 3 + t, neighborIdx * 12 + t + 9, 1.0));
					coeff.push_back(Eigen::Triplet<Scalar>(k * 3 + t, nodeIdx * 12 + t + 9, -1.0));
				}

                Scalar dist = dv.norm();
                if(dist > 0)
                {
					smoothw[k * 3] = smoothw[k * 3 + 1] = smoothw[k * 3 + 2] = 1.0 / dist;
                }
                else
                {
					//smoothw[k * 3] = 0.0;
                    std::cout << "node repeat";
                    exit(1);
                }
				matD[k * 3] = dv[0];
				matD[k * 3 + 1] = dv[1];
				matD[k * 3 + 2] = dv[2];
                edge_id++;
            }
        }
        matB.setFromTriplets(coeff.begin(), coeff.end());
        matD.conservativeResize(edge_id*3);
        matB.conservativeResize(edge_id*3, matPV.cols());
        smoothw.conservativeResize(edge_id*3);
        smoothw *= edge_id/(smoothw.sum()/3.0);
    }


    void nodeSampler::print_nodes(Mesh & mesh, std::string file_path)
    {
        std::string namev = file_path + "nodes.obj";
        std::ofstream out1(namev);
        for (size_t i = 0; i < m_nodeContainer.size(); i++)
        {
            int vexid = m_nodeContainer[i].second;
            out1 << "v " << mesh.point(mesh.vertex_handle(vexid))[0] << " " << mesh.point(mesh.vertex_handle(vexid))[1]
                << " " << mesh.point(mesh.vertex_handle(vexid))[2] << std::endl;
        }
        Eigen::VectorXi nonzero_num = Eigen::VectorXi::Zero(m_nodeContainer.size());
        for (size_t nodeIdx = 0; nodeIdx < m_nodeContainer.size(); ++nodeIdx)
        {
            for (auto &eachNeighbor : m_nodeGraph[nodeIdx])
            {
                size_t neighborIdx = eachNeighbor.first;
                out1 << "l " << nodeIdx+1 << " " << neighborIdx+1 << std::endl;
            }
            nonzero_num[nodeIdx] = m_nodeGraph[nodeIdx].size();
        }
        std::cout << "node neighbor min = " << nonzero_num.minCoeff() << " max = "
                  << nonzero_num.maxCoeff() << " average = " << nonzero_num.mean() << std::endl;
        out1.close();
    }
}
