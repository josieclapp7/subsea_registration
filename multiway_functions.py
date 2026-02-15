'''
Functions needed to execute multiway registration
'''

import open3d as o3d
import copy
import numpy as np
import copy
import random

def load_point_clouds_down(pcd_files, voxel_size = 0.0):
    '''
    pcd_file: a list of filenames
    Downsamples point cloud according to voxel size, computes normals for the ICP registration
    Returns list of preprocessed point clouds
    '''
    point_clouds = []
    for path in pcd_files:
        cloud = o3d.io.read_point_cloud(path)
        # downsample
        pcd_down = cloud.voxel_down_sample(voxel_size=voxel_size)
        # compute normals
        radius_normal = voxel_size * 2
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
        )
        point_clouds.append(pcd_down)
        
    return point_clouds
    

def pairwise_registration(source, target, max_correspondence_distance_coarse, max_correspondence_distance_fine, plane=True):
    '''
    Coarse and fine icp registration
    Returns transform and its corresponding information
    '''
    if plane == True:
        print("Apply point-to-plane ICP")
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
    else:
        print("Apply Point to Point ICp")
        icp_coarse = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_coarse, np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        icp_fine = o3d.pipelines.registration.registration_icp(
            source, target, max_correspondence_distance_fine,
            icp_coarse.transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
    transformation_icp = icp_fine.transformation
    information_icp = o3d.pipelines.registration.get_information_matrix_from_point_clouds(
        source, target, max_correspondence_distance_fine,
        icp_fine.transformation)
    print("ICP inlier RMSE:", icp_fine.inlier_rmse)
    return transformation_icp, information_icp


def full_registration(pcds, max_correspondence_distance_coarse,
                      max_correspondence_distance_fine):
    '''Undergoes the full registration with pose graphs'''
    pose_graph = o3d.pipelines.registration.PoseGraph()
    odometry = np.identity(4)
    pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(odometry))
    n_pcds = len(pcds)
    
    #LOOP OVER ALL PAIRS OF POINT CLOUDS! (MAYBE CHANGE TO CHECK FOR PAN AND TILT INFO)
    for source_id in range(n_pcds):
        for target_id in range(source_id + 1, n_pcds):
            transformation_icp, information_icp = pairwise_registration(
                pcds[source_id], pcds[target_id], max_correspondence_distance_coarse, max_correspondence_distance_fine)
            print("Build o3d.pipelines.registration.PoseGraph")
            if target_id == source_id + 1:  # odometry (direct neighbors)
                odometry = np.dot(transformation_icp, odometry) #this is the cumulative transformation
                #add node and edge to the pose graph
                pose_graph.nodes.append(
                    o3d.pipelines.registration.PoseGraphNode(
                        np.linalg.inv(odometry)))
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=False))
            else:  # loop closure case (non-adjacent scans)
                pose_graph.edges.append(
                    o3d.pipelines.registration.PoseGraphEdge(source_id,
                                                             target_id,
                                                             transformation_icp,
                                                             information_icp,
                                                             uncertain=True))
    return pose_graph #contains nodes (scan poses) and edges (transformations and weights)

def compute_global_rmse(pcds_down, pose_graph, threshold=0.009, adjacent_only=True):
    '''
    Compute average RMSE and standard deviation between registered point clouds
    after pose graph optimization.
    pcds_down: List of downsampled point clouds that were used in the registration
    pose_graph: Optimized pose graph containing final transformations (as found by full_registration and optimized with global optimization)
    threshold: Distance threshold for inlier evaluation !!!!! This could effect things
    adjacent_only: if True, only compares adjacent clouds; if False, compares all pairs.
    '''
    transformed_pcds = []
    for point_id, pcd in enumerate(pcds_down):
        pcd_copy = copy.deepcopy(pcd)
        pcd_copy.transform(pose_graph.nodes[point_id].pose)
        transformed_pcds.append(pcd_copy)

    rmse_values = []

    n = len(transformed_pcds)
    for i in range(n - 1):  # stop before last element to avoid index error
        compare_range = [i + 1] if adjacent_only else range(i + 1, n)
        for j in compare_range:
            eval_result = o3d.pipelines.registration.evaluate_registration(
                transformed_pcds[i],
                transformed_pcds[j],
                threshold,
                np.eye(4)
            )
            rmse = eval_result.inlier_rmse
            print(f"RMSE between scan {i} and {j}: {rmse}")
            rmse_values.append(rmse)

    if rmse_values:
        avg_rmse = np.mean(rmse_values)
        std_rmse = np.std(rmse_values)
    else:
        avg_rmse = std_rmse = None

    print(f"\nAverage global RMSE after optimization: {avg_rmse}")
    print(f"Standard deviation of RMSE: {std_rmse}")

    return avg_rmse, std_rmse, rmse_values

def color_from_name(name):
    rnd = random.Random(hash(name) & 0xffffffff)
    return [rnd.random(), rnd.random(), rnd.random()]
