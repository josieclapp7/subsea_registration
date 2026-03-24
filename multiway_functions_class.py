'''
Functions needed to execute multiway registration
'''
import os
import copy
import time
import random
import numpy as np
import open3d as o3d
from dataclasses import dataclass


@dataclass
class RegistrationConfig:
    '''
    Holds tuneable parameters for multiway registration
    '''
    voxel_size: float = 0.005
    edge_prune_threshold: float = 0.25  # removes unreliable pairwise transformations 
    reference_node: int = 0             # fixed scan to anchor the global alignment
    use_point_to_plane: bool = True     # use point-to-plane? y/n

    #@@@@@@@@@@@@@@@@@@@ do I want a __post_init___?? (pg 35)

    @property
    def coarse_distance(self) -> float:
        return self.voxel_size * 15
    
    @property
    def fine_distance(self) -> float:
        return self.voxel_size * 5
    
    @property
    def rmse_threshold(self) -> float:
        return self.voxel_size * 0.4
    
    @property
    def normal_radius(self) -> float:
        return self.voxel_size * 2
    
@dataclass
class ScanPair:
    '''
    Results of a single pairwise ICP registration between two scans
    '''
    source_id: int
    target_id: int
    transformation: np.ndarray
    information: np.ndarray
    inlier_rmse: float
    is_adjacent: bool

@dataclass
class RegistrationResult:
    '''
    Output container for completed multiway registration run
    '''
    filenames: list
    pose_graph: object
    transformed_clouds: list
    original_clouds: list
    rmse_values: list
    avg_rmse: float
    std_rmse: float
    elapsed_seconds: float

    def print_summary(self):
        print(f'\nTime: {self.elapsed_seconds/60:.4f} minutes')
        print(f'avg RMSE: {self.avg_rmse}')
        print(f'std RMSE: {self.std_rmse}')
        print(f'rmse_values: {self.rmse_values}')

    def print_transforms(self):
        for i in range(len(self.filenames)):
            print(i)
            print(self.pose_graph.nodes[i].pose)

    def save_transforms(self, filepath: str):
        with open(filepath, "w") as f:
            f.write("FINAL TRANSFORMS (cloud -> global)\n\n")
            for i, fname in enumerate(self.filenames):
                T = self.pose_graph.nodes[i].pose
                f.write(f"--- {fname} ---\n")
                f.write(
                    np.array2string(
                        T,
                        formatter={'float_kind': lambda x: f"{x: .6f}"}
                    )
                )
                f.write("\n\n")
        print(f"Transforms saved to: {filepath}")

    def colorize(self):
        '''Paint each cloud (registered and original) with name-seeded color'''
        for pc_reg, pc_orig, fname in zip(self.transformed_clouds, self.original_clouds, self.filenames):
            color = color_from_name(fname)
            pc_reg.paint_uniform_color(color)
            pc_orig.paint_uniform_color(color)

    def combined_cloud(self) -> o3d.geometry.PointCloud:
        '''Merge all registered clouds into one for export/visualization'''
        combined = copy.deepcopy(self.transformed_clouds[0])
        for pc in self.transformed_clouds[1:]:
            combined += pc
        return combined
    
    def visualize_original(self):
        print("Showing ORIGINAL alignment")
        o3d.visualization.draw_geometries(self.original_clouds)

    def visualize_registered(self):
        print("Showing REGISTERED alignment")
        o3d.visualization.draw_geometries(self.transformed_clouds)

class PointCloudLoader:
    '''
    Loads the point clouds so it is ready for ICP registration
    '''
    def __init__(self, pcd_folder:str, config: RegistrationConfig):
        self.pcd_folder = pcd_folder
        self.config = config
        self.filepaths: list = []
        self.filenames: list = []
        self.clouds: list = []

    def discover(self) -> 'PointCloudLoader':
        '''Scan the folder and collect valid .pcd file paths'''
        print("Building List of Files...")
        for filename in os.listdir(self.pcd_folder):
            full_path = os.path.join(self.pcd_folder, filename)
            if not os.path.isfile(full_path):
                continue
            parts = filename.split("_")
            if len(parts) >= 3: #make sure there is a 2nd underscore
                self.filepaths.append(self.pcd_folder + "/" + filename)
            else:
                continue # skip QC files
        self.filenames = [os.path.basename(f) for f in self.filepaths]
        print(f'    Found {len(self.filepaths)} scan file(s).')
        return self #allows chaining
    
    def load(self) -> 'PointCloudLoader':
        '''Downsample each file and estimate normals'''
        self.clouds = load_point_clouds_down(self.filepaths, self.config)
        return self
    
class MultiwaySolver:
    '''
    Executes multiway point cloud registration
    '''
    def __init__(self, config: RegistrationConfig):
        self.config = config
        self.scan_pairs: list = []

    def run(self, clouds: list, filenames: list, original_clouds: list) -> RegistrationResult:
        ''' Full registration pipeline'''
        cfg = self.config
        start = time.time()

        # Build Pose Graph
        print("Full registration... ")
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error): # change to .Debug for full output
            pose_graph = self._full_registration (clouds)

        # Optimize Pose Graph
        print("Optimizing PoseGraph... ")
        option = o3d.pipelines.registration.GlobalOptimizationOption(
            max_correspondence_distance=cfg.fine_distance,  # max distance for considering point matches
            edge_prune_threshold=cfg.edge_prune_threshold,  # removes unreliable pairwise transformations
            reference_node=cfg.reference_node)              # fixed scan to anchor the global alignment
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Error):
            o3d.pipelines.registration.global_optimization(
                pose_graph,
                o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),  # EXPERIMENT WITH THIS
                o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
                option)
 
        # --- Apply optimized poses ---
        transformed_clouds = []
        for point_id in range(len(clouds)):
            print(point_id)
            print(pose_graph.nodes[point_id].pose)          # print transformation matrix
            pc_copy = copy.deepcopy(clouds[point_id])
            pc_copy.transform(pose_graph.nodes[point_id].pose)  # apply optimized pose in global space
            transformed_clouds.append(pc_copy)
 
        end = time.time()

        # Compute RMSE
        avg_rmse, std_rmse, rmse_values = compute_global_rmse(
            clouds, pose_graph, threshold=cfg.rmse_threshold
        )
 
        return RegistrationResult(
            filenames=filenames,
            pose_graph=pose_graph,
            transformed_clouds=transformed_clouds,
            original_clouds=list(original_clouds),
            rmse_values=rmse_values,
            avg_rmse=avg_rmse,
            std_rmse=std_rmse,
            elapsed_seconds=end - start,
        )


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

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
