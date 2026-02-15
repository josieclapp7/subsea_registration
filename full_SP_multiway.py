'''
Author: Josie Clapp
File to preform multiway registration on a full scan postition
NODE: Pose - 4x4 transformation matrix (nodes contain final aligned pose of each scan in global coords)
EDGE: constraints between two nodes (scans) - contains transformation matrix from ICP and information matrix (confidence in transition)
    -odometry edge: between spatial scans
    -loop closure edge: between non sequential scans (not really needed here?? there is no loop here)
    
Notes:
    -Hard coded e57 folder again
'''
import multiway_functions as mf
import os
import random
import open3d as o3d
import time
import winsound
import numpy as np
import copy

# -------- CONFIG --------------------------------------------------------------------------------------------
e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1447/SP33"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1601/SP411"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1580/Jumper 001-selected/SP01"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1601/SP414"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1580/Jumper 001-selected/SP02"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1601/SP412"


#convert the e57s to pcds
# convert_e57_to_pcd(e57_SP_folder)
voxel_size = 0.01
max_correspondence_distance_coarse = voxel_size * 15
max_correspondence_distance_fine = voxel_size * 5 
# ------------------------------------------------------------------------------------------------------------

# builds a list of pcd filepaths
print("Building List of Files...")
pcd_SP_folder = e57_SP_folder + '/pcd_files'
# pcd_SP_folder = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/Transformed2"
pcd_files = []
for filename in os.listdir(pcd_SP_folder):
    if os.path.isfile(os.path.join(pcd_SP_folder, pcd_SP_folder + "/" + filename)):
        parts = filename.split("_")
        if len(parts) >= 3:  # make sure there's a 2nd underscore
            pcd_files.append(pcd_SP_folder + "/" + filename)
        else: # skip QC files
            continue

pcd_filenames = [os.path.basename(f) for f in pcd_files]



#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@TRY TO NOT DOWNSAMPLE
point_cloud_down = mf.load_point_clouds_down(pcd_files, voxel_size)
original_clouds = [copy.deepcopy(pc) for pc in point_cloud_down]

start = time.time()

# Generate pose graph (pose_graph stores all transformatons required to align scans)
print("Full registration ...")
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Error) as cm: #change to .Debug if want full print statements
    pose_graph = mf.full_registration(point_cloud_down,
                                   max_correspondence_distance_coarse,
                                   max_correspondence_distance_fine)
    
# Optimize pose graph
print("Optimizing PoseGraph ...")
option = o3d.pipelines.registration.GlobalOptimizationOption(
    max_correspondence_distance=max_correspondence_distance_fine, #max distance for considering point matches
    edge_prune_threshold=0.25, # removes unreliable pairwise transformations#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@EXPERIMENT WITH THIS AS WELL
    reference_node=0) #fized scan to anchor the global alignment
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Error) as cm:
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(), #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@experiment with these too
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option)
    
# Transform each cloud in point_cloud_down to their determined postitions
transform_list = []
for point_id in range(len(point_cloud_down)):
    print(point_id)
    print(pose_graph.nodes[point_id].pose) #print transformation matrix
    point_cloud_down[point_id].transform(pose_graph.nodes[point_id].pose) #apply optimized pose in global space
    # point_cloud_down[point_id].paint_uniform_color(colors[point_id])

end = time.time()

# ----------------- Final visualization: color each original cloud and display ---------------------------
# assign random colors to each transformed downsampled scan
random.seed(42)
for pc_reg, pc_orig, fname in zip(point_cloud_down, original_clouds, pcd_filenames):
    color = mf.color_from_name(fname)
    pc_reg.paint_uniform_color(color)
    pc_orig.paint_uniform_color(color)

rmse, std_rmse, rmse_values = mf.compute_global_rmse(point_cloud_down, pose_graph, voxel_size * 0.4) #THis only compares the adjacent clouds

# print(f'Time: {(end-start)/60} minutes  \nvoxel_size: {voxel_size} \nmax_correspondence_distance_coarse: {max_correspondence_distance_coarse} ')
print(f'Time: {(end-start)/60}')
print(f'max_correspondence_distance_fine: {max_correspondence_distance_fine}')
print(rmse_values)

with open("final_transforms_dirsig_data_2.txt", "w") as f: #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    f.write("FINAL TRANSFORMS (cloud -> global)\n\n")
    for i, fname in enumerate(pcd_filenames):
        T = pose_graph.nodes[i].pose
        f.write(f"--- {fname} ---\n")
        f.write(
            np.array2string(
                T,
                formatter={'float_kind': lambda x: f"{x: .6f}"}
            )
        )
        f.write("\n\n")

winsound.Beep(2500, 1000) 

#combine to save out cloud
combined = point_cloud_down[0]
for pc in point_cloud_down[1:]:
    combined += pc

# o3d.visualization.draw_geometries(point_cloud_down)
print("Showing ORIGINAL alignment")
o3d.visualization.draw_geometries(original_clouds)

print("Showing REGISTERED alignment")
o3d.visualization.draw_geometries(point_cloud_down)
# o3d.visualization.draw_geometries([pcd_combined])
# o3d.io.write_point_cloud("Multiway_registration_1580_SP01.pcd", combined)#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!