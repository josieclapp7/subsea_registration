'''
Author: Josie Clapp
working_icp_tree.py
File to register entire scan position from a given file
Notes:
    - Hard coded filepath, maybe add the select file and the converting file
    - Maybe add to export the final file    
'''
import os
import copy
import random
from collections import Counter, defaultdict
import numpy as np
import open3d as o3d
import time
import winsound

rmse_list = []

def parse_angle(angle_str):
    '''
    Turns a string of angle into float number
    Returns float angle number
    '''
    if angle_str.startswith("N"):
        return -float(angle_str[1:])
    else:
        return float(angle_str)
    
def draw_registration_result(source, target, transformation):
    '''Visualize the registration result, with blue and yellow'''
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def read_point_clouds_dict(pcd_folder):
    '''
    Read pcd filenames and metadata into initial dict (keys are order of read into dict)
    Returns dictionary of point clouds {filename: pan, tilt}
    '''
    point_clouds = {}
    i = 1
    for filename in os.listdir(pcd_folder):
        full = os.path.join(pcd_folder, filename)
        if os.path.isfile(full) and filename.lower().endswith(".pcd"):
            parts = filename.split("_")
            if len(parts) >= 3:
                info = parts[2]
                dash_parts = info.split("-")
                if len(dash_parts) >= 3:
                    key = f"{dash_parts[0]}_point_cloud_{i}"
                    point_clouds[key] = {"filename": filename, "Pan": dash_parts[1], "Tilt": dash_parts[2]}
                    i += 1
    return point_clouds

def build_combined_cloud(member_filenames, original_clouds, final_transforms):
    '''
    Create a combined (temporary) cloud from the provided filenames
    original_clouds: dict {filename} of original full-res clouds
    final_transforms: dict {filename: transform} of transform associated with original cloud
    Returns combined point cloud of all given clouds
    '''
    # print("Build combined cloud here")
    pieces = []
    for filename in member_filenames:
        pc = copy.deepcopy(original_clouds[filename])
        T = final_transforms.get(filename, np.eye(4))
        pc.transform(T)
        pieces.append(pc)
    if not pieces:
        return None
    combined = pieces[0]
    if len(pieces) > 1:
        for p in pieces[1:]:
            combined += p
    return combined

#@@@@@@@@@@@@@@@@@@@@@@@@@@ Dont need anymore
def refine_registration(source, target, init_transformation, voxel_size, plane=True):
    '''Runs Point-to-plane or point-to-point ICP on given clouds'''
    distance_threshold = voxel_size * 0.4
    # print(":: Refinement with ICP (point-to-plane), dist_thresh = %.3f" % distance_threshold)
    if plane == True:
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target,
            distance_threshold,
            init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
    else:
        result_icp = o3d.pipelines.registration.registration_icp(
            source, target,
            distance_threshold,
            init_transformation,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )
    return result_icp

def execute_icp(cloud1, cloud2, voxel_size, show=True, plane=True):
    '''Executes point-to-plane or point-to-point registration '''
    print(f"\tExecute ICP -- With Show: {show}")
    # No global registration, starts with the indentity matrix
    trans_init = np.eye(4)
    
    # ICP     
    distance_threshold = voxel_size * 0.4

    if plane == True:
    # point-to-plane
        radius_normal = voxel_size * 2
        cloud1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        cloud2.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        
        result_icp = o3d.pipelines.registration.registration_icp(
            cloud1, cloud2,
            distance_threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
    else:
    # point-to-plane
        result_icp = o3d.pipelines.registration.registration_icp(
            cloud1, cloud2,
            distance_threshold,
            trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        )

    if show:
        print(f'RMSE of this registration was {result_icp.inlier_rmse}')
        draw_registration_result(cloud1, cloud2, result_icp.transformation)
    rmse_list.append(result_icp.inlier_rmse)
    return np.linalg.inv(result_icp.transformation)

def compute_transform_between(cloud_ref, cloud_mov, voxel_size, show=True):
    '''
    Calls your gicp.register function and returns a 4x4 matrix mapping cloud_mov -> cloud_ref.
    Expects gicp.execute_global_and_icp(...) to return a numpy 4x4 matrix.
    '''
    print(f"\tcompute transformation between -- With Show: {show}")
    T = execute_icp(cloud_ref, cloud_mov, voxel_size, show)
    # tolerate both numpy arrays and Open3D style results
    if isinstance(T, np.ndarray) and T.shape == (4, 4):
        return T
    # if gicp was not updated and returns a point cloud (old behaviour), raise informative error
    raise RuntimeError(
        "gicp.execute_global_and_icp did not return a 4x4 transform. "
    )
    
def register_pairs_entries(entry_list, reverse=False):
    '''Takes list of entries (each has 'members','Pan','Tilt'), pairs them and returns new list of merged entries.'''
    new = []
    items = entry_list[::-1] if reverse else entry_list[:]
    i = 0
    while i <= len(items) - 2:
        e1 = items[i]
        e2 = items[i + 1]
        print(f"Pair-registering groups: {e1['id']}  <---  {e2['id']}")
        cloud_ref = build_combined_cloud(e1["members"], original_clouds, final_transforms)
        cloud_mov = build_combined_cloud(e2["members"], original_clouds, final_transforms)
        T_mov2ref = compute_transform_between(cloud_ref, cloud_mov, voxel_size, show=True)
        # rmse_list.append(T_mov2ref.inlier_rmse)
        # update transforms
        for fname in e2["members"]:
            final_transforms[fname] = T_mov2ref @ final_transforms[fname]
        # create merged entry
        merged = {
            "id": f"{e1['id']}_plus_{e2['id']}",
            "members": e1["members"] + e2["members"],
            "Pan": f"{e1['Pan']},{e2['Pan']}",
            "Tilt": f"{e1['Tilt']},{e2['Tilt']}"
        }
        new.append(merged)
        i += 2
    # handle odd leftover
    if i == len(items) - 1:
        leftover = items[-1]
        print(f"Carrying leftover entry forward: {leftover['id']}")
        new.append(leftover)
    # if we reversed at top, reverse back to original ordering for next round
    return new[::-1] if reverse else new

def color_from_name(name):
    rnd = random.Random(hash(name) & 0xffffffff)
    return [rnd.random(), rnd.random(), rnd.random()]

# -------- CONFIG --------------------------------------------------------------------------------------------
e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1447/SP33"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1601/SP411"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1580/Jumper 001-selected/SP01"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1601/SP414"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1580/Jumper 001-selected/SP01"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1601/SP412"


# e57_SP_folder = "C:\Users\josie\Documents\_Senior Project\Fake Data\Dirsig Data\pcd\Transformed"

pcd_SP_folder = os.path.join(e57_SP_folder, "pcd_files")
# pcd_SP_folder = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/Transformed2"

voxel_size = 0.015  # for downsampling / FPFH radius factors @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Make a global var
rmse_list = []
# ------------------------------------------------------------------------------------------------------------

print("Loading original point clouds...")
point_clouds = read_point_clouds_dict(pcd_SP_folder) 
if not point_clouds:
    raise SystemExit(f'No .pcd files found in {pcd_SP_folder}')
# print("ORIGINAL CLOUD DICT")
# print(point_clouds)

start = time.time()
# Keep original full-res clouds in memory keyed by filename (filename: cloud(filename)) 
original_clouds = {}
for key, value in point_clouds.items():
    fn = value["filename"]
    fullpath = os.path.join(pcd_SP_folder, fn)
    original_clouds[fn] = o3d.io.read_point_cloud(fullpath)

# initial transforms: every cloud starts with identity
final_transforms = {fn: np.eye(4) for fn in original_clouds.keys()}

# each entry is: {"members": [filenames], "Pan": ..., "Tilt": ...}
print("Preparing sorted order...")
tilt_counts = Counter(info["Tilt"] for info in point_clouds.values())
most_common_tilt, _ = tilt_counts.most_common(1)[0]
print(f"Most common tilt is {most_common_tilt} degrees")

sorted_point_clouds = dict(
    sorted(
        point_clouds.items(),
        key=lambda item: (
            0 if item[1]["Tilt"] == most_common_tilt else 1,
            item[1]["Tilt"],
            parse_angle(item[1]["Pan"])
        )
    )
)

# print("SORTED CLOUDS")
# print(sorted_point_clouds)

# Convert sorted_point_clouds into entries (list of dict corresponding to the sorted point clouds)
# Each entry is arow of information about a scan (or group of scans later on).'''
entries = []
for name, info in sorted_point_clouds.items():
    entries.append({
        "id": name,
        "members": [info["filename"]],
        "Pan": info["Pan"],
        "Tilt": info["Tilt"]
    })

# ----------------- Stage 1: handle duplicates (same Pan & Tilt adjacent) ---------------------------------------
print("\nSTAGE 1 — Identifying adjacent duplicates and registering them (non-destructive)")
i = 0
new_entries = []
while i < len(entries):
    if i < len(entries) - 1:
        a = entries[i]
        b = entries[i + 1]
        if a["Pan"] == b["Pan"] and a["Tilt"] == b["Tilt"]:
            # register b → a, update transforms for all members in b
            print(f"Registering duplicate pair: {a['members']}  <---  {b['members']}  (Pan {a['Pan']}, Tilt {a['Tilt']})")
            cloud_a = build_combined_cloud(a["members"], original_clouds, final_transforms)
            cloud_b = build_combined_cloud(b["members"], original_clouds, final_transforms)
            T_b2a = compute_transform_between(cloud_a, cloud_b, voxel_size, show=True)
            # rmse_list.append(T_b2a.inlier_rmse)
            # update global transforms for every member in b:
            for fname in b["members"]:
                final_transforms[fname] = T_b2a @ final_transforms[fname]
            # merge groups logically, make a single entry
            merged_members = a["members"] + b["members"]
            merged_entry = {"id": f"{a['id']}_plus_{b['id']}", "members": merged_members, "Pan": a["Pan"], "Tilt": a["Tilt"]}
            new_entries.append(merged_entry)
            i += 2
            continue
    new_entries.append(entries[i])
    i += 1

entries = new_entries
print(f"Entries after Stage 1: {len(entries)}")

# ----------------- Stage 2: attach additional tilts to base-most-common-tilt per Pan ----------------------------------
print("\nSTAGE 2 — Register additional tilts onto base (most common tilt) per Pan")
# group entries by pan; identify base (Tilt == most_common_tilt)
entries_by_pan = defaultdict(list)
for e in entries:
    entries_by_pan[e["Pan"]].append(e)

pan_entries_after_attach = []
for pan, group in entries_by_pan.items():
    base_list = [g for g in group if g["Tilt"] == most_common_tilt]
    other_list = [g for g in group if g["Tilt"] != most_common_tilt]
    if not base_list:
        # no base, pick first as base
        base = group[0]
        rest = group[1:]
    else:
        base = base_list[0]
        rest = [g for g in group if g is not base]
    # sequentially register each additional entry to the base
    for additional in rest:
        print(f"Registering additional tilt {additional['members']} onto base {base['members']} (Pan {pan})")
        cloud_base = build_combined_cloud(base["members"], original_clouds, final_transforms)
        cloud_add = build_combined_cloud(additional["members"], original_clouds, final_transforms)
        T_add2base = compute_transform_between(cloud_base, cloud_add, voxel_size, show=True)
        # rmse_list.append(T_add2base.inlier_rmse)
        # update transforms for all members in 'add'
        for fname in additional["members"]:
            final_transforms[fname] = T_add2base @ final_transforms[fname]
        # merge logically
        base["members"].extend(additional["members"])
        # update base Tilt string to include appended tilts (for bookkeeping)
        base["Tilt"] = f"{base['Tilt']}, {additional['Tilt']}"
    pan_entries_after_attach.append(base)

# sort resulting pan entries by pan angle
additional_sort_entries = sorted(pan_entries_after_attach, key=lambda e: parse_angle(e["Pan"]))
print(f"Entries after Stage 2 (per-pan bases): {len(additional_sort_entries)}")

# ----------------- Stage 3: hierarchical binary tree registration (non-destructive) -------------
print("\nSTAGE 3 — Hierarchical binary registration until single scene remains")

# iterative binary pairing until a single entry remains
current = additional_sort_entries[:]
round_idx = 0
reverse_flag = False
while len(current) > 1:
    round_idx += 1
    print(f"\n--- Hierarchical round {round_idx} (reverse={reverse_flag}) with {len(current)} entries ---")
    current = register_pairs_entries(current, reverse=reverse_flag)
    reverse_flag = not reverse_flag

final_entry = current[0]
# print(f"\nFinal entry id: {final_entry['id']}, members count: {len(final_entry['members'])}")
print("Finished! ")

# ----------------- Final visualization: color each original cloud and display ------------
print("\nApplying final transforms to original clouds and coloring them for display...")
colored_clouds = []
# paint every cloud a random color
random.seed(42)
for fname, cloud in original_clouds.items():
    pc = copy.deepcopy(cloud)
    pc.transform(final_transforms[fname])
    pc.paint_uniform_color(color_from_name(fname))
    colored_clouds.append(pc)

end = time.time()
# show all colored clouds together
print(f' Time for voxel size {voxel_size} was {end-start} seconds, or {((end-start)/60)} minutes')
avg_rmse = np.mean(rmse_list)
std_rmse = np.std(rmse_list)
print(f'{rmse_list} \n average: {avg_rmse} \n stdev: {std_rmse}')

combined = colored_clouds[0]
for pc in colored_clouds[1:]:
    combined += pc

winsound.Beep(2500, 1000) 
with open("final_transforms_ICP_dirsig_data_2.txt", "w") as f:
    f.write("FINAL TRANSFORMS (cloud -> global)\n\n")
    for fname in sorted(final_transforms.keys()):
        T = final_transforms[fname]
        f.write(f"--- {fname} ---\n")
        f.write(np.array2string(T, formatter={'float_kind': lambda x: f"{x: .6f}"}))
        f.write("\n\n")


# Save the merged result
o3d.io.write_point_cloud("ICP_registration_1580_SP01.pcd", combined)
# o3d.io.write_point_cloud("ICP_registration_dirsig_data_2.pcd", combined)
o3d.visualization.draw_geometries(colored_clouds)

