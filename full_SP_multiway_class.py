'''
Notes:
    - Choose if the e57s need to be converted or if they are already pcds
'''

import multiway_functions_class as mf
import copy
import winsound

# -------- CONFIG --------------------------------------------------------------------------------------------
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1447/SP33"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1601/SP411"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1580/Jumper 001-selected/SP01"
e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1601/SP414"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1580/Jumper 001-selected/SP02"
# e57_SP_folder = "C:/Users/josie/Documents/_Senior Project/senior_project/full_scan_cleaned/1601/SP412"


config = mf.RegistrationConfig(voxel_size=0.005, edge_prune_threshold=0.25, reference_node=0, use_point_to_plane=True)
OUTPUT_TRANSFORMS_FILE = "final_transforms_1601.txt"
# ------------------------------------------------------------------------------------------------------------

# Convert the e57s to pcds
# convert_e57_to_pcd(e57_SP_folder)
 
# Load point clouds 
pcd_SP_folder = e57_SP_folder + '/pcd_files'

loader = mf.PointCloudLoader(pcd_SP_folder, config)
loader.discover().load()

original_clouds = [copy.deepcopy(pc) for pc in loader.clouds]

# Register
solver = mf.MultiwaySolver(config)
result = solver.run(clouds=loader.clouds, filenames=loader.filenames, original_clouds=original_clouds)

# Output
result.print_summary()
result.print_transforms()
# result.save_transforms(OUTPUT_TRANSFORMS_FILE)

winsound.Beep(2500, 1000)

# Assign Colors and vizualize
result.colorize()
result.visualize_original()
result.visualize_registered()