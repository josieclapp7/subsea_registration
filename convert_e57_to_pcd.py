'''
Josie edited Wades original file to work without a GUI involved
'''
import pye57
import numpy as np
import open3d as o3d
import os

def convert_e57_to_pcd(folder_path):
    
    '''This is a function to import a point cloud in a E57 Format and convert it into the open3D o3d.geometry.PointCloud class.
        Then it exports the data as a PCD format.
        
        If you just want to import E57 data into Open3D for other processes you can remove the bit that exports the data as PCD.
        
        Subsea LiDAR data does not have color information so the intensity values are normalized and used as grayscale colors.
    
    '''
    pcd_folder = os.path.join(folder_path, "pcd_files")
    os.makedirs(pcd_folder, exist_ok=True)
    point_clouds = []

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(".e57"):  # only process .e57 files
            # Read the .e57 file
            e57 = pye57.E57(file_path)
            data = e57.read_scan(0, intensity=True, colors=False)
            points = np.column_stack((data['cartesianX'], data['cartesianY'], data['cartesianZ']))
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd_filename = os.path.splitext(filename)[0] + ".pcd"
            pcd_path = os.path.join(pcd_folder, pcd_filename)
            o3d.io.write_point_cloud(pcd_path, pcd)
            if 'intensity' in data:
                intensity = data['intensity']
                intensity_norm = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
                colors = np.column_stack([intensity_norm, intensity_norm, intensity_norm])
                pcd.colors = o3d.utility.Vector3dVector(colors)

            point_clouds.append(pcd)

    # if point_clouds:
    #     o3d.visualization.draw_geometries(point_clouds)
    # else:
    #     print("No .pcd files found in the folder.")

# # Usage
