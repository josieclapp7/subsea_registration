'''
Apply pre determined transforms to point cloud data
'''

import open3d as o3d
import numpy as np

def apply_translation(input_cloud, output_cloud, T):
    pcd = o3d.io.read_point_cloud(input_cloud)

    # Apply transform (in-place)
    pcd.transform(T)

    # Save transformed point cloud
    o3d.io.write_point_cloud(output_cloud, pcd)

    print("Saved transformed point cloud to:", output_cloud)
    print("Applied 4x4 transform:\n", T)

if __name__ == "__main__":
    in_N80 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/20240204_055523_SP01-N80-N10-1000_2K-2K_aL_u.pcd"
    out_N80 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/Transformed2/20240204_055523_SP01-N80-N10-1000_2K-2K_aL_u.pcd"
    in_N60 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/20240204_055523_SP01-N60-N10-1000_2K-2K_aL_u.pcd"
    out_N60 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/Transformed2/20240204_055523_SP01-N60-N10-1000_2K-2K_aL_u.pcd"
    in_N40 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/20240204_055523_SP01-N40-N10-1000_2K-2K_aL_u.pcd"   
    out_N40 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/Transformed2/20240204_055523_SP01-N40-N10-1000_2K-2K_aL_u.pcd"   
    in_N20 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/20240204_055523_SP01-N20-N10-1000_2K-2K_aL_u.pcd"
    out_N20 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/Transformed2/20240204_055523_SP01-N20-N10-1000_2K-2K_aL_u.pcd"
    in_0 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/20240204_055523_SP01-0-N10-1000_2K-2K_aL_u.pcd"
    out_0 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/Transformed2/20240204_055523_SP01-0-N10-1000_2K-2K_aL_u.pcd"
    in_20 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/20240204_055523_SP01-20-N10-1000_2K-2K_aL_u.pcd" 
    out_20 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/Transformed2/20240204_055523_SP01-20-N10-1000_2K-2K_aL_u.pcd"
    in_40 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/20240204_055523_SP01-40-N10-1000_2K-2K_aL_u.pcd"
    out_40 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/Transformed2/20240204_055523_SP01-40-N10-1000_2K-2K_aL_u.pcd"
    in_60 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/20240204_055523_SP01-60-N10-1000_2K-2K_aL_u.pcd"
    out_60 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/Transformed2/20240204_055523_SP01-60-N10-1000_2K-2K_aL_u.pcd"
    in_80 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/20240204_055523_SP01-80-N10-1000_2K-2K_aL_u.pcd"
    out_80 = "C:/Users/josie/Documents/_Senior Project/Fake Data/Dirsig Data/pcd/Transformed2/20240204_055523_SP01-80-N10-1000_2K-2K_aL_u.pcd"


    # T1 = np.array([[ 1.000000,  0.000000,  0.000000,  0.000000],
    # [ 0.000000,  1.000000,  0.000000,  0.000000],
    # [ 0.000000,  0.000000,  1.000000,  0.000000],
    # [ 0.000000,  0.000000,  0.000000,  1.000000]])

    # T2 = np.array([
    # [ 1.000000, -0.000024, -0.000365, -0.001960],
    # [ 0.000024,  1.000000, -0.000118, -0.000794],
    # [ 0.000365,  0.000118,  1.000000, -0.004639],
    # [ 0.000000,  0.000000,  0.000000,  1.000000]
    # ])

    # T3 = np.array([
    # [ 1.000000, -0.000558,  0.000054, -0.001363],
    # [ 0.000558,  1.000000,  0.000218, -0.002787],
    # [-0.000054, -0.000218,  1.000000,  0.001744],
    # [ 0.000000,  0.000000,  0.000000,  1.000000]
    # ])

    # T4 = np.array([[0.999999, -0.001029, -0.000933,  0.001455],
    # [ 0.001030,  0.999999,  0.000775, -0.004535],
    # [ 0.000932, -0.000776,  0.999999, -0.002628],
    # [ 0.000000,  0.000000,  0.000000,  1.000000]])
    
    # T5 = np.array([[0.999999, -0.001013, -0.001251, -0.000618],
    # [ 0.001014,  0.999999, 0.000622, -0.005343],
    # [ 0.001251, -0.000623,  0.999999, -0.006961],
    # [ 0.000000,  0.000000,  0.000000,  1.000000]])

    # T6 = np.array([[ 0.999999, -0.001364, -0.000625, -0.000168],
    # [ 0.001365,  0.999999,  0.000985, -0.006498],
    # [ 0.000623, -0.000986,  0.999999,  0.000418],
    # [ 0.000000,  0.000000,  0.000000,  1.000000]])

    # T7 = np.array([[ 0.999998, -0.001062, -0.001670,  0.001801],
    # [ 0.001063,  0.999999,  0.000618, -0.006008],
    # [ 0.001670, -0.000619,  0.999998, -0.005548],
    # [ 0.000000,  0.000000,  0.000000,  1.000000]])

    # T8 = np.array([[ 0.999998, -0.001051, -0.001935,  0.000566],
    # [ 0.001051,  0.999999,  0.000369, -0.006950],
    # [ 0.001935, -0.000371,  0.999998, -0.009895],
    # [ 0.000000,  0.000000,  0.000000,  1.000000]])

    # T9 = np.array([[ 1.000000, -0.000024, -0.000365, -0.001960],
    # [ 0.000024,  1.000000, -0.000118, -0.000794],
    # [ 0.000365,  0.000118,  1.000000, -0.004639],
    # [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T1_N80 = np.array([[ 1.000000,  0.000000,  0.000000,  0.000000],
    [ 0.000000,  1.000000,  0.000000,  0.000000],
    [ 0.000000,  0.000000,  1.000000,  0.000000],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T2_N60 = np.array([[ 1.000000, -0.000137, -0.000125,  0.001520],
    [ 0.000136,  0.999999, -0.001599, -0.004640],
    [ 0.000125,  0.001599,  0.999999, -0.002127],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T3_N40 = np.array([[ 0.999999, -0.001033,  0.000112,  0.008127],
    [ 0.001033,  0.999995, -0.003095, -0.012769],
    [-0.000108,  0.003095,  0.999995, -0.005066],
    [ 0.000000,  0.000000,  0.000000,  1.000000]]
)

    T4_N20 = np.array([[ 0.999999, -0.000849,  0.000915,  0.010003],
    [ 0.000853,  0.999990, -0.004429, -0.013251],
    [-0.000911,  0.004430,  0.999990, -0.007289],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T5_0 = np.array([[ 0.999997, -0.000850,  0.002115,  0.014906],
    [ 0.000861,  0.999985, -0.005421, -0.014807],
    [-0.002111,  0.005423,  0.999983, -0.009850],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T6_20 = np.array([[ 0.999994, -0.001008, 0.003450,  0.019717],
    [ 0.001029,  0.999981, -0.006023, -0.013844],
    [-0.003444,  0.006027,  0.999976, -0.012358],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T7_40 = np.array([[ 0.999987, -0.001250,  0.005031,  0.025150],
    [ 0.001281,  0.999981, -0.006009, -0.011843],
    [-0.005023,  0.006015,  0.999969, -0.014446],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T8_60 = np.array([[ 0.999987, -0.001057,  0.004986,  0.025286],
    [ 0.001087,  0.999982, -0.005970, -0.013446],
    [-0.004980,  0.005976,  0.999970, -0.013760],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T9_80 = np.array([[ 0.999987, -0.001140 , 0.005009,  0.025407],
    [ 0.001170,  0.999981, -0.006082, -0.012809],
    [-0.005002,  0.006087,  0.999969, -0.014336],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    apply_translation(in_N80, out_N80, T1_N80)
    apply_translation(in_N60, out_N60, T2_N60)
    apply_translation(in_N40, out_N40, T3_N40)
    apply_translation(in_N20, out_N20, T4_N20)
    apply_translation(in_0, out_0, T5_0)
    apply_translation(in_20, out_20, T6_20)
    apply_translation(in_40, out_40, T7_40)
    apply_translation(in_60, out_60, T8_60)
    apply_translation(in_80, out_80, T9_80)
