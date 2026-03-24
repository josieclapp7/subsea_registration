import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R


# Load PCDs
target_pcd = o3d.io.read_point_cloud("C:/Users/josie/Documents/GitHub/subsea_registration/Animation Data/N40-N10.pcd")
source_pcd = o3d.io.read_point_cloud("C:/Users/josie/Documents/GitHub/subsea_registration/Animation Data/N60-N10.pcd")

target_pcd = target_pcd.voxel_down_sample(0.2)
source_pcd = source_pcd.voxel_down_sample(0.2)

target = np.asarray(target_pcd.points)


# Apply artificial misalignment
angle = np.deg2rad(30)
R_init = R.from_euler('z', angle).as_matrix()
t_init = np.array([2, -1, 0.5])

T_init = np.eye(4)
T_init[:3,:3] = R_init
T_init[:3,3] = t_init

source_pcd.transform(T_init)
source = np.asarray(source_pcd.points)


threshold = 0.2
trans_init = np.eye(4)

reg = o3d.pipelines.registration.registration_icp(
    source_pcd,
    target_pcd,
    threshold,
    trans_init,
    o3d.pipelines.registration.TransformationEstimationPointToPoint()
)

T_final = reg.transformation

R_final = T_final[:3,:3]
t_final = T_final[:3,3]

rotvec = R.from_matrix(R_final).as_rotvec()
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

sc_target = ax.scatter(target[:,0], target[:,1], target[:,2],
                       s=1, c='blue', alpha=0.5)

sc_source = ax.scatter(source[:,0], source[:,1], source[:,2],
                       s=1, c='red', alpha=0.5)

ax.set_box_aspect([1,1,1])

frames = 150

def ease_out(u):
    return 1 - (1 - u)**3

def update(frame):
    u = ease_out(frame / frames)

    # residual = 1 - u
    # jitter = 0.002 * residual
    # transformed += np.random.randn(*transformed.shape) * jitter


    # Interpolate rotation
    current_rot = R.from_rotvec(rotvec * u).as_matrix()

    # Interpolate translation
    current_t = t_final * u

    transformed = (current_rot @ source.T).T + current_t

    sc_source._offsets3d = (
        transformed[:,0],
        transformed[:,1],
        transformed[:,2]
    )

    return sc_source,

ani = FuncAnimation(fig, update, frames=frames, interval=40)
plt.show()

