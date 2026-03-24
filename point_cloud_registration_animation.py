import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

np.random.seed(42)

# -----------------------------
# Base point cloud
# -----------------------------
N = 800
theta = np.random.uniform(0, 2*np.pi, N)
phi   = np.random.uniform(0, np.pi, N)
r     = 1 + 0.05*np.random.randn(N)

x = r * np.sin(phi) * np.cos(theta)
y = r * np.sin(phi) * np.sin(theta)
z = r * np.cos(phi)

cloud1 = np.vstack((x, y, z)).T

# -----------------------------
# Initial misalignment
# -----------------------------
angle = np.deg2rad(35)
t     = np.array([0.6, -0.3, 0.4])

def Rz(a):
    return np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0,          0,         1]
    ])

cloud2 = (Rz(angle) @ cloud1.T).T + t

# -----------------------------
# Figure setup
# -----------------------------
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')

sc1 = ax.scatter(cloud1[:,0], cloud1[:,1], cloud1[:,2],
                 s=5, c='blue', alpha=0.6)

sc2 = ax.scatter(cloud2[:,0], cloud2[:,1], cloud2[:,2],
                 s=5, c='red', alpha=0.6)

ax.set_xlim([-2,2])
ax.set_ylim([-2,2])
ax.set_zlim([-2,2])

# -----------------------------
# Animation parameters
# -----------------------------
motion_frames = 200
hold_frames   = 60
total_frames  = motion_frames + hold_frames

def ease_out_cubic(u):
    return 1 - (1 - u)**3

# -----------------------------
# Animation update
# -----------------------------
def update(frame):

    # Reset color at start of loop
    if frame == 0:
        sc2.set_color('red')

    if frame < motion_frames:
        u = frame / motion_frames
        alpha = ease_out_cubic(u)
    else:
        alpha = 1.0

    # Turn magenta when converged
    if frame == motion_frames:
        sc2.set_color('magenta')

    # Interpolated transform
    current_angle = angle * (1 - alpha)
    t_interp      = t * (1 - alpha)

    transformed = (Rz(current_angle) @ cloud1.T).T + t_interp

    # -----------------------------
    # Decaying jitter (ICP feel)
    # -----------------------------
    if frame < motion_frames:
        residual = 1 - alpha

        # jitter magnitude proportional to residual
        jitter_rot = np.deg2rad(2.0) * residual
        jitter_trans = 0.05 * residual

        random_angle = np.random.randn() * jitter_rot
        random_shift = np.random.randn(3) * jitter_trans

        transformed = (Rz(random_angle) @ transformed.T).T + random_shift

    sc2._offsets3d = (
        transformed[:,0],
        transformed[:,1],
        transformed[:,2]
    )

    return sc2,

ani = FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=40,
    repeat=True
)

plt.show()

# Save if desired:
ani.save("point_cloud_registration_animation.mp4", writer="pillow", fps=25)