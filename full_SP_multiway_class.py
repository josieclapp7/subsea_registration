from dataclasses import dataclass, field
import numpy as np
import open3d as o3d

@dataclass
class RegistrationConfig:
    voxel_size: float = 0.005
    edge_prune_threshold: float = 0.25
    reference_node: int = 0

    @property
    def coarse_distance(self) -> float:
        return self.voxel_size * 15

    @property
    def fine_distance(self) -> float:
        return self.voxel_size * 5

    @property
    def rmse_threshold(self) -> float:
        return self.voxel_size * 0.4


@dataclass
class ScanPair:
    source_id: int
    target_id: int
    transformation: np.ndarray
    information: np.ndarray
    inlier_rmse: float
    is_adjacent: bool  # False = loop closure edge


@dataclass
class RegistrationResult:
    filenames: list[str]
    pose_graph: o3d.pipelines.registration.PoseGraph
    transformed_clouds: list[o3d.geometry.PointCloud]
    rmse_values: list[float]
    avg_rmse: float
    std_rmse: float
    elapsed_seconds: float

    def print_summary(self):
        print(f"Time: {self.elapsed_seconds / 60:.2f} min")
        print(f"Avg RMSE: {self.avg_rmse:.6f}  Std: {self.std_rmse:.6f}")
        for fname, rmse in zip(self.filenames, self.rmse_values):
            print(f"  {fname}: {rmse:.6f}")

    def save_transforms(self, path: str):
        with open(path, "w") as f:
            f.write("FINAL TRANSFORMS (cloud -> global)\n\n")
            for i, fname in enumerate(self.filenames):
                T = self.pose_graph.nodes[i].pose
                f.write(f"--- {fname} ---\n")
                f.write(np.array2string(T, formatter={'float_kind': lambda x: f"{x:.6f}"}))
                f.write("\n\n")