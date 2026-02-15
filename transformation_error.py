import numpy as np
from transformations import *
def invert_transform(T):
    return np.linalg.inv(T)

def invert_transforms(transform_list):
    return [np.linalg.inv(T) for T in transform_list]
    T1 = np.array([[ 1.000000, -0.000067, -0.000180, -0.004089],
    [ 0.000067,  1.000000, -0.000135, -0.001389],
    [ 0.000180,  0.000135,  1.000000, -0.000345],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T2 = np.array([[ 1.000000, -0.000074, -0.000256, -0.004085],
    [ 0.000074,  1.000000, -0.000247, -0.004480],
    [ 0.000256,  0.000247,  1.000000, -0.000889],
    [ 0.000000,  0.00000,  0.000000,  1.000000]])

    T3 = np.array([[ 1.000000,  0.000000,  0.000000,  0.000000],
    [ 0.000000,  1.000000,  0.000000,  0.000000],
    [ 0.000000,  0.000000,  1.000000,  0.000000],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T4 = np.array([[ 1.000000,  0.000013,  0.000072,  0.002750],
    [-0.000013,  1.000000,  0.000163,  0.001254],
    [-0.000072, -0.000163,  1.000000,  0.000985],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])
    
    T5 = np.array([[ 1.000000,  0.000000,  0.000000,  0.000000],
    [ 0.000000,  1.000000,  0.000000,  0.000000],
    [ 0.000000,  0.000000,  1.000000,  0.000000],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T6 = np.array([[ 1.000000, -0.000146, -0.000422, -0.006870],
    [ 0.000145,  1.000000, -0.000130, -0.003803],
    [ 0.000422,  0.000130,  1.000000, -0.001590],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T7 = np.array([[ 1.000000, -0.000146, -0.000422, -0.006870],
    [ 0.000145,  1.000000, -0.000130, -0.003803],
    [ 0.000422,  0.000130,  1.000000, -0.001590],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T8 = np.array([[ 1.000000, -0.000067, -0.000180, -0.004089],
    [ 0.000067,  1.000000, -0.000135, -0.001389],
    [ 0.000180,  0.000135,  1.000000, -0.000345],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T9 = np.array([[ 1.000000, -0.000067, -0.000180, -0.004089],
    [ 0.000067,  1.000000, -0.000135, -0.001389],
    [ 0.000180,  0.000135,  1.000000, -0.000345],
    [ 0.000000,  0.000000,  0.000000,  1.000000]])

    T_gt_list = [T1, T2, T3, T4, T5, T6, T7, T8, T9]
    return T_gt_list

def write_transform_comparison(filename, original_transforms, attempt_back_transforms):
    """
    Writes original, attempt-back, and inverse(attempt-back) transforms to a text file.

    Parameters
    ----------
    original_transforms : list[np.ndarray]
        List of 4x4 transforms originally applied to the clouds.
    attempt_back_transforms : list[np.ndarray]
        List of 4x4 transforms estimated to return clouds to original pose.
    """

    if len(original_transforms) != len(attempt_back_transforms):
        raise ValueError("Transform lists must be the same length")

    # Validate transforms
    for i, T in enumerate(original_transforms):
        if not isinstance(T, np.ndarray) or T.shape != (4, 4):
            raise TypeError(f"original_transforms[{i}] is not a 4x4 numpy array")

    for i, T in enumerate(attempt_back_transforms):
        if not isinstance(T, np.ndarray) or T.shape != (4, 4):
            raise TypeError(f"attempt_back_transforms[{i}] is not a 4x4 numpy array")

    inverse_attempts = [np.linalg.inv(T) for T in attempt_back_transforms]

    with open(filename, "w") as f:
        f.write("TRANSFORM COMPARISON\n")
        f.write("Original vs Attempt-Back vs Inverse(Attempt-Back)\n\n")

        for i, (T_orig, T_attempt, T_inv) in enumerate(
            zip(original_transforms, attempt_back_transforms, inverse_attempts)
        ):
            f.write(f"================ Cloud {i} ================\n\n")

            f.write("ORIGINAL TRANSFORM:\n")
            f.write(np.array2string(
                T_orig,
                formatter={'float_kind': lambda x: f"{x: .6f}"}
            ))
            f.write("\n\n")

            f.write("ATTEMPT-BACK TRANSFORM:\n")
            f.write(np.array2string(
                T_attempt,
                formatter={'float_kind': lambda x: f"{x: .6f}"}
            ))
            f.write("\n\n")

            f.write("INVERSE(ATTEMPT-BACK) TRANSFORM:\n")
            f.write(np.array2string(
                T_inv,
                formatter={'float_kind': lambda x: f"{x: .6f}"}
            ))
            f.write("\n\n")

    """
    Computes global alignment G such that:
        G @ T_est ≈ T_gt
    """
    gt_centers = extract_centers(T_gt_list)
    est_centers = extract_centers(T_est_list)

    mu_gt = gt_centers.mean(axis=0)
    mu_est = est_centers.mean(axis=0)

    X = est_centers - mu_est
    Y = gt_centers - mu_gt

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_gt - R @ mu_est

    G = np.eye(4)
    G[:3, :3] = R
    G[:3, 3] = t
    return G

def translation_error(T_gt, T_est):
    T_err = np.linalg.inv(T_est) @ T_gt
    return np.linalg.norm(T_err[:3, 3])

def translation_error_list(T_gt_list, T_est_list):
    """
    Computes per-cloud translation errors.

    Returns
    -------
    errors : list[float]
        Translation error for each cloud.
    """
    if len(T_gt_list) != len(T_est_list):
        raise ValueError("Transform lists must be the same length")

    errors = []
    for i, (T_gt, T_est) in enumerate(zip(T_gt_list, T_est_list)):
        err = translation_error(T_gt, T_est)
        errors.append(err)

    return errors

def rotation_error(T_gt, T_est):
    R_err = T_est[:3,:3].T @ T_gt[:3,:3]
    cos_theta = (np.trace(R_err) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.degrees(np.arccos(cos_theta))

def rotation_error_list(T_gt_list, T_est_list):
    if len(T_gt_list) != len(T_est_list):
        raise ValueError("Transform lists must be same length")

    return [
        rotation_error(T_gt, T_est)
        for T_gt, T_est in zip(T_gt_list, T_est_list)
    ]

def main():
    original_transforms = original_translation_list_2()
    icp_transforms = final_translation_icp_list_2()
    multiway_transforms = final_translation_multiway_list_2()

    # write_transform_comparison("icp_stuff.txt", original_transforms, icp_transforms)
    # write_transform_comparison("multiway_stuff.txt", original_transforms, multiway_transforms)

    print("ICP ERRORS -------------------")
    for i, (T_gt, T_est) in enumerate(zip(original_transforms, icp_transforms)):
        r = rotation_error(T_gt, T_est)
        t = translation_error(T_gt, T_est)
        print(f"Cloud {i:02d}: rot = {r:.4f} deg   trans = {t:.6f} m")

    print("MULTIWAY ERRORS -------------------")
    for i, (T_gt, T_est) in enumerate(zip(original_transforms, multiway_transforms)):
        r = rotation_error(T_gt, T_est)
        t = translation_error(T_gt, T_est)
        print(f"Cloud {i:02d}: rot = {r:.4f} deg   trans = {t:.6f} m")


    print("done :(")

if __name__ == "__main__":
    main()