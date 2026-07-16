import cv2
import numpy as np


def load_kitti_calib(calib_path):
    data = {}
    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key] = np.array([float(x) for x in value.split()])

    p2 = data["P2"].reshape(3, 4)
    tr = np.eye(4)
    tr[:3, :4] = data["Tr_velo_to_cam"].reshape(3, 4)
    return {
        "P2": p2,
        "K": p2[:3, :3],
        "R0_rect": data["R0_rect"].reshape(3, 3),
        "Tr": tr,
    }


def load_kitti_poses(pose_path):
    poses = []
    with open(pose_path, "r") as f:
        for line in f:
            values = np.array([float(x) for x in line.split()])
            pose = np.eye(4)
            pose[:3, :4] = values.reshape(3, 4)
            poses.append(pose)
    return poses


def camera_to_ground_rotation(pitch_deg=0.0):
    pitch = np.deg2rad(pitch_deg)
    camera_axis_to_ground = np.diag([1.0, -1.0, 1.0])
    r_pitch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    return r_pitch @ camera_axis_to_ground


def image_to_ground_ipm(points, K, camera_height=1.65, pitch_deg=0.0, z_range=(0.0, 80.0)):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    points = points.reshape(-1, 2)
    pixels_h = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    rays_camera = (np.linalg.inv(K) @ pixels_h.T).T
    rays_ground = (camera_to_ground_rotation(pitch_deg) @ rays_camera.T).T

    camera_center = np.array([0.0, camera_height, 0.0])
    dy = rays_ground[:, 1]
    valid = dy < -1e-8
    scale = np.full(len(points), np.nan, dtype=np.float64)
    scale[valid] = -camera_height / dy[valid]
    valid &= scale > 0.0

    ground_xyz = camera_center + scale[:, None] * rays_ground
    ground_xz = ground_xyz[:, [0, 2]]
    valid &= np.isfinite(ground_xz).all(axis=1)
    valid &= ground_xz[:, 1] >= z_range[0]
    valid &= ground_xz[:, 1] <= z_range[1]
    return ground_xz[valid]


def ground_to_image_ipm(points, K, camera_height=1.65, pitch_deg=0.0):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    points = points.reshape(-1, 2)
    r_g2c = camera_to_ground_rotation(pitch_deg).T
    ground_xyz = np.column_stack([points[:, 0], np.zeros(len(points)), points[:, 1]])
    camera_xyz = (r_g2c @ (ground_xyz - np.array([0.0, camera_height, 0.0])).T).T
    valid = camera_xyz[:, 2] > 1e-8
    pixels_h = (K @ camera_xyz[valid].T).T
    return pixels_h[:, :2] / pixels_h[:, 2:3]


def compute_image_to_bev_homography(
    K,
    camera_height=1.65,
    pitch_deg=0.0,
    bev_size=(800, 800),
    x_range=(-10.0, 10.0),
    z_range=(3.0, 50.0),
):
    x_min, x_max = x_range
    z_min, z_max = z_range
    bev_h, bev_w = bev_size
    ground_corners = np.array(
        [[x_min, z_min], [x_max, z_min], [x_max, z_max], [x_min, z_max]],
        dtype=np.float64,
    )
    image_corners = ground_to_image_ipm(ground_corners, K, camera_height, pitch_deg)
    if len(image_corners) != 4:
        raise ValueError("Selected BEV ground rectangle is not visible in the image.")

    bev_corners = np.array(
        [[0.0, bev_h - 1.0], [bev_w - 1.0, bev_h - 1.0], [bev_w - 1.0, 0.0], [0.0, 0.0]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(image_corners.astype(np.float32), bev_corners)


def warp_image_to_bev(
    image_rgb,
    K,
    camera_height=1.65,
    pitch_deg=0.0,
    bev_size=(800, 800),
    x_range=(-10.0, 10.0),
    z_range=(3.0, 50.0),
):
    h_image_to_bev = compute_image_to_bev_homography(
        K, camera_height, pitch_deg, bev_size, x_range, z_range
    )
    bev_h, bev_w = bev_size
    return cv2.warpPerspective(image_rgb, h_image_to_bev, (bev_w, bev_h))


def ground_xz_to_camera_xyz(points_xz, camera_height=1.65, pitch_deg=0.0):
    points_xz = np.asarray(points_xz, dtype=np.float64)
    if points_xz.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    points_xz = points_xz.reshape(-1, 2)
    ground_xyz = np.column_stack([points_xz[:, 0], np.zeros(len(points_xz)), points_xz[:, 1]])
    r_g2c = camera_to_ground_rotation(pitch_deg).T
    return (r_g2c @ (ground_xyz - np.array([0.0, camera_height, 0.0])).T).T


def camera_xyz_to_ground_xz(camera_xyz, camera_height=1.65, pitch_deg=0.0):
    camera_xyz = np.asarray(camera_xyz, dtype=np.float64)
    if camera_xyz.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    camera_xyz = camera_xyz.reshape(-1, 3)
    ground_xyz = (camera_to_ground_rotation(pitch_deg) @ camera_xyz.T).T
    ground_xyz += np.array([0.0, camera_height, 0.0])
    return ground_xyz[:, [0, 2]]


def transform_lane_points_by_pose(points_xz, T_w_cam_src, T_w_cam_ref, camera_height=1.65, pitch_deg=0.0):
    if len(points_xz) == 0:
        return np.empty((0, 2), dtype=np.float64)

    points_cam_src = ground_xz_to_camera_xyz(points_xz, camera_height, pitch_deg)
    points_cam_src_h = np.column_stack([points_cam_src, np.ones(len(points_cam_src))])
    T_cam_ref_cam_src = np.linalg.inv(T_w_cam_ref) @ T_w_cam_src
    points_cam_ref = (T_cam_ref_cam_src @ points_cam_src_h.T).T[:, :3]
    return camera_xyz_to_ground_xz(points_cam_ref, camera_height, pitch_deg)

