import os

import cv2
import numpy as np

from .geometry import (
    image_to_ground_ipm,
    load_kitti_calib,
    load_kitti_poses,
    transform_lane_points_by_pose,
    warp_image_to_bev,
)
from .visualization import draw_bev_lanes, draw_image_lanes


def default_intrinsics_for_image(width, height):
    focal = max(width, height) * 0.9
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def run_single_frame(
    detector,
    image_path,
    output_dir,
    calib_path=None,
    camera_height=1.65,
    pitch_deg=0.0,
    bev_size=(800, 800),
    x_range=(-10.0, 10.0),
    z_range=(3.0, 50.0),
):
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(image_path)

    detection = detector.detect(image_bgr)
    image_rgb = detection["image_rgb"]
    lanes = detection["lanes"]

    if calib_path:
        K = load_kitti_calib(calib_path)["K"]
    else:
        K = default_intrinsics_for_image(image_bgr.shape[1], image_bgr.shape[0])

    lanes_ground = [image_to_ground_ipm(lane, K, camera_height, pitch_deg, z_range) for lane in lanes]
    try:
        bev_rgb = warp_image_to_bev(image_rgb, K, camera_height, pitch_deg, bev_size, x_range, z_range)
    except ValueError:
        bev_rgb = np.zeros((bev_size[0], bev_size[1], 3), dtype=np.uint8)

    image_vis = draw_image_lanes(image_rgb, lanes)
    bev_vis = draw_bev_lanes(bev_rgb, lanes_ground, x_range, z_range)

    os.makedirs(output_dir, exist_ok=True)
    image_out = os.path.join(output_dir, "single_frame_lanes.jpg")
    bev_out = os.path.join(output_dir, "single_frame_bev.jpg")
    cv2.imwrite(image_out, cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR))
    cv2.imwrite(bev_out, cv2.cvtColor(bev_vis, cv2.COLOR_RGB2BGR))
    return {"lanes": lanes, "lanes_ground": lanes_ground, "image_out": image_out, "bev_out": bev_out}


def run_multi_frame(
    detector,
    image_dir,
    output_dir,
    calib_path,
    pose_path,
    frame_ids,
    ref_id=0,
    image_pattern="um_{frame_id:06d}.png",
    camera_height=1.65,
    pitch_deg=0.0,
    bev_size=(800, 800),
    x_range=(-10.0, 10.0),
    z_range=(3.0, 50.0),
):
    K = load_kitti_calib(calib_path)["K"]
    poses = load_kitti_poses(pose_path)
    T_w_cam_ref = poses[ref_id]
    fused = np.zeros((bev_size[0], bev_size[1], 3), dtype=np.uint8)
    debug_frames = []

    for frame_id in frame_ids:
        image_path = os.path.join(image_dir, image_pattern.format(frame_id=frame_id))
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(image_path)

        detection = detector.detect(image_bgr)
        lanes_ground = [
            image_to_ground_ipm(lane, K, camera_height, pitch_deg, z_range)
            for lane in detection["lanes"]
        ]
        aligned = [
            transform_lane_points_by_pose(points, poses[frame_id], T_w_cam_ref, camera_height, pitch_deg)
            for points in lanes_ground
        ]
        fused = draw_bev_lanes(fused, aligned, x_range, z_range, thickness=3)
        debug_frames.append(draw_image_lanes(detection["image_rgb"], detection["lanes"]))

    os.makedirs(output_dir, exist_ok=True)
    fused_out = os.path.join(output_dir, "multi_frame_fused_bev.jpg")
    cv2.imwrite(fused_out, cv2.cvtColor(fused, cv2.COLOR_RGB2BGR))

    for idx, frame in enumerate(debug_frames):
        out = os.path.join(output_dir, f"debug_frame_{idx:02d}.jpg")
        cv2.imwrite(out, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    return {"fused_out": fused_out, "debug_count": len(debug_frames)}

