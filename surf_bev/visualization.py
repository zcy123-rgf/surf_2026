import cv2
import numpy as np


LANE_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def draw_image_lanes(image_rgb, lanes, thickness=5):
    out = image_rgb.copy()
    for idx, lane in enumerate(lanes):
        points = np.asarray(lane, dtype=np.int32)
        if len(points) < 2:
            continue
        cv2.polylines(out, [points], False, LANE_COLORS[idx % len(LANE_COLORS)], thickness)
    return out


def metric_to_bev_pixels(points, bev_shape, x_range=(-10.0, 10.0), z_range=(3.0, 50.0)):
    points = np.asarray(points, dtype=np.float64)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.int32)

    h, w = bev_shape[:2]
    points = points.reshape(-1, 2)
    valid = (
        (points[:, 0] >= x_range[0])
        & (points[:, 0] <= x_range[1])
        & (points[:, 1] >= z_range[0])
        & (points[:, 1] <= z_range[1])
    )
    points = points[valid]
    if len(points) == 0:
        return np.empty((0, 2), dtype=np.int32)

    u = (points[:, 0] - x_range[0]) / (x_range[1] - x_range[0]) * (w - 1)
    v = (z_range[1] - points[:, 1]) / (z_range[1] - z_range[0]) * (h - 1)
    return np.column_stack([u, v]).astype(np.int32)


def draw_bev_lanes(bev_rgb, lanes_ground, x_range=(-10.0, 10.0), z_range=(3.0, 50.0), thickness=4):
    out = bev_rgb.copy()
    for idx, lane in enumerate(lanes_ground):
        points = metric_to_bev_pixels(lane, out.shape, x_range, z_range)
        if len(points) < 2:
            continue
        cv2.polylines(out, [points], False, LANE_COLORS[idx % len(LANE_COLORS)], thickness)
    return out

