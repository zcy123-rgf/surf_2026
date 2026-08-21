"""Small KITTI calibration reader shared by the final pipeline."""

from pathlib import Path

import numpy as np


def parse_projection(path: Path) -> dict[str, np.ndarray]:
    """Read KITTI P2/P0 and return camera intrinsics plus image pose mapping."""

    values: dict[str, np.ndarray] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, raw = line.split(":", 1)
        values[key.strip()] = np.asarray(
            [float(item) for item in raw.split()], dtype=np.float64
        )
    key = "P2" if "P2" in values else "P0"
    if key not in values or values[key].size != 12:
        raise ValueError(f"Calibration file has no valid P2/P0 projection: {path}")
    projection = values[key].reshape(3, 4)
    intrinsic = projection[:, :3]
    image_from_cam0 = np.eye(4, dtype=np.float64)
    image_from_cam0[:3, 3] = np.linalg.solve(intrinsic, projection[:, 3])
    return {"K": intrinsic, "cam0_from_image": np.linalg.inv(image_from_cam0)}
