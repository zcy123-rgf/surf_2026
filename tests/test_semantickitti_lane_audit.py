import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_semantickitti_lane_markings.py"
SPEC = importlib.util.spec_from_file_location("semantic_lane_audit", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_raw_semantic_ids_preserve_original_lane_marking_class():
    packed = np.asarray(
        [60, (123 << 16) | 60, 40, (9 << 16) | 10], dtype=np.uint32
    )
    assert MODULE.raw_semantic_ids(packed).tolist() == [60, 60, 40, 10]


def test_contiguous_runs_do_not_bridge_missing_frames():
    assert MODULE.contiguous_runs([0, 1, 2, 5, 6, 9]) == [
        (0, 2, 3),
        (5, 6, 2),
        (9, 9, 1),
    ]


def test_audit_checks_point_label_pairing_and_counts_lane_points(tmp_path):
    velo = tmp_path / "velodyne"
    labels = tmp_path / "labels"
    images = tmp_path / "image_2"
    velo.mkdir()
    labels.mkdir()
    images.mkdir()

    scans = [
        np.asarray([[1, 0, 0, 0.1], [2, 0, 0, 0.2]], dtype=np.float32),
        np.asarray([[1, 1, 0, 0.3], [2, 1, 0, 0.4]], dtype=np.float32),
    ]
    packed_labels = [
        np.asarray([60, 40], dtype=np.uint32),
        np.asarray([(5 << 16) | 60, 60], dtype=np.uint32),
    ]
    for frame_id, (scan, label) in enumerate(zip(scans, packed_labels)):
        scan.tofile(str(velo / f"{frame_id:06d}.bin"))
        label.tofile(str(labels / f"{frame_id:06d}.label"))
        (images / f"{frame_id:06d}.png").write_bytes(b"test")

    rows, total_points, total_lane = MODULE.audit_frames(
        MODULE.numeric_files(velo, ".bin"),
        MODULE.numeric_files(labels, ".label"),
        60,
        images,
    )
    assert total_points == 4
    assert total_lane == 3
    assert [row["lane_marking_points_raw_id_60"] for row in rows] == [1, 2]
    assert all(row["image_2_exists"] for row in rows)
