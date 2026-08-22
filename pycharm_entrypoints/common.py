"""Shared paths and helpers for direct PyCharm entry points."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_ROOT = PROJECT_ROOT / "pycharm_outputs"
LATEST_PIPELINE = OUTPUT_ROOT / "LATEST_PIPELINE.json"
DATASET_ROOT = Path(
    os.environ.get(
        "KITTI_ODOMETRY_ROOT",
        r"F:\BaiduNetdiskDownload\kitti\odometry",
    )
)
IMAGE_DIR = (
    DATASET_ROOT
    / "data_odometry_color"
    / "dataset"
    / "sequences"
    / "00"
    / "image_2"
)
CALIB = (
    DATASET_ROOT
    / "data_odometry_calib"
    / "dataset"
    / "sequences"
    / "00"
    / "calib.txt"
)
POSES = (
    DATASET_ROOT
    / "data_odometry_poses"
    / "dataset"
    / "poses"
    / "00.txt"
)
CLRNET_ROOT = Path(os.environ.get("CLRNET_ROOT", str(PROJECT_ROOT / "CLRNet")))


def require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} was not found: {path}")
    return path


def require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} was not found: {path}")
    return path


def new_output(prefix: str) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = OUTPUT_ROOT / f"{prefix}_{stamp}"
    if output.exists():
        raise FileExistsError(f"Generated output path already exists: {output}")
    return output


def run_main(main: Callable[[], None], arguments: Sequence[str]) -> None:
    """Call an existing CLI main in this process so PyCharm can debug it."""

    previous = sys.argv[:]
    sys.argv = [getattr(main, "__module__", "entrypoint"), *map(str, arguments)]
    print("Python:", sys.executable)
    print("Project:", PROJECT_ROOT)
    print("Arguments:", " ".join(sys.argv[1:]))
    try:
        main()
    finally:
        sys.argv = previous


def pipeline_files(output: Path) -> dict[str, Path] | None:
    detected = output / "00_metadata" / "detected_lane_points.json"
    aligned = output / "00_metadata" / "aligned_lane_points.json"
    if detected.is_file() and aligned.is_file():
        return {"output": output, "detected": detected, "aligned": aligned}
    return None


def record_latest_pipeline(output: Path) -> dict[str, Path]:
    files = pipeline_files(output)
    if files is None:
        raise FileNotFoundError(
            f"Pipeline completed without the required metadata files: {output}"
        )
    document = {
        "status": "complete",
        "output": str(files["output"].resolve()),
        "detected": str(files["detected"].resolve()),
        "aligned": str(files["aligned"].resolve()),
    }
    LATEST_PIPELINE.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return files


def latest_pipeline_files() -> dict[str, Path]:
    if LATEST_PIPELINE.is_file():
        document = json.loads(LATEST_PIPELINE.read_text(encoding="utf-8-sig"))
        files = pipeline_files(Path(document["output"]))
        if files is not None:
            return files

    if OUTPUT_ROOT.is_dir():
        candidates = sorted(
            (item for item in OUTPUT_ROOT.iterdir() if item.is_dir()),
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            files = pipeline_files(candidate)
            if files is not None:
                return files

    raise FileNotFoundError(
        "No valid pose-aligned pipeline output was found. "
        "Run 01_generate_pose_aligned_points.py first."
    )
