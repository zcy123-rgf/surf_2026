"""Validate the Windows workstation environment and optionally run CLRNet once."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--run-clrnet", action="store_true")
    parser.add_argument("--image", type=Path, default=ROOT / "data" / "000001_original.jpg")
    args = parser.parse_args()

    import cv2
    import numpy as np
    import torch
    import torchvision

    report = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_runtime_bundled_with_torch": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "clrnet_submodule": (ROOT / "CLRNet" / "clrnet").is_dir(),
        "clrnet_checkpoint": (ROOT / "CLRNet" / "weights" / "culane_r18.pth").is_file(),
    }

    if args.device == "cuda" and not torch.cuda.is_available():
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    if args.run_clrnet:
        if not report["clrnet_submodule"]:
            raise FileNotFoundError("CLRNet submodule is missing.")
        if not report["clrnet_checkpoint"]:
            raise FileNotFoundError("CLRNet checkpoint is missing.")
        image_path = args.image.resolve()
        encoded = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode sample image: {image_path}")

        sys.path.insert(0, str(ROOT))
        from surf_bev.detectors import CLRNetLaneDetector

        detector = CLRNetLaneDetector(
            clrnet_root=str(ROOT / "CLRNet"),
            config="configs/clrnet/clr_resnet18_culane.py",
            checkpoint="weights/culane_r18.pth",
            device=args.device,
        )
        result = detector.detect(image)
        report["sample_image"] = str(image_path)
        report["detected_lanes"] = len(result["lanes"])
        report["lane_point_counts"] = [len(lane) for lane in result["lanes"]]

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
