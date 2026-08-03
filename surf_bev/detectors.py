import os
import sys

import cv2
import numpy as np


def _load_python_config_without_temp(Config, config_path):
    """Load a standalone CLRNet Python config without creating temp .py files."""
    with open(config_path, "r", encoding="utf-8") as stream:
        config_text = stream.read()
    namespace = {"__file__": config_path, "__name__": "clrnet_runtime_config"}
    exec(compile(config_text, config_path, "exec"), namespace)
    config_dict = {
        name: value for name, value in namespace.items() if not name.startswith("__")
    }
    if "_base_" in config_dict:
        raise RuntimeError(
            "The no-temp Windows config loader only supports standalone CLRNet configs."
        )
    return Config(config_dict, cfg_text=config_text, filename=config_path)


def _order_polyline(points):
    points = np.asarray(points)
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float64)

    points = points.reshape(-1, 2).tolist()
    start = max(points, key=lambda p: p[1])
    ordered = [start]
    points.remove(start)
    while points:
        last = np.array(ordered[-1])
        idx = int(np.argmin([np.linalg.norm(last - np.array(p)) for p in points]))
        ordered.append(points.pop(idx))
    return np.asarray(ordered, dtype=np.float64)


def _smooth_polyline(points, window=5):
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        return points

    smoothed = []
    for idx in range(len(points)):
        start = max(0, idx - window)
        end = min(len(points), idx + window + 1)
        smoothed.append(points[start:end].mean(axis=0))
    return np.asarray(smoothed)


class HoughLaneDetector:
    """Fast fallback detector copied from the original BEV notebook."""

    def detect(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
        height, width = edges.shape

        mask = np.zeros_like(edges)
        roi = np.array([[(0, height), (width, height), (width // 2, int(height * 0.6))]], dtype=np.int32)
        cv2.fillPoly(mask, roi, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked, rho=2, theta=np.pi / 180, threshold=60, minLineLength=60, maxLineGap=50)
        left_points, right_points = [], []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = np.asarray(line).reshape(-1)[:4]
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.6:
                    continue
                if slope < 0 and x1 < width // 2:
                    left_points.extend([[x1, y1], [x2, y2]])
                elif slope > 0 and x1 > width // 2:
                    right_points.extend([[x1, y1], [x2, y2]])

        lanes = []
        for points in (left_points, right_points):
            lane = _smooth_polyline(_order_polyline(points))
            if len(lane) > 1:
                lanes.append(lane)
        return {"image_rgb": image_rgb, "lanes": lanes}


class CLRNetLaneDetector:
    """Adapter that turns CLRNet predictions into image-space lane polylines."""

    def __init__(
        self,
        clrnet_root="CLRNet",
        config="configs/clrnet/clr_resnet18_culane.py",
        checkpoint="weights/culane_r18.pth",
        device="cpu",
    ):
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.clrnet_root = os.path.abspath(clrnet_root)
        sys.path.insert(0, self.clrnet_root)

        from clrnet.models.registry import build_net
        from clrnet.utils.config import Config

        config_path = config if os.path.isabs(config) else os.path.join(self.clrnet_root, config)
        checkpoint_path = checkpoint if os.path.isabs(checkpoint) else os.path.join(self.clrnet_root, checkpoint)

        try:
            self.cfg = Config.fromfile(config_path)
        except PermissionError:
            # Some managed Windows workstations forbid creating temporary .py
            # files. The bundled CLRNet config is standalone, so execute the
            # existing repository file directly instead of copying it to TEMP.
            self.cfg = _load_python_config_without_temp(Config, config_path)
        self.base_ori_img_h = self.cfg.ori_img_h
        self.base_cut_height = self.cfg.cut_height
        self.cfg.gpus = 1
        self.cfg.view = False
        self.cfg.seed = 0
        self.cfg.backbone.pretrained = False
        self.model = build_net(self.cfg).to(self.device)

        checkpoint_data = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint_data["net"] if "net" in checkpoint_data else checkpoint_data
        state = {key.replace("module.", "", 1): value for key, value in state.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()

    def _preprocess(self, image_bgr):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        cut_height = int(round(self.base_cut_height * image_bgr.shape[0] / float(self.base_ori_img_h)))
        crop = image_bgr[cut_height:, :, :]
        resized = cv2.resize(crop, (self.cfg.img_w, self.cfg.img_h), interpolation=cv2.INTER_CUBIC)
        tensor = self.torch.from_numpy(resized.astype(np.float32) / 255.0)
        return image_rgb, tensor.permute(2, 0, 1).unsqueeze(0), cut_height

    def _prediction_to_polyline(self, prediction, image_shape, cut_height):
        prediction = prediction.detach().cpu().numpy()
        image_h, image_w = image_shape[:2]
        scale_x = image_w / float(self.cfg.img_w)
        scale_y = (image_h - cut_height) / float(self.cfg.img_h)
        n_offsets = self.cfg.num_points
        ys_norm = np.linspace(1, 0, n_offsets)

        xs = prediction[6:] * (self.cfg.img_w - 1)
        start = max(0, min(n_offsets - 1, int(round(prediction[2] * (n_offsets - 1)))))
        length = max(0, int(round(prediction[5])))
        end = min(n_offsets - 1, start + length - 1)

        points = []
        for idx in range(start, end + 1):
            x = xs[idx]
            if x < 0 or x >= self.cfg.img_w:
                continue
            y = ys_norm[idx] * self.cfg.img_h
            points.append([x * scale_x, y * scale_y + cut_height])
        return np.asarray(points, dtype=np.float64)

    def detect(self, image_bgr):
        image_rgb, tensor, cut_height = self._preprocess(image_bgr)
        with self.torch.no_grad():
            raw = self.model(tensor.to(self.device))
            predictions = self.model.heads.get_lanes(raw, as_lanes=False)[0]

        lanes = []
        for prediction in predictions:
            lane = self._prediction_to_polyline(prediction, image_bgr.shape, cut_height)
            if len(lane) > 1:
                lanes.append(lane)
        return {"image_rgb": image_rgb, "lanes": lanes}


def build_detector(name="clrnet", **kwargs):
    if name == "clrnet":
        return CLRNetLaneDetector(**kwargs)
    if name == "hough":
        return HoughLaneDetector()
    raise ValueError(f"Unknown detector: {name}")
