"""Package the verified KITTI 00 first-five-frame fusion results.

This script only copies existing, audited outputs and builds a deterministic
comparison canvas. It does not synthesize or alter lane/fusion measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FRAME_IDS = range(5)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.chmod(0o644)
    shutil.copyfile(require(source), destination)


def load_font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    names = ["msyhbd.ttc" if bold else "msyh.ttc", "simhei.ttf", "arial.ttf"]
    for name in names:
        candidate = Path("C:/Windows/Fonts") / name
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default(size=size)


def contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    output = image.copy().convert("RGB")
    output.thumbnail(size, Image.Resampling.LANCZOS)
    return output


def paste_centered(
    canvas: Image.Image,
    image: Image.Image,
    box: tuple[int, int, int, int],
    *,
    border: bool = True,
) -> None:
    left, top, right, bottom = box
    fitted = contain(image, (right - left, bottom - top))
    x = left + (right - left - fitted.width) // 2
    y = top + (bottom - top - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    if border:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((x, y, x + fitted.width - 1, y + fitted.height - 1), outline="black", width=2)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    center_x: int,
    top: int,
    font: ImageFont.FreeTypeFont,
    *,
    fill: str = "black",
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    draw.text((center_x - (bounds[2] - bounds[0]) / 2, top), text, font=font, fill=fill)


def build_overview(output_dir: Path, audit: dict) -> Path:
    width = 2960
    margin = 40
    col_width = 560
    col_gap = 20
    canvas = Image.new("RGB", (width, 1850), "white")
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(46, bold=True)
    row_font = load_font(31, bold=True)
    frame_font = load_font(24)
    metric_font = load_font(29)
    note_font = load_font(24)

    draw_centered_text(
        draw,
        "KITTI Odometry Sequence 00 前5帧：车道线提取、BEV与位姿融合全流程",
        width // 2,
        24,
        title_font,
    )

    rows = [
        ("① 原始图像（000000—000004）", "01_original_frames", "frame_{:06d}.png", 112, 170),
        ("② CLRNet左右双车道线提取", "02_lane_detection", "frame_{:06d}_clrnet_two_lanes.png", 372, 170),
        ("③ 固定矩阵BEV投影（仅车道线）", "03_bev_projection", "frame_{:06d}_bev.png", 632, 410),
    ]

    for label, folder, pattern, top, image_height in rows:
        draw.text((margin, top), label, font=row_font, fill="black")
        image_top = top + 48
        for frame_id in FRAME_IDS:
            x = margin + frame_id * (col_width + col_gap)
            image = Image.open(output_dir / folder / pattern.format(frame_id))
            paste_centered(canvas, image, (x, image_top, x + col_width, image_top + image_height))
            draw_centered_text(
                draw,
                f"{frame_id:06d}",
                x + col_width // 2,
                image_top + image_height + 4,
                frame_font,
            )

    fusion_top = 1145
    draw.text((margin, fusion_top), "④ 结合位姿的五帧融合：去噪前后对比", font=row_font, fill="black")
    raw_path = output_dir / "05_fusion" / "fusion_without_denoise.png"
    den_path = output_dir / "05_fusion" / "fusion_with_ransac.png"
    fusion_y = fusion_top + 92
    fusion_size = 560
    raw_x = 250
    den_x = 1000
    paste_centered(canvas, Image.open(raw_path), (raw_x, fusion_y, raw_x + fusion_size, fusion_y + fusion_size))
    paste_centered(canvas, Image.open(den_path), (den_x, fusion_y, den_x + fusion_size, fusion_y + fusion_size))
    draw_centered_text(draw, "未去噪", raw_x + fusion_size // 2, fusion_y - 38, row_font)
    draw_centered_text(draw, "RANSAC去噪后", den_x + fusion_size // 2, fusion_y - 38, row_font)

    raw = audit["fusion"]["without_denoise"]
    ransac = audit["fusion"]["ransac"]
    den = ransac["after_denoise"]
    metrics = [
        "数据与处理参数",
        "数据集：KITTI Odometry Sequence 00",
        "帧：000000—000004；参考帧：000004",
        "车道线：每帧保留CLRNet左右两条",
        "BEV范围：X[-10,10] m，Z[3,50] m",
        f"未去噪：{raw['points']}点，重叠率{raw['overlap_fraction'] * 100:.1f}%",
        f"RANSAC后：{ransac['points']}点，重叠率{den['overlap_fraction'] * 100:.1f}%",
        f"点保留率：{ransac['point_retention_fraction'] * 100:.1f}%",
        "RANSAC：阈值0.3 m，100次迭代",
    ]
    text_x = 1745
    text_y = fusion_y + 18
    for index, line in enumerate(metrics):
        font = row_font if index == 0 else metric_font
        draw.text((text_x, text_y), line, font=font, fill="black")
        text_y += 55 if index == 0 else 49
    draw.multiline_text(
        (text_x, text_y + 18),
        "说明：重叠率用于描述多帧一致性，\n不等同于车道线检测精度；本组5帧无车道线真值。",
        font=note_font,
        fill=(70, 70, 70),
        spacing=10,
    )

    overview = output_dir / "06_overview" / "all_stages_comparison.png"
    overview.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(overview, format="PNG", optimize=True)
    return overview


def write_manifest(output_dir: Path, audit: dict, provenance: dict) -> None:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(output_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    manifest = {
        "dataset": audit["dataset"],
        "official_image_provenance": provenance,
        "detector": {
            "name": audit["detector"]["name"],
            "config": audit["detector"]["config"],
            "checkpoint": audit["detector"]["checkpoint"],
            "checkpoint_sha256": audit["detector"]["checkpoint_sha256"],
        },
        "projection": audit["projection"],
        "pose_alignment": audit["pose_alignment"],
        "fusion": audit["fusion"],
        "files": files,
    }
    destination = output_dir / "audit" / "manifest.json"
    destination.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--stage-dir", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--provenance-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = json.loads(require(args.audit_json).read_text(encoding="utf-8"))
    provenance = json.loads(require(args.provenance_json).read_text(encoding="utf-8"))

    for frame_id in FRAME_IDS:
        copy_file(
            args.raw_dir / f"{frame_id}.png",
            output_dir / "01_original_frames" / f"frame_{frame_id:06d}.png",
        )
        copy_file(
            args.stage_dir / f"frame_{frame_id:02d}_CLRNet双车道线选择.png",
            output_dir / "02_lane_detection" / f"frame_{frame_id:06d}_clrnet_two_lanes.png",
        )
        copy_file(
            args.stage_dir / f"frame_{frame_id:02d}_双车道线固定矩阵BEV.png",
            output_dir / "03_bev_projection" / f"frame_{frame_id:06d}_bev.png",
        )
        copy_file(
            args.stage_dir / f"frame_{frame_id:02d}_双车道线位姿对齐到000004.png",
            output_dir / "04_pose_aligned_bev" / f"frame_{frame_id:06d}_aligned_to_000004.png",
        )

    copy_file(
        args.stage_dir / "03_双车道线位姿融合_未去噪.png",
        output_dir / "05_fusion" / "fusion_without_denoise.png",
    )
    copy_file(
        args.stage_dir / "04_双车道线位姿融合_RANSAC去噪后.png",
        output_dir / "05_fusion" / "fusion_with_ransac.png",
    )
    copy_file(
        args.stage_dir / "05_双车道线融合_RANSAC前后对比.png",
        output_dir / "05_fusion" / "fusion_before_after_comparison.png",
    )
    copy_file(args.audit_json, output_dir / "audit" / "two_lane_bev_audit.json")
    copy_file(args.provenance_json, output_dir / "audit" / "kitti_first5_provenance.json")
    copy_file(args.raw_dir / "calib.txt", output_dir / "metadata" / "calib.txt")
    pose_lines = require(args.raw_dir / "00.txt").read_text(encoding="utf-8").splitlines()[:5]
    (output_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata" / "poses_00_first5.txt").write_text("\n".join(pose_lines) + "\n", encoding="utf-8")

    build_overview(output_dir, audit)
    write_manifest(output_dir, audit, provenance)
    print(output_dir)


if __name__ == "__main__":
    main()
