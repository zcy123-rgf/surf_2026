"""Create a compact, filename-labelled contact sheet from an image folder."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--cell-width", type=int, default=360)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite contact sheet: {args.output}")
    paths = sorted(
        path
        for path in args.input_dir.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {args.input_dir}")
    images = []
    cell_height = None
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot decode {path}")
        scale = args.cell_width / image.shape[1]
        resized = cv2.resize(
            image,
            (args.cell_width, int(round(image.shape[0] * scale))),
            interpolation=cv2.INTER_AREA,
        )
        labelled = cv2.copyMakeBorder(
            resized, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        cv2.putText(
            labelled,
            path.stem,
            (8, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        cell_height = max(cell_height or 0, labelled.shape[0])
        images.append(labelled)
    assert cell_height is not None
    columns = max(1, args.columns)
    rows = math.ceil(len(images) / columns)
    canvas = np.full(
        (rows * cell_height, columns * args.cell_width, 3), 255, dtype=np.uint8
    )
    for index, image in enumerate(images):
        row, column = divmod(index, columns)
        y = row * cell_height
        x = column * args.cell_width
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise OSError(f"Failed to write {args.output}")
    print(f"Contact sheet: {args.output.resolve()}")


if __name__ == "__main__":
    main()
