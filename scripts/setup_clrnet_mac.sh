#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLRNET_DIR="${ROOT_DIR}/CLRNet"
WEIGHT_URL="https://github.com/Turoad/CLRNet/releases/download/models/culane_r18.pth.zip"

if [ ! -d "${CLRNET_DIR}/.git" ]; then
  echo "CLRNet submodule is missing. Run:"
  echo "  git submodule update --init --recursive"
  exit 1
fi

python3 -m venv "${CLRNET_DIR}/.venv-clrnet-demo"
"${CLRNET_DIR}/.venv-clrnet-demo/bin/pip" install --upgrade pip
"${CLRNET_DIR}/.venv-clrnet-demo/bin/pip" install \
  torch torchvision opencv-python numpy addict pyyaml tqdm yapf six

mkdir -p "${CLRNET_DIR}/weights"
if [ ! -f "${CLRNET_DIR}/weights/culane_r18.pth" ]; then
  curl -L -o "${CLRNET_DIR}/weights/culane_r18.pth.zip" "${WEIGHT_URL}"
  unzip -o "${CLRNET_DIR}/weights/culane_r18.pth.zip" -d "${CLRNET_DIR}/weights"
fi

patch -d "${CLRNET_DIR}" -N -p1 < "${ROOT_DIR}/scripts/patches/clrnet_mac_compat.patch" || true
cp -R "${ROOT_DIR}/scripts/clrnet_compat/mmcv" "${CLRNET_DIR}/"

echo "CLRNet Mac demo environment is ready."
echo "Try:"
echo "  CLRNet/.venv-clrnet-demo/bin/python run_demo.py --mode single --detector clrnet --image data/000001_original.jpg"

