#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLRNET_DIR="${ROOT_DIR}/CLRNet"
WEIGHT_URL="${WEIGHT_URL:-https://github.com/Turoad/CLRNet/releases/download/models/culane_r18.pth.zip}"
CLRNET_WEIGHT_FILE="${CLRNET_WEIGHT_FILE:-}"

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
  if [ -n "${CLRNET_WEIGHT_FILE}" ]; then
    if [ ! -f "${CLRNET_WEIGHT_FILE}" ]; then
      echo "CLRNET_WEIGHT_FILE does not exist: ${CLRNET_WEIGHT_FILE}"
      exit 1
    fi
    cp "${CLRNET_WEIGHT_FILE}" "${CLRNET_DIR}/weights/culane_r18.pth"
  else
    if ! curl -fL --retry 3 --retry-delay 5 -o "${CLRNET_DIR}/weights/culane_r18.pth.zip" "${WEIGHT_URL}"; then
      echo ""
      echo "Failed to download CLRNet weights from:"
      echo "  ${WEIGHT_URL}"
      echo ""
      echo "Manual fallback:"
      echo "  1. Get culane_r18.pth from another teammate or a shared drive."
      echo "  2. Re-run this script with:"
      echo "     CLRNET_WEIGHT_FILE=/path/to/culane_r18.pth bash scripts/setup_clrnet_mac.sh"
      echo ""
      echo "Alternative URL fallback:"
      echo "     WEIGHT_URL=https://your-mirror/culane_r18.pth.zip bash scripts/setup_clrnet_mac.sh"
      exit 1
    fi
    unzip -o "${CLRNET_DIR}/weights/culane_r18.pth.zip" -d "${CLRNET_DIR}/weights"
  fi
fi

patch -d "${CLRNET_DIR}" -N -p1 < "${ROOT_DIR}/scripts/patches/clrnet_mac_compat.patch" || true
cp -R "${ROOT_DIR}/scripts/clrnet_compat/mmcv" "${CLRNET_DIR}/"

echo "CLRNet Mac demo environment is ready."
echo "Try:"
echo "  CLRNet/.venv-clrnet-demo/bin/python run_demo.py --mode single --detector clrnet --image data/000001_original.jpg"
