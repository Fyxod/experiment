#!/usr/bin/env bash
set -euo pipefail

# Install and verify the experiment archive on an Ubuntu A6000 GPU machine.
# No sudo or apt is used. Defaults create a Python 3.11 venv in .venv-a6000.

PYTHON_BIN="${PYTHON_BIN:-auto}"
ENV_DIR="${ENV_DIR:-.venv-a6000}"
PYTORCH_CUDA="${PYTORCH_CUDA:-cu118}"
LINUX_GPU_CONSTRAINTS="${LINUX_GPU_CONSTRAINTS:-linux-gpu/constraints-a6000.txt}"
SKIP_TORCH="${SKIP_TORCH:-0}"
NO_VENV="${NO_VENV:-0}"
VERIFY_ONLY="${VERIFY_ONLY:-0}"
INSTALL_GEOMETRIC_V1="${INSTALL_GEOMETRIC_V1:-1}"
GEOMETRIC_V1_PATH="${GEOMETRIC_V1_PATH:-../geometric-v1}"
USE_MICROMAMBA_IF_NEEDED="${USE_MICROMAMBA_IF_NEEDED:-1}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$HOME/.local/bin/micromamba}"
MICROMAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT_PREFIX:-$HOME/.local/micromamba}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_PATH="${REPO_ROOT}/${ENV_DIR}"

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

die() {
  printf '\nERROR: %s\n' "$*" >&2
  exit 2
}

python_is_311() {
  "$1" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 11) else 1)
PY
}

find_python311() {
  if [[ "${PYTHON_BIN}" != "auto" ]]; then
    command -v "${PYTHON_BIN}" >/dev/null 2>&1 || return 1
    python_is_311 "${PYTHON_BIN}" || return 1
    command -v "${PYTHON_BIN}"
    return
  fi

  for candidate in python3.11 python3 python; do
    if command -v "${candidate}" >/dev/null 2>&1 && python_is_311 "${candidate}"; then
      command -v "${candidate}"
      return
    fi
  done
  return 1
}

download_micromamba() {
  if [[ -x "${MICROMAMBA_BIN}" ]]; then
    return
  fi
  command -v curl >/dev/null 2>&1 || die "curl is required to download micromamba, or set PYTHON_BIN to an existing Python 3.11."
  command -v tar >/dev/null 2>&1 || die "tar is required to unpack micromamba, or set PYTHON_BIN to an existing Python 3.11."

  log "Downloading micromamba into ${MICROMAMBA_BIN}"
  tmp_dir="$(mktemp -d)"
  mkdir -p "$(dirname "${MICROMAMBA_BIN}")"
  curl -L "https://micro.mamba.pm/api/micromamba/linux-64/latest" -o "${tmp_dir}/micromamba.tar.bz2"
  tar -xjf "${tmp_dir}/micromamba.tar.bz2" -C "${tmp_dir}"
  mv "${tmp_dir}/bin/micromamba" "${MICROMAMBA_BIN}"
  chmod +x "${MICROMAMBA_BIN}"
  rm -rf "${tmp_dir}"
}

create_micromamba_env() {
  download_micromamba
  export MAMBA_ROOT_PREFIX="${MICROMAMBA_ROOT_PREFIX}"

  if [[ ! -d "${ENV_PATH}" ]]; then
    log "Creating micromamba environment: ${ENV_PATH}"
    "${MICROMAMBA_BIN}" create -y -p "${ENV_PATH}" -c conda-forge python=3.11 pip setuptools wheel
  else
    log "Using existing environment: ${ENV_PATH}"
  fi

  # shellcheck disable=SC1090
  eval "$("${MICROMAMBA_BIN}" shell hook -s bash)"
  micromamba activate "${ENV_PATH}"
}

create_venv() {
  local python_bin="$1"

  if [[ "${NO_VENV}" == "1" ]]; then
    log "NO_VENV=1, using current Python environment"
    return
  fi

  if [[ ! -d "${ENV_PATH}" ]]; then
    log "Creating virtual environment: ${ENV_PATH}"
    "${python_bin}" -m venv "${ENV_PATH}" || die "Could not create venv. Rerun with USE_MICROMAMBA_IF_NEEDED=1 if python3.11-venv is unavailable."
  else
    log "Using existing virtual environment: ${ENV_PATH}"
  fi

  # shellcheck disable=SC1091
  source "${ENV_PATH}/bin/activate"
}

install_torch() {
  if [[ "${SKIP_TORCH}" == "1" ]]; then
    log "Skipping PyTorch install because SKIP_TORCH=1"
    return
  fi

  case "${PYTORCH_CUDA}" in
    cu128|cu126|cu118)
      log "Installing PyTorch CUDA wheels: ${PYTORCH_CUDA}"
      python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${PYTORCH_CUDA}"
      ;;
    cpu)
      log "Installing CPU-only PyTorch wheels"
      python -m pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cpu"
      ;;
    *)
      die "Unsupported PYTORCH_CUDA=${PYTORCH_CUDA}. Use cu128, cu126, cu118, or cpu."
      ;;
  esac
}

write_torch_constraints() {
  local constraints_path="$1"
  python - "${constraints_path}" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, version

with open(sys.argv[1], "w", encoding="utf-8") as handle:
    for package in ("torch", "torchvision", "torchaudio"):
        try:
            handle.write(f"{package}=={version(package)}\n")
        except PackageNotFoundError:
            pass
PY
}

write_requirements_without_accelerate() {
  local requirements_path="$1"
  python - "${requirements_path}" <<'PY'
import sys
from pathlib import Path

lines = []
for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
    if line.strip().startswith("accelerate"):
        continue
    lines.append(line)
Path(sys.argv[1]).write_text("\n".join(lines) + "\n", encoding="utf-8")
PY
}

install_requirements() {
  command -v git >/dev/null 2>&1 || die "git is required because requirements.txt installs Diffusers from GitHub."

  log "Installing global experiment requirements"
  core_requirements="$(mktemp)"
  torch_constraints="$(mktemp)"
  write_requirements_without_accelerate "${core_requirements}"
  write_torch_constraints "${torch_constraints}"

  constraints_args=(-c "${torch_constraints}")
  if [[ -n "${LINUX_GPU_CONSTRAINTS}" ]]; then
    [[ -f "${LINUX_GPU_CONSTRAINTS}" ]] || die "Missing constraints file: ${LINUX_GPU_CONSTRAINTS}"
    constraints_args+=(-c "${LINUX_GPU_CONSTRAINTS}")
  fi

  PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/${PYTORCH_CUDA}" python -m pip install -r "${core_requirements}" "${constraints_args[@]}"
  rm -f "${core_requirements}" "${torch_constraints}"

  # TensorFlow 2.12 pins typing-extensions too low for newer PyTorch/Accelerate.
  python -m pip install "typing-extensions>=4.14,<5"
  python -m pip install "psutil>=5.9" -c "${LINUX_GPU_CONSTRAINTS}"
  python -m pip install --no-deps "accelerate>=0.30" -c "${LINUX_GPU_CONSTRAINTS}"
}

install_geometric_v1() {
  if [[ "${INSTALL_GEOMETRIC_V1}" != "1" ]]; then
    log "Skipping geometric-v1 install because INSTALL_GEOMETRIC_V1=${INSTALL_GEOMETRIC_V1}"
    return
  fi

  if [[ -d "${GEOMETRIC_V1_PATH}" ]]; then
    log "Installing local geometric-v1 package without dependency resolution: ${GEOMETRIC_V1_PATH}"
    python -m pip install -e "${GEOMETRIC_V1_PATH}" --no-deps
  else
    log "Local geometric-v1 path not found; installing from GitHub without dependency resolution"
    python -m pip install "git+https://github.com/Fyxod/geometric-v1.git" --no-deps
  fi
}

verify_install() {
  log "Verifying Python imports, CUDA availability, and script syntax"
  if [[ -d "${GEOMETRIC_V1_PATH}" ]]; then
    export PYTHONPATH="${GEOMETRIC_V1_PATH}:${PYTHONPATH:-}"
  fi
  python - <<'PY'
import py_compile
from pathlib import Path

import numpy
import PIL.Image
import scipy
import cv2
import torch
import diffusers
import transformers
import accelerate
import deepface
import geometric_v1
from diffusers import Flux2KleinPipeline
from geometric_v1.config import DeepFaceConfig
from geometric_v1.deepface_compare import compare_images

print("python imports: ok")
print(f"torch: {torch.__version__}")
print(f"torch cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"cuda device: {torch.cuda.get_device_name(0)}")
    print(f"vram total gb: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f}")
print(f"diffusers: {diffusers.__version__}")
print(f"transformers: {transformers.__version__}")
print("Flux2KleinPipeline import: ok")
print("geometric_v1 import: ok")

for script in sorted(Path(".").glob("**/scripts/*.py")):
    py_compile.compile(str(script), doraise=True)
print("script py_compile: ok")
PY

  CUDA_VISIBLE_DEVICES=-1 python - <<'PY'
import tensorflow as tf
print(f"tensorflow: {tf.__version__}")
PY
}

main() {
  cd "${REPO_ROOT}"

  if [[ "${VERIFY_ONLY}" != "1" ]]; then
    python_bin=""
    if python_bin="$(find_python311)"; then
      create_venv "${python_bin}"
    elif [[ "${USE_MICROMAMBA_IF_NEEDED}" == "1" ]]; then
      create_micromamba_env
    else
      die "Python 3.11 not found. Set PYTHON_BIN or enable USE_MICROMAMBA_IF_NEEDED=1."
    fi

    python -m pip install --upgrade pip setuptools wheel
    install_torch
    install_requirements
    install_geometric_v1
  else
    if [[ "${PYTHON_BIN}" != "auto" ]]; then
      python_bin="$(find_python311)" || die "PYTHON_BIN must point to Python 3.11 for VERIFY_ONLY=1"
      export PATH="$(dirname "${python_bin}"):${PATH}"
    fi
    log "VERIFY_ONLY=1, skipping package installation"
  fi

  verify_install
  log "Install verification completed"
}

main "$@"
