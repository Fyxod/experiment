# Verification Notes

Verified on 2026-06-15 with the existing GPU environment:

```bash
VERIFY_ONLY=1 \
PYTHON_BIN=/workspace/geometric-v1/.venv-linux-gpu/bin/python3.11 \
GEOMETRIC_V1_PATH=/workspace/geometric-v1 \
NO_VENV=1 \
bash linux-gpu/install_a6000.sh
```

Result:

- Python imports passed for NumPy, PIL, SciPy, OpenCV, PyTorch, Diffusers, Transformers, Accelerate, DeepFace, TensorFlow, and `geometric_v1`.
- `Flux2KleinPipeline` import passed.
- All archived `**/scripts/*.py` files passed `py_compile`.
- PyTorch reported CUDA available.
- Observed environment: `torch==2.7.1+cu118`, `diffusers==0.39.0.dev0`, `transformers==4.57.6`, `tensorflow==2.12.1`.
- Observed GPU during verification: NVIDIA H100 80GB HBM3 MIG 3g.40gb, 39.38 GB visible VRAM.

A clean isolated install was also attempted with:

```bash
PYTHON_BIN=/workspace/geometric-v1/.venv-linux-gpu/bin/python3.11 \
ENV_DIR=.venv-a6000-check \
PYTORCH_CUDA=cu118 \
GEOMETRIC_V1_PATH=/workspace/geometric-v1 \
bash linux-gpu/install_a6000.sh
```

That attempt stopped before dependency resolution because this container could not reach PyPI (`Network is unreachable`) while fetching `wheel`. No package conflict was reached.

`pip check` on the pre-existing `geometric-v1` environment reports known metadata issues: TensorFlow 2.12 requests `typing-extensions<4.6`, while the PyTorch/Accelerate stack needs a newer version; the existing env also has OpenCV headless even though DeepFace metadata asks for `opencv-python`. Runtime imports and script compilation passed despite those metadata warnings, and `install_a6000.sh` installs the OpenCV packages before restoring modern `typing-extensions` for PyTorch/Accelerate.
