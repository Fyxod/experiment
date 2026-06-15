# Linux GPU Install

Use `install_a6000.sh` to recreate the environment for the archived experiment scripts on an Ubuntu A6000 machine.

```bash
cd /path/to/experiment
bash linux-gpu/install_a6000.sh
```

Defaults:

- Python 3.11, recorded in `.python-version`
- virtual environment: `.venv-a6000`
- PyTorch CUDA wheel index: `cu118`
- Linux resolver constraints: `linux-gpu/constraints-a6000.txt`
- local `geometric-v1` checkout: `../geometric-v1`

Useful overrides:

```bash
PYTORCH_CUDA=cu126 bash linux-gpu/install_a6000.sh
GEOMETRIC_V1_PATH=/workspace/geometric-v1 bash linux-gpu/install_a6000.sh
VERIFY_ONLY=1 PYTHON_BIN=/workspace/geometric-v1/.venv-linux-gpu/bin/python3.11 bash linux-gpu/install_a6000.sh
```

The archive scripts import `geometric_v1` for configuration and DeepFace comparison helpers. The installer uses `pip install -e ../geometric-v1 --no-deps` when that checkout exists, otherwise it installs the GitHub repository without dependency resolution after the global requirements are installed.
