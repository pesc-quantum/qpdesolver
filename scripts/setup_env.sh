#!/usr/bin/env bash
set -euo pipefail

# Provision local .venv with Python 3.9 base and pinned requirements
PYTHON_BASE="${PYTHON_BASE:-/usr/bin/python3}"

echo "Creating .venv using $PYTHON_BASE..."
"$PYTHON_BASE" -m venv .venv

echo "Activating .venv..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing pinned numpy and base requirements..."
pip install "numpy>=1.24.0,<2.0.0"
pip install -r requirements.txt

echo "Installing devito..."
pip install devito "numpy>=1.24.0,<2.0.0"

echo "Installing qpdesolver editable with dev and generator extras..."
pip install -e ".[dev,generator]"

echo "Running test suite sanity check..."
pytest -v tests/unit/test_io_hdf5.py

echo "Environment setup complete!"
