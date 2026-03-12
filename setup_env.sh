#!/bin/bash
set -e  # Exit on error

ENV_NAME="mental_health_models"
PYTHON_VERSION="3.11.15"
VENV_DIR=".venv"

echo "=========================================="
echo "Setting up uv environment: $ENV_NAME"
echo "=========================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
	echo "Error: uv is not installed or not in PATH"
	echo "Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
	exit 1
fi

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
	echo "Environment '$VENV_DIR' already exists."
	read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]; then
		echo "Removing existing environment..."
		rm -rf "$VENV_DIR"
	else
		echo "Keeping existing environment. Updating packages..."
		uv pip install -r requirements.txt
		echo "=========================================="
		echo "Environment updated successfully!"
		echo "Activate with: source $VENV_DIR/bin/activate"
		echo "=========================================="
		exit 0
	fi
fi

# Create new environment with specified Python version
echo "Creating uv environment with Python $PYTHON_VERSION..."
uv venv "$VENV_DIR" --python "$PYTHON_VERSION"

# Activate environment
echo "Activating environment..."
source "$VENV_DIR/bin/activate"

# Install requirements
echo "Installing packages from requirements.txt..."
uv pip install -r requirements.txt

echo "=========================================="
echo "Setup complete!"
echo "Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo "=========================================="
