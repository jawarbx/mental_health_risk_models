#!/bin/bash

set -e  # Exit on error

ENV_NAME="mental_health_models"
PYTHON_VERSION="3.13.2"

echo "=========================================="
echo "Setting up conda environment: $ENV_NAME"
echo "=========================================="

# Check if conda is available
if ! command -v conda &> /dev/null; then
	echo "Error: conda is not installed or not in PATH"
	exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
	echo "Environment '$ENV_NAME' already exists."
	read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]; then
		echo "Removing existing environment..."
		conda env remove -n $ENV_NAME -y
	else
		echo "Keeping existing environment. Updating packages..."
		eval "$(conda shell.bash hook)"
		conda activate $ENV_NAME
		pip install -r requirements.txt
		echo "=========================================="
		echo "Environment updated successfully!"
		echo "Activate with: conda activate $ENV_NAME"
		echo "=========================================="
		exit 0
	fi
fi

# Create new environment
echo "Creating conda environment with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install requirements
echo "Installing packages from requirements.txt..."
pip install -r requirements.txt

echo "=========================================="
echo "Setup complete!"
echo "Activate the environment with:"
echo "  conda activate $ENV_NAME"
echo "=========================================="
