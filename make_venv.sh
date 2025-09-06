#!/bin/bash
set -e

ENV_NAME="notecast"

echo ">>> Creating environment $ENV_NAME from environment.yml"
conda env create -f environment.yml || conda env update -f environment.yml --prune

echo ">>> Activating environment"
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo ">>> Download completed!"