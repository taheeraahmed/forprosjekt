#!/bin/bash
DATE=$(date +%Y%m%d-%H%M%S)
USER=$(whoami)
TYPE="tensorflow-binary-classification-longer-test"
echo "Current USER is: $USER"

ID=${DATE}-${TYPE}
echo "Current ID is: $ID"
OUTPUT_FILE="/cluster/home/taheeraa/code/forprosjekt/idun/$ID.out"
echo "Current OUTPUT_FILE is: $ID"

# Define the destination path for the code
CODE_PATH="/cluster/home/$USER/runs/code/${ID}"

# Copy the code with rsync, excluding .venv
echo "Copying code to $CODE_PATH"
mkdir -p $CODE_PATH
rsync -av \
  --exclude='.venv' \
  --exclude='idun' \
  --exclude='images' \
  --exclude='runs' \
  --exclude='scripts' \
  --exclude='.git' \
  --exclude='__pycache' \
  /cluster/home/$USER/code/tdt17-visuell-intelligens/ $CODE_PATH

echo "Running slurm job from $CODE_PATH"
sbatch --partition=GPUQ \
  --account=ie-idi \
  --time=28:15:00 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --mem=50G \
  --gres=gpu:1 \
  --job-name=$ID \
  --output=$OUTPUT_FILE \
  $CODE_PATH/train.slurm