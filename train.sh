#!/bin/bash
DATE=$(date +%Y%m%d-%H%M%S)
USER=$(whoami)
TYPE="tensorflow-binary-classification-test"
echo "Current USER is: $USER"

ID=${DATE}-${TYPE}
echo "Current ID is: $ID"
OUTPUT_FILE="/cluster/home/taheeraa/code/forprosjekt/idun/$ID.out"
echo "Current OUTPUT_FILE is: $ID"

sbatch --partition=GPUQ \
  --account=ie-idi \
  --time=08:15:00 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --mem=50G \
  --gres=gpu:1 \
  --job-name=$ID \
  --output=$OUTPUT_FILE \
  ./train.slurm