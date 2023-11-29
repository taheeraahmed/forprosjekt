#!/bin/bash
DATE=$(date "+%Y-%m-%d-%H:%M:%S")
USER=$(whoami)
JOB_NAME="pytorch-binary-classify-one-fourth-of-data"
IDUN_TIME=90:00:00
CURRENT_PATH=$(pwd)
TEST_MODE=true

# Check if TEST_MODE is true and append "test" to JOB_NAME
if [ "$TEST_MODE" = true ]; then
    JOB_NAME="TEST-${JOB_NAME}"
    IDUN_TIME=00:15:00
fi

OUTPUT_FOLDER=${DATE}-${JOB_NAME}

mkdir /cluster/home/taheeraa/code/forprosjekt/output/$OUTPUT_FOLDER
echo "Made directory: /cluster/home/taheeraa/code/forprosjekt/output/$OUTPUT_FOLDER"
OUTPUT_FILE="/cluster/home/taheeraa/code/forprosjekt/output/$OUTPUT_FOLDER/idun_out.out"
echo "Current OUTPUT_FOLDER is: $OUTPUT_FOLDER"

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
  --exclude='notebooks' \
  --exclude='scripts' \
  --exclude='output' \
  --exclude='.git' \
  --exclude='__pycache' \
  --exclude='utils/__pycache' \
  --exclude='mlruns/' \
  /cluster/home/$USER/code/forprosjekt/ $CODE_PATH

CURRENT_PATH=$(pwd)

echo "Current user is: $USER"
echo "Current path is: $CURRENT_PATH"
echo "Current job name is: $JOB_NAME"
echo "Running slurm job from $CODE_PATH"

sbatch --partition=GPUQ \
  --account=ie-idi \
  --time=$TIME \
  --nodes=1 \
  --ntasks-per-node=1 \
  --mem=50G \
  --gres=gpu:1 \
  --job-name=$JOB_NAME \
  --output=$OUTPUT_FILE \
  --export=TEST_MODE=$TEST_MODE,OUTPUT_FOLDER=$OUTPUT_FOLDER,IDUN_TIME=$IDUN_TIME \
  $CODE_PATH/train.slurm