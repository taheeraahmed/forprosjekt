#!/bin/bash
TEST_MODE=true

MODELS=("swin" "densenet")
CLASS_IMBALANCES=("true" "false")

IDUN_TIME=90:00:00

#    ======= DO NOT EDIT THIS SCRIPT =======

DATE=$(date "+%Y-%m-%d-%H:%M:%S")
USER=$(whoami)
CURRENT_PATH=$(pwd)

for MODEL in "${MODELS[@]}"; do
    for CLASS_IMBALANCE in "${CLASS_IMBALANCES[@]}"; do

        # Check if CLASS_IMBALANCE is true and modify JOB_NAME accordingly
        if [ "$CLASS_IMBALANCE" == "true" ]; then
            JOB_NAME=${DATE}-${MODEL}-imbalance-tf
        else
            JOB_NAME=${DATE}-${MODEL}-tf
        fi

        mkdir -p /cluster/home/$USER/code/forprosjekt/output/tf/$JOB_NAME

        echo "Made directory: /cluster/home/$USER/code/forprosjekt/output/tf/$JOB_NAME"
        OUTPUT_FILE="/cluster/home/$USER/code/forprosjekt/output/tf/$JOB_NAME/idun_out.out"
        echo "Current OUTPUT_FOLDER is: $JOB_NAME"

        # Define the destination path for the code
        CODE_PATH="/cluster/home/$USER/runs/code/tf/${JOB_NAME}"

        echo "Copying code to $CODE_PATH"
        mkdir -p $CODE_PATH
        rsync -av \
            --exclude='.venv' \
            --exclude='idun' \
            --exclude='images' \
            --exclude='runs' \
            --exclude='notebooks' \
            --exclude='output' \
            --exclude='.git' \
            --exclude='__pycache__' \
            --exclude='utils/__pycache__' \
            --exclude='trainers/__pycache__' \
            --exclude='mlruns/' \
            /cluster/home/$USER/code/forprosjekt/ $CODE_PATH

        echo "Current user is: $USER"
        echo "Current path is: $CURRENT_PATH"
        echo "Current job name is: $JOB_NAME"
        echo "Running slurm job from $CODE_PATH"

        sbatch --partition=GPUQ \
            --account=ie-idi \
            --time=$IDUN_TIME \
            --nodes=1 \
            --ntasks-per-node=1 \
            --mem=50G \
            --gres=gpu:1 \
            --job-name=$JOB_NAME \
            --output=$OUTPUT_FILE \
            --export=CODE_PATH=$CODE_PATH,DATE=$DATE,MODEL=$MODEL,CLASS_IMBALANCE=$CLASS_IMBALANCE \
            $CODE_PATH/scripts/train-tf.slurm
    done
done
