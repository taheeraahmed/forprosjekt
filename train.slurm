#!/bin/sh
echo "We are running from this directory: $SLURM_SUBMIT_DIR"
echo "The name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Total time at IDUN $IDUN_TIME"

echo "--- nvidia-smi ---"
nvidia-smi
module load Python/3.10.8-GCCcore-12.2.0
pip install -q -r requirements.txt

echo "--- Running python-script ---"
python train.py -t $TEST_MODE -of $OUTPUT_FOLDER -it $IDUN_TIME -m $MODEL -bs $BATCH_SIZE -lr $LEARNING_RATE -e $NUM_EPOCHS

echo " --- Done running train.py ---"