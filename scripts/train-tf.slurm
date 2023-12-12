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

# Check if CLASS_IMBALANCE is true and modify JOB_NAME accordingly
if [ "$CLASS_IMBALANCE" == "true" ]; then
    JOB_NAME=${DATE}-${MODEL}-imbalance-tf
else
    JOB_NAME=${DATE}-${MODEL}-tf
fi

echo "--- nvidia-smi ---"
nvidia-smi

module load TensorFlow/2.11.0-foss-2022a-CUDA-11.7.0 # for TensorFlow

pip install -q -r requirements.txt
pip list >> /cluster/home/taheeraa/code/forprosjekt/output/tf/$JOB_NAME/python-packages.txt

echo "--- Running python-script ---"
python $CODE_PATH/src/train-tensorflow.py \
  --model $MODEL \
  --class_imbalance $CLASS_IMBALANCE \
  --date $DATE \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS 


echo " --- Done running train.py ---"