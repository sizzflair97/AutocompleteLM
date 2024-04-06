#!/bin/bash
#SBATCH --job-name=T5
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:a10:4
#SBATCH --cpus-per-task=56
#SBATCH -o ./_out/%j.%N.out
#SBATCH -e ./_err/%j.%N.err

## SBATCH --output=%x-%j.out

module rm CUDA
module load CUDA/11.3.0

echo "start at: $(date)"
echo "node: $(hostname)"
echo "jobid: $(SLURM_JOB_ID)"

source ~/miniconda3/bin/activate ~/miniconda3/envs/torch

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=1479

MONITOR_GPU_SCRIPT=$(cat <<EOF
    hostnode=\`hostname -s\`
    /usr/local/bin/gpustat -i > $HOME/myubai/AutocompleteLM/_log/\$hostnode.gpu &
EOF
)
SRUN_SCRIPT='python -m torch.distributed.run \
 --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
 --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
train.py --deepspeed ds_config.json'

srun --jobid $SLURM_JOBID bash -c "${MONITOR_GPU_SCRIPT}${SRUN_SCRIPT}"