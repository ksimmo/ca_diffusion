#!/bin/bash
#SBATCH --job-name=example
#SBATCH --output=slurm_outputs/example.log
#SBATCH --error=slurm_outputs/example_error.log
#SBATCH --nodes=4                                   #number of nodes to use
#SBATCH --ntasks-per-node=4                         #must match number of GPUs
#SBATCH --cpus-per-task=16                          #number of CPU per GPU
#SBATCH --gres=gpu:4                                #number of GPUs (maybe add GPU identifier if necessary like gpu:a100:4)
#SBATCH -p <partition name>                         #on which partition the job should run
#SBATCH --time=12:00:00                             #run time of the job
#SBATCH --signal=SIGUSR1@120                        #raise a signal 120 seconds before the job ends


export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
#export MASTER_ADDR= #on some clusters we might need to set the master addr explicitly to use infiniband connection


echo "################################"
echo SLURM_PROCID=${SLURM_JOB_ID}
echo SLURM_JOBNAME=${SLURM_JOB_NAME}
echo NUM_NODES=${SLURM_NNODES}
echo MASTER_ADDR=${MASTER_ADDR}
echo CPUS_PER_TSASK=${SLURM_CPUS_PER_TASK}
echo NODELIST=${SLURM_JOB_NODELIST}
echo "################################"

#one can also query job information via scontrol
scontrol show jobid ${SLURM_JOB_ID}

#activate python environment
source <source to anaconda or pyenv>/bin/activate
conda activate <env name>

#maybe set some env variables
#export NCCL_IB_TIMEOUT=50
#export UCX_RC_TIMEOUT=4s
#export NCCL_IB_RETRY_CNT=10
#export NCCL_DEBUG=INFO
#export NCCL_ASYNC_ERROR_HANDLING=1
#export CUDA_LAUNCH_BLOCKING=0
#export PYTHONFAULTHANDLER=1


cd <path to your project directory>

#export CUDA_VISIBLE_DEVICES=0,1,2,3 #on some clusters it might be necessary to explicitily set CUDA_VISIBLE_DEVICES!
#we do not need to give specific devices as lightning uses all avaialble resources if necessary
srun python main.py experiment=ca_ae3d