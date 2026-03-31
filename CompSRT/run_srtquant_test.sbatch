#!/bin/bash
#
# SLURM directives
#
#SBATCH --job-name=srtquant_job           # name of job
#SBATCH --output=logs/%x_%j.out          # stdout → logs/<jobname>_<jobid>.out
#SBATCH --error=logs/%x_%j.err           # stderr → logs/<jobname>_<jobid>.err
#SBATCH --time=00:06:00                  # HH:MM:SS walltime
#SBATCH --partition=gpu                  # GPU partition
#SBATCH --gres=gpu:h200:1                  # request 1 GPU v100-pcie or a100 v100-sxm2
# #SBATCH --constraint=a100@80g
#SBATCH --ntasks=1                       # single task
#SBATCH --cpus-per-task=1                # number of CPU cores
#SBATCH --mem=48G                        # total memory
#SBATCH --signal=SIGUSR1@30

nvidia-smi --query-gpu=name,memory.total --format=csv 

SIF_FILE=./compsrt_image.sif #update this 
# export SINGULARITYENV_NVIDIA_DRIVER_CAPABILITIES=compute,utility 
# export APPTAINERENV_NVIDIA_DRIVER_CAPABILITIES=compute,utility 
# export SINGULARITY_NV=1
# export SINGULARITY_NVLIBLIST="libcuda.so.1,libnvidia-ml.so.1,libcudart.so.11.0,libnvrtc.so.11.1,libcublas.so.11,libcublasLt.so.11,libcufft.so.10,libcurand.so.10,libcusolver.so.11,libcusparse.so.11"

singularity exec -B  "CompSRT:CompSRT" --pwd CompSRT --env PYTHONPATH=CompSRT:$PYTHONPATH,PYTHONDONTWRITEBYTECODE=1  --nv "${SIF_FILE}" run_srt_test.sh "$@" 

