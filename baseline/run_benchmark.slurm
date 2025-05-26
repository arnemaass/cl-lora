#!/bin/bash
#SBATCH --job-name=cl_benchmark          # Jobname
#SBATCH --gres=gpu:1                 # Request one GPU
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=72G                    # Memory update if necessary
#SBATCH --time=02:00:00              # Time limit
#SBATCH --output=log/pytorch_job_%j.out  # Standard output log (%j is the job ID)
#SBATCH --error=log/pytorch_job_%j.err   # Standard error log


# 1) Arbeitsverzeichnis wechseln
cd /home/anton/src/cl-lora/baseline

# 2) Logs-Ordner anlegen
mkdir -p log

# 3) Absolute Python‐Binär aus deinem Env nutzen
#    (das ist das Ergebnis von `which python` in deinem Env)
PYTHON=/faststorage/arne/mamba/envs/cl_lora_env/bin/python

# 4) Skript starten
srun $PYTHON baseline.py --config config_baseline.yml