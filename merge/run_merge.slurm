#!/bin/bash
#SBATCH --job-name=cl_benchmark          # Jobname
#SBATCH --gres=gpu:1                 # Request one GPU
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=72G                    # Memory update if necessary
#SBATCH --time=07:00:00              # Time limit
#SBATCH --output=log/pytorch_job_%j.out  # Standard output log (%j is the job ID)
#SBATCH --error=log/pytorch_job_%j.err   # Standard error log


# 1) Arbeitsverzeichnis wechseln
# cd /home/anton/src/cl-lora/merge
# cd /home/arne/src/baseline
# 2) Logs-Ordner anlegen
mkdir -p log

# 3) Absolute Python‐Binär aus deinem Env nutzen
#    (das ist das Ergebnis von `which python` in deinem Env)
PYTHON=/faststorage/SpectralGPT/env/bin/python

export PYTHONPATH="/faststorage/shuocheng/LoRA_ViT:/home/arne/LoRA-ViT:${PYTHONPATH:-}"

# 5) run baseline.py, teeing logs into a timestamped file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE=log/pytorch_job_${TIMESTAMP}.log


# 4) Skript starten
srun $PYTHON merge.py --config config_merge.yml 2>&1 | tee "$LOGFILE"
