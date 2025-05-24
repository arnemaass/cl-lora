#!/bin/bash
#SBATCH --job-name=cl_benchmark          # Jobname
#SBATCH --output=logs/%x_%j.out          # Standard output log
#SBATCH --error=logs/%x_%j.err           # Standard error log
#SBATCH --time=04:00:00                  # Laufzeit (HH:MM:SS)
#SBATCH --partition=compute              # Partition/Queue (z.B. compute, gpu)
#SBATCH --gres=gpu:1                     # GPU Ressourcen (falls benötigt)
#SBATCH --ntasks=1                       # Anzahl der Tasks (MPI)
#SBATCH --cpus-per-task=4               # CPU-Kerne pro Task
#SBATCH --mem=32G                        # Arbeitsspeicher

# Optional: Lade Module oder Umgebungen
module load anaconda/2021.11
source activate cl_env

# Erstelle Verzeichnisse für Logs
mkdir -p logs

# Pfade
CONFIG=config.yml
SCRIPT=run_experiment.py

# Starte das Python-Skript
srun python ${SCRIPT} --config ${CONFIG}
