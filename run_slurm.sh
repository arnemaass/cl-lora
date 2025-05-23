#!/bin/bash
#SBATCH --job-name=finland       # Job name
#SBATCH --gres=gpu:1                 # Request one GPU
#SBATCH --cpus-per-task=4            # CPU cores per task
#SBATCH --mem=72G                    # Memory update if necessary
#SBATCH --time=02:00:00              # Time limit
#SBATCH --output=log/pytorch_job_%j.out  # Standard output log (%j is the job ID)
#SBATCH --error=log/pytorch_job_%j.err   # Standard error log

# Run the Python script directly within the micromamba environment
# You can also activate the environment and then run the script
micromamba run -n cl_lora_env python finetune.py
