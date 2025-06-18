#!/usr/bin/env bash
set -euo pipefail

# we assume you're already in the cl_lora_env
# if not, activate it however you normally would

# 1) cd into your repo
# cd /home/anton/src/cl-lora/baseline
cd /home/arne/src/baseline


# 2) make sure the log folder exists
mkdir -p log

# 3) point to the exact python binary
PYTHON=/faststorage/arne/mamba/envs/cl_lora_env/bin/python

# 4) export PYTHONPATH so that `import pos_embed` works
export PYTHONPATH="/faststorage/shuocheng/LoRA_ViT:/home/arne/LoRA-ViT:${PYTHONPATH:-}"

# 5) run baseline.py, teeing logs into a timestamped file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGFILE=log/pytorch_job_${TIMESTAMP}.log

$PYTHON baseline.py --config config_baseline.yml 2>&1 | tee "$LOGFILE"