# cl-lora



## Baseline

To generate baseline results, change your directory to *./baseline*.

Change configurations in *config_baseline.yml* for:
- Country permutations
- Number of train and test samples
- Save directory
- Batch_size
- Num_worker
- Epoch
- Learning rate
- Rank of LoRA matrices
- Model: 'SpectralGPT' or 'SoftCon'
- Continual learning schema: 'replay' or 'no_replay'



Change configurations in *run_benchmark.\** for:
- Python executable/binary


Then run
```bash
./run_benchmark.sh
```

or run with slurm
```bash
sbatch run_benchmark.slurm
```


## Merge

To generate merging results, change your directory to *./merge*.

Change configurations in *config_merge.yml* for:
- Merging strategy to test
- Country permutations
- Number of train and test samples
- subset_fraction for stratified sampling for stratified sampling
- Size of memory set for continual learning
- Save directory
- Batch_size
- Num_worker
- Epoch
- Learning rate
- Rank of LoRA matrices
- Model: 'SpectralGPT' or 'SoftCon'
- Continual learning schema: 'replay' or 'no_replay'
- Whether to use saved datasets to run on different machine (as well as its directory)


Change configurations in *run_merge.\** for:
- Python executable/binary


Then run
```bash
./run_merge.sh
```

or run with slurm
```bash
sbatch run_merge.slurm
```