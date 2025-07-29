## Configure SpectralGPT

```bash
wget https://zenodo.org/records/13139925/files/SpectralGPT.pth?download=1
mv SpectralGPT.pth?download=1 SpectralGPT.pth
```

Change directory in SpectralGPT.py to navigate to the checkpoint downloaded above

## Install requirements (requirements.txt)

python=3.11.9
pip install -r requirements.txt


## copy weight folders

```powershell
pscp -r user@erde.rsim.tu-berlin.de:/faststorage/continual_low_rank_adaptation_of_remote_sensing_foundation_models/SpectralGPT/saved_models/epoch20/task_tuning .\
```

## Use saved_dataset instead of full BENv2

- Run save_datasets.py on erde to sample from datasets
- Transfer dataset to your new device
- In yml do additional configuration:

```yml
  use_saved_datasets: true
  saved_datasets_dir: "~/saved_datasets"
```