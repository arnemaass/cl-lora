## SpectralGPT

```bash
wget https://zenodo.org/records/13139925/files/SpectralGPT.pth?download=1
mv SpectralGPT.pth?download=1 SpectralGPT.pth
```

Change directory in SpectralGPT.py

## Install requirements (requirements.txt)
pip install -r requirements.txt


## copy weight folders
```powershell
pscp -r credentials@erde.rsim.tu-berlin.de:/faststorage/continual_low_rank_adaptation_of_remote_sensing_foundation_models/SpectralGPT/saved_models/epoch20/task_tuning .\
```
## Use saved_dataset instead of full BENv2
- run save_datasets.py
- in yml do additional configuration:
```yml
  use_saved_datasets: true
  saved_datasets_dir: "~/saved_datasets"
```