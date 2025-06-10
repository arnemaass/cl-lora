import time
import logging
import random
from typing import Any, Dict, List, Callable, Tuple
import pandas as pd
from collections import defaultdict
import os
import joblib
import json
import yaml
import argparse
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import accuracy_score
from SpectralGPT import *


# --- DataSet factory based on configilm / BENv2 ---
from configilm import util
from configilm.extra.DataSets.BENv2_DataSet import BENv2DataSet

# Pfade zu den Daten
datapath = {
    "images_lmdb": "/faststorage/BigEarthNet-V2/BigEarthNet-V2-LMDB",
    "metadata_parquet": "/faststorage/BigEarthNet-V2/metadata.parquet",
    "metadata_snow_cloud_parquet": "/faststorage/BigEarthNet-V2/metadata_for_patches_with_snow_cloud_or_shadow.parquet",
}
util.MESSAGE_LEVEL = util.MessageLevel.INFO

def load_country_train_val(
    country: str,
    n_samples: int,
    seed: int = None,
    img_size=(14, 120, 120),
    include_snowy=False,
    include_cloudy=False,
    split = "train",
    frac = 0.8
):
    """
    Samples n_samples patches from the TRAIN split of `country`,
    then does an internal 80/20 train/val split (both from the original TRAIN data).

    Returns (train_ds, val_ds).

    """
    meta = pd.read_parquet(datapath["metadata_parquet"])

    # 2) restrict metadata to train-split patches for this country
    mask = (meta.country == country) & (meta.split == split)
    available = meta.loc[mask, "patch_id"].tolist()
    if not available:
        raise ValueError(f"No TRAIN patches found for country={country!r}")
    if n_samples > len(available):
        raise ValueError(
            f"Requested {n_samples} samples but only {len(available)} TRAIN patches available for {country!r}"
        )

    # 3) pick your N
    rng = random.Random(seed)
    sampled = rng.sample(available, k=n_samples)

    # 4) 80/20 split
    # 4) 80/20 split
    split_at = int(n_samples * frac)
    train_ids = set(sampled[:split_at])
    val_ids = set(sampled[split_at:])
    ids = set(sampled[:])

    # 5) helper to build a dataset that still uses split="train"
    def _make_ds(keep_ids):
        return BENv2DataSet(
            data_dirs=datapath,
            img_size=img_size,
            split=split,            # draw only from the original train partition
            include_snowy=include_snowy,
            include_cloudy=include_cloudy,
            patch_prefilter=lambda pid: pid in keep_ids,
            transform=transform
        )

    ds_train = _make_ds(train_ids)
    ds_val =  _make_ds(val_ids)

    #ds = _make_ds(ids)
    return ds_train, ds_val



def get_datasets(
    countries: List[str],
    permutation: List[int],
    split: str = "train",
    img_size: tuple = (12, 128, 128),
    include_snowy: bool = False,
    include_cloudy: bool = False,
    samples_per_country: int = None,
    seed: int = 0,
    frac =0.8
) -> Tuple[List[BENv2DataSet], List[BENv2DataSet]]:
    train_datasets: List[BENv2DataSet] = []
    val_datasets: List[BENv2DataSet] = []

    for idx in permutation:
        country = countries[idx]
        train_ds, val_ds = load_country_train_val(country, samples_per_country, 123, split=split,frac=frac)
        train_datasets.append(train_ds)
        val_datasets.append(val_ds)

    return train_datasets, val_datasets

def default_metrics(y_true: List[Any], y_pred: List[Any]) -> Dict[str, float]:
    """
    Default evaluation metric: wraps sklearn.metrics.accuracy_score.
    """
    return {'accuracy': accuracy_score(y_true, y_pred)}


def test_continual_learning(
    model: Any,
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Continual Learning mit vollständigem Replay über permutierte Länder.
    Daten werden über DataLoader verarbeitet.
    """
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    countries = params['countries']
    permutation = params['permutation']
    train_samples = params.get('train_samples')
    test_samples = params.get('test_samples')
    seed = params.get('seed', 0)
    batch_size = params.get('batch_size', 16)
    num_workers = params.get('num_workers', 4)
    epoch = params.get('epoch', 15)
    lr = float(params.get('lr', 1e-4))

    train_sets, val_sets = get_datasets(countries, permutation, split='train',
                              img_size=params.get('img_size', (12, 128, 128)),
                              include_snowy=params.get('include_snowy', False),
                              include_cloudy=params.get('include_cloudy', False),
                              samples_per_country=train_samples,
                              seed=seed, frac=0.9)
    test_sets, _ = get_datasets(countries, permutation, split='test',
                             img_size=params.get('img_size', (12, 128, 128)),
                             include_snowy=params.get('include_snowy', False),
                             include_cloudy=params.get('include_cloudy', False),
                             samples_per_country=test_samples,
                             seed=seed,
                             frac =1)

    print(len(test_sets[0]))  # now only patches from Austria
    print(len(val_sets[0]))  # now only patches from Austria
    print(len(train_sets[0]))  # now only patches from Austria
    #img, lbl = train_sets[0][0]
    #print("test")
    #print(img.shape)

    #save unchanged lora_weights once

    # (1) Compute an absolute directory under your home or your repo
    repo_dir = os.path.dirname(os.path.abspath(__file__))  # e.g. .../anton/src/cl-lora/baseline
    save_dir = os.path.join(repo_dir, "saved_models")  # e.g. .../anton/src/cl-lora/baseline/saved_models
    os.makedirs(save_dir, exist_ok=True)  # create it if needed

    # (2) Build a fully‐qualified filename
    out_file = os.path.join(save_dir, "lora_weights_baseline.safetensors")

    # (3) Pass that to save_fc_parameters and save_lora_parameters
    model.save_fc_parameters(out_file)
    model.save_lora_parameters(out_file)
    path_to_file = "/home/anton/src/cl-lora/baseline/saved_models/lora_weights_baseline.safetensors"

    metrics_fn = params.get('metrics_fn', default_metrics)
    save_dir = params.get('save_dir')
    if save_dir: os.makedirs(save_dir, exist_ok=True)

    results = []
    for step in range(1, len(permutation)+1):
        seen_idx = permutation[:step]
        seen_countries = [countries[i] for i in seen_idx]
        log.info(f"Step {step}: Countries {seen_countries}")

        # Build training DataLoader
        concat_train = ConcatDataset(train_sets[:step])
        concat_val = ConcatDataset(val_sets[:step])
        train_loader = DataLoader(concat_train, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(concat_val, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers)

        # TODO always reload pretrain weights
        model.load_fc_parameters(path_to_file)
        model.load_lora_parameters(path_to_file)
        #model = load_model(r=4)
        # Train
        start_train = time.time()
        model = train_model_replay(model,train_loader,val_loader,epochs=epoch,lr=lr)
        train_time = time.time() - start_train
        log.info(f"Training time: {train_time:.2f}s")

        # Save model
        if save_dir:
            path = os.path.join(save_dir, f"step{step}-{'-'.join(seen_countries)}.pkl")
            joblib.dump(model, path)
            log.info(f"Model saved: {path}")

        # Eval on all seen
        eval_metrics: Dict[str, float] = {}
        start_eval = time.time()
        for idx in seen_idx:
            country = countries[idx]
            test_loader = DataLoader(test_sets[idx], batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers)
            m = eval_model(model,test_loader)

            for name, val in m.items():
                eval_metrics[f"{country}_{name}"] = val
        eval_time = time.time() - start_eval
        log.info(f"Evaluation time: {eval_time:.2f}s")

        # Mean metrics
        sums, counts = defaultdict(float), defaultdict(int)
        for k, v in eval_metrics.items():
            _, met = k.split('_',1)
            sums[met] += v; counts[met] += 1
        mean_metrics = {f"mean_{m}": sums[m]/counts[m] for m in sums}

        row = {'step': step, 'countries': tuple(seen_countries),
               'train_time_s': train_time, 'eval_time_s': eval_time}
        row.update(eval_metrics); row.update(mean_metrics)
        results.append(row)
        log.info(f"Result: {row}")

    return pd.DataFrame(results)


def test_continual_learning_no_replay(
    model: Any,
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Continual Learning ohne Replay: Nur neues Land, evaluiere alle bisherigen.
    Nutzt DataLoader für Training und Test.
    """
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger(__name__)

    countries = params['countries']
    permutation = params['permutation']
    train_samples = params.get('train_samples')
    test_samples = params.get('test_samples')
    seed = params.get('seed', 0)
    batch_size = params.get('batch_size', 16)
    num_workers = params.get('num_workers', 4)
    epoch = params.get('epoch', 15)
    lr = float(params.get('lr', 1e-4))



    train_sets, val_sets = get_datasets(countries, permutation, split='train',
                                        img_size=params.get('img_size', (12, 128, 128)),
                                        include_snowy=params.get('include_snowy', False),
                                        include_cloudy=params.get('include_cloudy', False),
                                        samples_per_country=train_samples,
                                        seed=seed)
    test_sets, _ = get_datasets(countries, permutation, split='test',
                                img_size=params.get('img_size', (12, 128, 128)),
                                include_snowy=params.get('include_snowy', False),
                                include_cloudy=params.get('include_cloudy', False),
                                samples_per_country=test_samples,
                                seed=seed,
                                frac=1)

    save_dir = params.get('save_dir')
    metrics_fn = params.get('metrics_fn', default_metrics)

    if save_dir: os.makedirs(save_dir, exist_ok=True)

    results = []
    prev_model_path = None
    for step in range(1, len(permutation)+1):
        seen_idx = permutation[:step]
        seen_countries = [countries[i] for i in seen_idx]
        log.info(f"Step {step}: Countries {seen_countries}")

        # ToDO load lora weights from last step
        """
        # Init/load
        if step == 1: #ToDO load lora weights from last step
            model = model_class()
        else:
            model = joblib.load(prev_model_path)
        """

        # Train on new country DataLoader
        ds_new = train_sets[step-1]
        train_loader = DataLoader(ds_new, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
        ds_new_val = val_sets[step-1]
        val_loader = DataLoader(ds_new_val, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers)

        # Train
        start_train = time.time()
        model = train_model_replay(model,train_loader,val_loader,epochs=epoch,lr=lr)
        train_time = time.time() - start_train
        log.info(f"Training time: {train_time:.2f}s")

        if save_dir:
            path = os.path.join(save_dir, f"step{step}-{seen_countries[-1]}.pkl")
            joblib.dump(model, path)
            prev_model_path = path
            log.info(f"Saved: {path}")

        # Eval all seen
        eval_metrics = {}
        start_eval = time.time()
        for idx in seen_idx:
            country = countries[idx]
            test_loader = DataLoader(test_sets[idx], batch_size=batch_size,
                                     shuffle=False, num_workers=num_workers)
            m = eval_model(model, test_loader)
            for name, val in m.items(): eval_metrics[f"{country}_{name}"] = val
        eval_time = time.time() - start_eval
        log.info(f"Eval time: {eval_time:.2f}s")

        sums, counts = defaultdict(float), defaultdict(int)
        for k, v in eval_metrics.items(): _, met = k.split('_',1); sums[met]+=v; counts[met]+=1
        mean_metrics = {f"mean_{m}": sums[m]/counts[m] for m in sums}

        row = {'step': step, 'countries': tuple(seen_countries),
               'train_time_s': train_time, 'eval_time_s': eval_time}
        row.update(eval_metrics); row.update(mean_metrics)
        results.append(row)
        log.info(f"Result: {row}")

    return pd.DataFrame(results)



def main_from_config(config_path: str) -> pd.DataFrame:
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f) if config_path.endswith(('.yml', '.yaml')) else json.load(f)
    from importlib import import_module
    params = cfg['params']
    # get_dataset und metrics_fn unverändert: nutzen get_datasets intern
    if isinstance(params.get('metrics_fn'), str):
        mod, fn = params['metrics_fn'].rsplit(':', 1)
        params['metrics_fn'] = getattr(import_module(mod), fn)

    test_type = cfg['test_type']
    if test_type == 'replay':
        #mod, fac = cfg['model_module'].split(':', 1)
        #model = getattr(import_module(mod), fac)(**cfg.get('model_params', {}))
        model = load_model(4)
        return test_continual_learning(model,params)
    elif test_type == 'no_replay':
        #mod, cls = cfg['model_module'].split(':', 1)
        #model_class = getattr(import_module(mod), cls)
        model = load_model(4)
        return test_continual_learning_no_replay(model, params)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s %(levelname)s %(message)s"
    )
    logging.info("Logger initialized")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    args = parser.parse_args()

    setup_logging()
    logging.info(f"Config path: {args.config}")

    # ← simple dump of every line in the config into the log
    with open(args.config, "r", encoding="utf-8") as f:
        logging.info("Config file contents:\n%s", f.read())

    print(main_from_config(args.config))

if __name__ == "__main__":
    main()