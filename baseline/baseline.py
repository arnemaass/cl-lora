import time
import logging
from typing import Any, Dict, List, Callable
import pandas as pd
from collections import defaultdict
import os
import joblib
import json
import yaml


def test_continual_learning(
    model: Any,
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Führt eine Continual Learning Evaluation mit vollständigem Replay durch.
    Siehe Dokumentation oben.
    """
    log = logging.getLogger(__name__)
    if params.get('log_every_step', True):
        logging.basicConfig(level=logging.INFO)

    countries: List[str] = params['countries']
    get_dataset: Callable[[str], Dict[str, pd.DataFrame]] = params['get_dataset']
    metrics_fn = params['metrics_fn']
    save_dir = params.get('save_dir')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    seen_countries: List[str] = []

    for step, country in enumerate(countries, start=1):
        seen_countries.append(country)
        log.info(f"=== Step {step}: Training on {seen_countries} ===")

        # Combine training data for seen countries
        X_train = pd.concat([get_dataset(c)['X_train'] for c in seen_countries], axis=0)
        y_train = pd.concat([get_dataset(c)['y_train'] for c in seen_countries], axis=0)

        # Training
        start_train = time.time()
        model.fit(X_train, y_train, **params.get('fit_kwargs', {}))
        train_time = time.time() - start_train
        log.info(f"Training time for step {step}: {train_time:.2f}s")

        # Save model
        if save_dir:
            slug = "-".join(seen_countries)
            path = os.path.join(save_dir, f"step{step}_{slug}.pkl")
            joblib.dump(model, path)
            log.info(f"Model saved to {path}")

        # Evaluation on all seen countries
        eval_metrics: Dict[str, float] = {}
        start_eval = time.time()
        for c in seen_countries:
            ds = get_dataset(c)
            preds = model.predict(ds['X_test'])
            m = metrics_fn(ds['y_test'], preds)
            for name, val in m.items():
                eval_metrics[f"{c}_{name}"] = val
        eval_time = time.time() - start_eval
        log.info(f"Evaluation time for step {step}: {eval_time:.2f}s")

        # Compute mean metrics
        sums = defaultdict(float)
        counts = defaultdict(int)
        for k, v in eval_metrics.items():
            _, metric = k.split('_', 1)
            sums[metric] += v; counts[metric] += 1
        mean_metrics = {f"mean_{m}": sums[m]/counts[m] for m in sums}

        # Collect results
        row = {
            'step': step,
            'countries': tuple(seen_countries),
            'train_time_s': train_time,
            'eval_time_s': eval_time,
        }
        row.update(eval_metrics)
        row.update(mean_metrics)
        results.append(row)
        log.info(f"Step {step} results: {row}")

    return pd.DataFrame(results)


def test_continual_learning_no_replay(
    model_class: Callable[[], Any],
    params: Dict[str, Any]
) -> pd.DataFrame:
    """
    Continual Learning ohne Replay: Trainiere sequentiell nur auf neuen Ländern,
    evaluiere nach jedem Schritt auf allen bisher gesehenen Ländern.
    """
    log = logging.getLogger(__name__)
    if params.get('log_every_step', True):
        logging.basicConfig(level=logging.INFO)

    countries: List[str] = params['countries']
    get_dataset: Callable[[str], Dict[str, pd.DataFrame]] = params['get_dataset']
    metrics_fn = params['metrics_fn']
    save_dir = params.get('save_dir')
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    results = []
    seen_countries: List[str] = []
    prev_model_path = None

    for step, country in enumerate(countries, start=1):
        seen_countries.append(country)
        log.info(f"=== Step {step}: Training on [{country}] ===")

        # Initialize or load model
        if step == 1:
            model = model_class()
        else:
            model = joblib.load(prev_model_path)

        # Train on new data
        ds = get_dataset(country)
        start_train = time.time()
        model.fit(ds['X_train'], ds['y_train'], **params.get('fit_kwargs', {}))
        train_time = time.time() - start_train
        log.info(f"Training time: {train_time:.2f}s")

        # Save model
        if save_dir:
            path = os.path.join(save_dir, f"step{step}_{country}.pkl")
            joblib.dump(model, path)
            prev_model_path = path
            log.info(f"Model saved: {path}")

        # Evaluation on all seen countries
        eval_metrics: Dict[str, float] = {}
        start_eval = time.time()
        for c in seen_countries:
            di = get_dataset(c)
            preds = model.predict(di['X_test'])
            m = metrics_fn(di['y_test'], preds)
            for name, val in m.items():
                eval_metrics[f"{c}_{name}"] = val
        eval_time = time.time() - start_eval
        log.info(f"Evaluation time: {eval_time:.2f}s")

        # Compute mean metrics
        sums = defaultdict(float)
        counts = defaultdict(int)
        for k, v in eval_metrics.items():
            _, metric = k.split('_', 1)
            sums[metric] += v; counts[metric] += 1
        mean_metrics = {f"mean_{m}": sums[m]/counts[m] for m in sums}

        # Collect results
        row = {
            'step': step,
            'countries': tuple(seen_countries),
            'train_time_s': train_time,
            'eval_time_s': eval_time,
        }
        row.update(eval_metrics)
        row.update(mean_metrics)
        results.append(row)
        log.info(f"Step {step} results: {row}")

    return pd.DataFrame(results)


def main_from_config(config_path: str) -> pd.DataFrame:
    """
    Lädt eine Konfigurationsdatei (YAML oder JSON) und führt den angegebenen Test aus.

    Die Config-Datei muss enthalten:
      test_type: 'replay' oder 'no_replay'
      model_module: Pfad zum Modellmodul, z.B. 'my_models'
      model_class: Name der Modellklasse (für no_replay) oder Instanz-Fabrik
      params:
        countries: List[str]
        get_dataset: importpfad oder Callable
        metrics_fn: importpfad oder Callable
        fit_kwargs: Dict
        save_dir: str (optional)
        log_every_step: bool

    Returns:
        DataFrame mit den Evaluationsergebnissen.
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f) if config_path.endswith(('.yml', '.yaml')) else json.load(f)

    # Dynamisches Importieren
    from importlib import import_module
    params = cfg['params']
    # Resolve get_dataset und metrics_fn
    if isinstance(params['get_dataset'], str):
        mod, fn = params['get_dataset'].rsplit(':', 1)
        params['get_dataset'] = getattr(import_module(mod), fn)
    if isinstance(params['metrics_fn'], str):
        mod, fn = params['metrics_fn'].rsplit(':', 1)
        params['metrics_fn'] = getattr(import_module(mod), fn)

    test_type = cfg['test_type']
    if test_type == 'replay':
        # Model instance
        model_path = cfg['model_module'] + ':' + cfg.get('model_factory', 'get_model')
        mod, fac = model_path.split(':', 1)
        model = getattr(import_module(mod), fac)(**cfg.get('model_params', {}))
        return test_continual_learning(model, params)
    elif test_type == 'no_replay':
        model_path = cfg['model_module'] + ':' + cfg['model_class']
        mod, cls = model_path.split(':', 1)
        model_class = getattr(import_module(mod), cls)
        return test_continual_learning_no_replay(model_class, params)
    else:
        raise ValueError(f"Unknown test_type: {test_type}")
