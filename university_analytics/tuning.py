"""Ограниченный поиск архитектуры и параметров только по validation."""
import hashlib
import json
import shutil
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
import joblib
from .baselines import PARTITIONS, load_split_bundle, scores
from .data import DatasetError
from .neural import NetworkConfig, train_network
from .preprocessing import feature_matrix, target_vector
from .portable import export_network


def load_network_grid(path, seed=42):
    try:
        values = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError("Сетка параметров не является корректным JSON.") from exc
    if not isinstance(values, list) or not 1 <= len(values) <= 8:
        raise DatasetError("Укажите список из 1–8 конфигураций сети.")
    result, fingerprints = [], set()
    for value in values:
        if not isinstance(value, dict):
            raise DatasetError("Каждый вариант сети должен быть объектом параметров.")
        parameters = NetworkConfig.from_dict({"seed": seed, **value})
        fingerprint = json.dumps(parameters.as_dict(), sort_keys=True)
        if fingerprint in fingerprints:
            raise DatasetError("В сетке повторяется одна и та же конфигурация.")
        fingerprints.add(fingerprint)
        result.append(parameters)
    return result


def fit_neural_candidates(datasets, config, candidates):
    if not 1 <= len(candidates) <= 8:
        raise DatasetError("Для поиска допустимо от одного до восьми вариантов.")
    trials, selected_model, selected_key, selected_number = [], None, None, None
    for number, parameters in enumerate(candidates, 1):
        model, training = train_network(datasets["train"], datasets["validation"], config, parameters)
        trials.append({"candidate": number, **training})
        key = (training["best_validation_macro_f1"], -training["architecture"]["parameter_count"])
        if selected_key is None or key > selected_key:
            selected_model, selected_key, selected_number = model, key, number
    # Test впервые применяется после окончательного выбора сети.
    evaluations = {name: scores(config.task, target_vector(datasets[name], config),
        selected_model.predict(feature_matrix(datasets[name], config))) for name in PARTITIONS}
    chosen = trials[selected_number-1]
    report = {"task": config.task, "target_description": config.target_description,
        "selected_model": "neural", "selected_candidate": selected_number,
        "selection_partition": "validation", "selection_metric": "macro_f1",
        "tie_breaker": "fewer_parameters_then_first_candidate",
        "architecture": chosen["architecture"], "parameters": chosen["parameters"],
        "best_epoch": chosen["best_epoch"], "metrics": {"neural": evaluations},
        "rows": {name: len(datasets[name].rows) for name in PARTITIONS},
        "candidates": [{key: item[key] for key in ["candidate", "parameters", "best_epoch", "epochs_run", "stop_reason", "best_validation_macro_f1"]} for item in trials]}
    return selected_model, report, {"selected_candidate": selected_number, "trials": trials}


def run_neural_experiment(split_directory, config, candidates, destination):
    destination = Path(destination)
    if destination.exists():
        raise DatasetError("Каталог эксперимента уже существует; выберите новый путь.")
    datasets, manifest = load_split_bundle(split_directory, config)
    started = time.perf_counter()
    model, report, training = fit_neural_candidates(datasets, config, candidates)
    report.update({"runtime_seconds": round(time.perf_counter()-started, 3),
        "source_sha256": manifest.get("source_sha256"), "data_origin": manifest.get("data_origin", "unspecified"),
        "split_manifest_sha256": hashlib.sha256((Path(split_directory)/"manifest.json").read_bytes()).hexdigest(),
        "versions": {name: version(name) for name in ["numpy", "scikit-learn", "joblib", "scipy"]}})
    report["intended_use"] = "development_demo_only" if report["data_origin"] in {"synthetic", "mixed"} else "research_requires_validation"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".neural-", dir=destination.parent))
    try:
        joblib.dump(model, temporary/"model3.joblib")
        report["artifacts"] = {"neural": {"file": "model3.joblib", "sha256": hashlib.sha256((temporary/"model3.joblib").read_bytes()).hexdigest()}}
        (temporary/"model.json").write_text(json.dumps(export_network(model, config, report["data_origin"]), ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        report["portable_artifact"] = {"file": "model.json", "sha256": hashlib.sha256((temporary/"model.json").read_bytes()).hexdigest()}
        for filename, contents in [("config.json", config.as_dict()), ("training.json", training),
                ("grid.json", [item.as_dict() for item in candidates])]:
            (temporary/filename).write_text(json.dumps(contents, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        report["training_artifact"] = {"file": "training.json", "sha256": hashlib.sha256((temporary/"training.json").read_bytes()).hexdigest()}
        (temporary/"metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.rename(destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
