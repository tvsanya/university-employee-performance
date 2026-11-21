"""Бакулин Александр Максимович.

Автоматизация процессов оценки эффективности работы сотрудника университета
с применением интеллектуального анализа данных.
Сравнение простого и линейного baseline без обучения на heldout-данных.
"""
import hashlib
import json
import math
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from .data import Dataset, DatasetError, load_dataset
from .experiment import ExperimentConfig
from .preprocessing import build_preprocessor, feature_matrix, target_vector
from .quality import validate_dataset

PARTITIONS = ("train", "validation", "test")


def load_split_bundle(directory: Path, config: ExperimentConfig):
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError("Манифест выборок не является корректным JSON.") from exc
    if not isinstance(manifest, dict) or manifest.get("target") != config.target or manifest.get("group_column") != config.group_column:
        raise DatasetError("Цель или группирующий столбец не совпадают с манифестом выборок.")
    datasets = {}
    for name in PARTITIONS:
        data = load_dataset(str(directory / f"{name}.csv"))
        config.check_columns(data)
        expected = manifest.get("outputs", {}).get(name, {})
        if data.sha256 != expected.get("sha256") or len(data.rows) != expected.get("rows"):
            raise DatasetError(f"Контрольная сумма или размер {name} не совпадают с манифестом.")
        if not data.rows:
            raise DatasetError(f"Выборка {name} пуста.")
        if data.origin_kind != manifest.get("data_origin", "unspecified"):
            raise DatasetError("Происхождение данных не совпадает с манифестом.")
        if datasets and data.columns != datasets["train"].columns:
            raise DatasetError("Структура колонок выборок различается.")
        datasets[name] = data
    keys = []
    for data in datasets.values():
        index = data.columns.index(config.group_column) if config.group_column else None
        keys.append({row[index] if index is not None else row for row in data.rows})
    if keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2]:
        raise DatasetError("Между выборками пересекаются группы или одинаковые строки.")
    combined = Dataset(datasets["train"].columns, tuple(row for data in datasets.values() for row in data.rows), manifest.get("source", ""), manifest.get("source_sha256", ""))
    checks = validate_dataset(combined, config.target)
    if not checks["ok"]:
        raise DatasetError("; ".join(checks["issues"]))
    return datasets, manifest


def scores(task, truth, predicted, labels=None):
    if task == "classification":
        classes = np.unique(np.concatenate([np.asarray(labels if labels is not None else []), np.asarray(truth), np.asarray(predicted)]))
        return {"accuracy": float(accuracy_score(truth, predicted)), "macro_f1": float(f1_score(truth, predicted, labels=classes, average="macro", zero_division=0))}
    if not np.isfinite(predicted).all():
        raise DatasetError("Модель вернула нечисловой или бесконечный прогноз.")
    return {"mae": float(mean_absolute_error(truth, predicted)),
            "rmse": math.sqrt(float(mean_squared_error(truth, predicted))),
            "r2": float(r2_score(truth, predicted)) if len(truth) > 1 and np.ptp(truth) > 0 else None}


def fit_baselines(datasets: dict[str, Dataset], config: ExperimentConfig):
    """Низкоуровневый конвейер; небольшие фикстуры допустимы только в тестах."""
    config.validate()
    matrices, targets = {}, {}
    for name in PARTITIONS:
        data = datasets[name]
        if not data.rows:
            raise DatasetError(f"Выборка {name} пуста.")
        config.check_columns(data)
        matrices[name] = feature_matrix(data, config)
        targets[name] = target_vector(data, config)
    if config.task == "classification":
        if len(np.unique(targets["train"])) < 2:
            raise DatasetError("В train должны быть минимум два класса для сравнения baseline.")
        estimators = {"dummy": DummyClassifier(strategy="most_frequent"),
                      "linear": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=config.seed)}
    else:
        estimators = {"dummy": DummyRegressor(strategy="mean"), "linear": Ridge(alpha=1.0)}
    models, evaluations = {}, {}
    for name, estimator in estimators.items():
        model = Pipeline([("preprocess", build_preprocessor(config)), ("model", estimator)])
        try:
            model.fit(matrices["train"], targets["train"])
        except ValueError as exc:
            raise DatasetError(f"Не удалось обучить {name}: {exc}") from exc
        evaluations[name] = {part: scores(config.task, targets[part], model.predict(matrices[part]),
            labels=np.unique(targets["train"]) if config.task == "classification" else None) for part in PARTITIONS}
        models[name] = model
    metric = "macro_f1" if config.task == "classification" else "mae"
    selected = min(evaluations, key=lambda name: -evaluations[name]["validation"][metric] if config.task == "classification" else evaluations[name]["validation"][metric])
    return models, {"task": config.task, "target_description": config.target_description,
                    "selection_partition": "validation", "selection_metric": metric,
                    "selected_model": selected, "metrics": evaluations,
                    "rows": {name: len(data.rows) for name, data in datasets.items()}}


def run_baseline_experiment(split_directory: Path, config: ExperimentConfig, destination: Path):
    destination = Path(destination)
    if destination.exists():
        raise DatasetError("Каталог эксперимента уже существует; выберите новый путь.")
    datasets, manifest = load_split_bundle(split_directory, config)
    models, report = fit_baselines(datasets, config)
    report["source_sha256"] = manifest.get("source_sha256")
    report["data_origin"] = manifest.get("data_origin", "unspecified")
    report["intended_use"] = "development_demo_only" if report["data_origin"] in {"synthetic", "mixed"} else "research_requires_validation"
    report["split_manifest_sha256"] = hashlib.sha256((Path(split_directory) / "manifest.json").read_bytes()).hexdigest()
    report["versions"] = {name: version(name) for name in ["numpy", "scikit-learn", "joblib", "scipy"]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".experiment-", dir=destination.parent))
    try:
        artifacts = {}
        for number, (name, model) in enumerate(models.items(), 1):
            filename = f"model{number}.joblib"
            joblib.dump(model, temporary / filename)
            artifacts[name] = {"file": filename, "sha256": hashlib.sha256((temporary / filename).read_bytes()).hexdigest()}
        report["artifacts"] = artifacts
        if config.task == "classification":
            from .portable import export_baseline
            portable = export_baseline(models[report["selected_model"]], config, report["data_origin"])
            (temporary/"model.json").write_text(json.dumps(portable, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
            report["portable_artifact"] = {"file": "model.json", "sha256": hashlib.sha256((temporary/"model.json").read_bytes()).hexdigest()}
        (temporary / "config.json").write_text(json.dumps(config.as_dict(), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        (temporary / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.rename(destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
