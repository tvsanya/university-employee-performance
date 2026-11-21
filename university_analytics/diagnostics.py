"""Матрица ошибок, показатели классов и описательные срезы без дообучения."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from .baselines import load_split_bundle, scores
from .data import DatasetError
from .experiment import ExperimentConfig
from .portable import PortableNetwork
from .preprocessing import target_vector


def classification_diagnostics(truth, predicted, classes):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    if not len(truth) or len(truth) != len(predicted):
        raise DatasetError("Нужны непустые одинаковые по длине цели и прогнозы.")
    labels = sorted(set(classes) | set(truth) | set(predicted))
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=labels, zero_division=0)
    return {"metrics": scores("classification", truth, predicted), "labels": labels,
        "confusion_matrix": confusion_matrix(truth, predicted, labels=labels).tolist(),
        "per_class": [{"label": label, "precision": float(precision[i]), "recall": float(recall[i]),
            "f1": float(f1[i]), "support": int(support[i])} for i, label in enumerate(labels)],
        "error_count": int(np.sum(truth != predicted)),
        "error_rows": [{"source_row": int(i)+2, "true": str(truth[i]), "predicted": str(predicted[i])}
            for i in np.flatnonzero(truth != predicted)[:50]]}


def diagnose_dataset(dataset, config, model, minimum_slice=20):
    truth = target_vector(dataset, config)
    predicted = model.predict(dataset)
    report = classification_diagnostics(truth, predicted, model.classes)
    slices, suppressed = [], 0
    for feature in config.categorical_features:
        index = dataset.columns.index(feature)
        groups = {}
        for row_index, row in enumerate(dataset.rows):
            groups.setdefault(row[index] or "(пропуск)", []).append(row_index)
        for value, indices in sorted(groups.items()):
            if len(indices) < minimum_slice:
                suppressed += 1
                continue
            slices.append({"feature": feature, "value": value, "rows": len(indices),
                "metrics": scores("classification", truth[indices], predicted[indices])})
    return {**report, "slices": slices, "minimum_slice_rows": minimum_slice, "suppressed_slices": suppressed}


def run_diagnostics(split_directory, experiment_directory, destination):
    root, destination = Path(experiment_directory), Path(destination)
    if destination.exists():
        raise DatasetError("Каталог диагностики уже существует.")
    try:
        metrics = json.loads((root/"metrics.json").read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError("Повреждён отчёт эксперимента.") from exc
    if not isinstance(metrics, dict) or not isinstance(metrics.get("portable_artifact"), dict):
        raise DatasetError("Эксперимент не содержит JSON-модель.")
    artifact = metrics["portable_artifact"]
    filename = artifact.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".json"):
        raise DatasetError("Недопустимое имя JSON-модели.")
    model = PortableNetwork.from_file(root/filename)
    if model.sha256 != artifact.get("sha256"):
        raise DatasetError("Контрольная сумма модели не совпадает с отчётом.")
    config = ExperimentConfig.from_file(root/"config.json")
    if model.config != config:
        raise DatasetError("Контракт модели не совпадает с экспериментом.")
    datasets, manifest = load_split_bundle(split_directory, config)
    split_sha = hashlib.sha256((Path(split_directory)/"manifest.json").read_bytes()).hexdigest()
    if metrics.get("source_sha256") != manifest.get("source_sha256") or metrics.get("split_manifest_sha256") != split_sha:
        raise DatasetError("Эксперимент и выборки относятся к разным данным или разбиениям.")
    report = {"source_sha256": manifest["source_sha256"], "split_manifest_sha256": split_sha,
        "model_sha256": model.sha256, "data_origin": model.data_origin, "intended_use": model.intended_use,
        "partitions": {name: diagnose_dataset(data, config, model) for name, data in datasets.items()},
        "limitation": "Срезы описывают выбранную выборку; различия не доказывают причинность или справедливость модели."}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".diagnostics-", dir=destination.parent))
    try:
        (temporary/"diagnostics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.rename(destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
