"""Прогноз без повторного обучения с происхождением модели в результате."""
import csv
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
from .data import DatasetError, load_dataset
from .portable import PortableNetwork


def run_prediction(model_path, source, destination):
    destination = Path(destination)
    if destination.exists():
        raise DatasetError("Каталог прогнозов уже существует; выберите новый путь.")
    model = PortableNetwork.from_file(model_path)
    dataset = load_dataset(source)
    if not dataset.rows:
        raise DatasetError("Набор для прогноза пуст.")
    probabilities = model.predict_proba(dataset)
    predictions = np.asarray(model.classes)[probabilities.argmax(axis=1)]
    report = {"rows": len(predictions), "classes": model.classes, "source_sha256": dataset.sha256,
        "model_sha256": model.sha256,
        "data_origin": model.data_origin, "input_data_origin": dataset.origin_kind,
        "intended_use": "development_demo_only" if dataset.origin_kind in {"synthetic", "mixed"} else model.intended_use,
        "probability_columns": {f"probability_{number+1}": name for number, name in enumerate(model.classes)}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prediction-", dir=destination.parent))
    try:
        group = model.config.group_column
        group_index = dataset.columns.index(group) if group in dataset.columns else None
        with (temporary/"predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["source_row", *([group] if group_index is not None else []), "predicted_class", *report["probability_columns"]])
            for number, (row, predicted, probabilities_row) in enumerate(zip(dataset.rows, predictions, probabilities), 2):
                writer.writerow([number, *([row[group_index]] if group_index is not None else []), predicted, *[f"{value:.12g}" for value in probabilities_row]])
        (temporary/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.rename(destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
