"""Прогноз без повторного обучения с происхождением модели в результате."""
import csv
import json
import shutil
import tempfile
from pathlib import Path
import numpy as np
from .data import DatasetError, load_dataset
from .portable import PortableNetwork


def spreadsheet_text(value):
    text = str(value)
    return "'"+text if text.lstrip().startswith(("=", "+", "-", "@")) else text


def run_prediction(model_path, source, destination, *, max_rows=None, max_classes=None, review_threshold=.7):
    if type(review_threshold) not in (float, int) or not np.isfinite(review_threshold) or not 0 <= review_threshold <= 1:
        raise DatasetError("Порог просмотра должен быть конечным числом от 0 до 1.")
    destination = Path(destination)
    if destination.exists():
        raise DatasetError("Каталог прогнозов уже существует; выберите новый путь.")
    model = PortableNetwork.from_file(model_path)
    dataset = load_dataset(source)
    if max_rows is not None and len(dataset.rows) > max_rows:
        raise DatasetError(f"В интерфейсе допустимо до {max_rows} строк за прогноз.")
    if max_classes is not None and len(model.classes) > max_classes:
        raise DatasetError(f"В интерфейсе допустимо до {max_classes} категорий модели.")
    if not dataset.rows:
        raise DatasetError("Набор для прогноза пуст.")
    probabilities = model.predict_proba(dataset)
    predictions = np.asarray(model.classes)[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    entropy = np.clip(-np.sum(probabilities*np.log(np.clip(probabilities, 1e-15, 1)), axis=1)/np.log(len(model.classes)), 0, 1)
    review = confidence < review_threshold
    report = {"rows": len(predictions), "classes": model.classes, "source_sha256": dataset.sha256,
        "model_sha256": model.sha256,
        "data_origin": model.data_origin, "input_data_origin": dataset.origin_kind,
        "intended_use": "development_demo_only" if dataset.origin_kind in {"synthetic", "mixed"} else model.intended_use,
        "probability_columns": {f"probability_{number+1}": name for number, name in enumerate(model.classes)},
        "review_policy": {"threshold": float(review_threshold), "comparison": "max_probability < threshold",
            "review_rows": int(review.sum()), "coverage_above_threshold": float((~review).mean()),
            "limitation": "Порог задаётся пользователем без подбора по test. Максимальная вероятность не является подтверждённой калиброванной уверенностью. Флаг помогает просмотреть результаты и не принимает кадровых решений."}}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".prediction-", dir=destination.parent))
    try:
        group = model.config.group_column
        group_index = dataset.columns.index(group) if group in dataset.columns else None
        with (temporary/"predictions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["source_row", *([group] if group_index is not None else []), "predicted_class", "max_probability", "normalized_entropy", "review_recommended", *report["probability_columns"]])
            for number, (row, predicted, probabilities_row) in enumerate(zip(dataset.rows, predictions, probabilities), 2):
                index = number-2
                writer.writerow([number, *([spreadsheet_text(row[group_index])] if group_index is not None else []), spreadsheet_text(predicted),
                    f"{confidence[index]:.12g}", f"{entropy[index]:.12g}", "yes" if review[index] else "no", *[f"{value:.12g}" for value in probabilities_row]])
        (temporary/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        temporary.rename(destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
