"""Сравнение сохранённых семейств по одному контракту и одним выборкам."""
import csv
import hashlib
import json
import math
import shutil
import tempfile
from pathlib import Path
from .data import DatasetError
from .experiment import ExperimentConfig


def compare_experiments(directories):
    if not 2 <= len(directories) <= 20:
        raise DatasetError("Укажите 2–20 экспериментов для сравнения.")
    if len({Path(path).resolve() for path in directories}) != len(directories):
        raise DatasetError("Один эксперимент указан несколько раз.")
    common, entries = None, []
    for number, directory in enumerate(directories, 1):
        root = Path(directory)
        try:
            report = json.loads((root/"metrics.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetError("Повреждён отчёт эксперимента.") from exc
        config = ExperimentConfig.from_file(root/"config.json")
        if not isinstance(report, dict) or not isinstance(report.get("metrics"), dict):
            raise DatasetError("В отчёте нет результатов моделей.")
        fingerprint = {key: report.get(key) for key in ["source_sha256", "split_manifest_sha256", "data_origin", "rows"]}
        fingerprint["config"] = config.as_dict()
        if any(not isinstance(fingerprint[key], str) or len(fingerprint[key]) != 64 for key in ["source_sha256", "split_manifest_sha256"]):
            raise DatasetError("В отчёте не хватает контрольных сумм данных и разбиения.")
        if common is not None and fingerprint != common:
            raise DatasetError("Нельзя сравнивать разные данные, выборки, контракты или происхождение.")
        common = fingerprint
        metric = "macro_f1" if config.task == "classification" else "mae"
        for name, metrics in report["metrics"].items():
            if not isinstance(metrics, dict):
                raise DatasetError("Некорректные метрики модели.")
            for partition in ["train", "validation", "test"]:
                value = metrics.get(partition, {}).get(metric) if isinstance(metrics.get(partition), dict) else None
                if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or (metric == "macro_f1" and value > 1):
                    raise DatasetError("Отсутствует конечная метрика каждой выборки.")
            entries.append({"id": f"{number}:{name}", "experiment": root.name, "model": name, "metrics": metrics,
                "report_sha256": hashlib.sha256((root/"metrics.json").read_bytes()).hexdigest()})
    if not entries:
        raise DatasetError("Нет моделей для сравнения.")
    selected = min(entries, key=lambda item: -item["metrics"]["validation"][metric] if metric == "macro_f1" else item["metrics"]["validation"][metric])
    return {**common, "config": common["config"], "selection_partition": "validation", "selection_metric": metric,
        "tie_breaker": "first_equal_validation", "selected": selected["id"], "entries": entries,
        "intended_use": "development_demo_only" if common["data_origin"] in {"synthetic", "mixed"} else "research_requires_validation"}


def run_comparison(directories, destination):
    destination = Path(destination)
    if destination.exists():
        raise DatasetError("Каталог сравнения уже существует.")
    report = compare_experiments(directories)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".comparison-", dir=destination.parent))
    try:
        (temporary/"comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
        metric = report["selection_metric"]
        lines = ["# Сравнение моделей", "", f"Происхождение: {report['data_origin']}. Назначение: {report['intended_use']}.",
            "Выбор по validation; test не участвует в ранжировании.", "", f"| Модель | Train {metric} | Validation | Test |", "| --- | ---: | ---: | ---: |"]
        with (temporary/"comparison.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["id", "model", "train", "validation", "test", "selected_by_validation"])
            for item in report["entries"]:
                values = [item["metrics"][part][metric] for part in ["train", "validation", "test"]]
                writer.writerow([item["id"], item["model"], *values, item["id"] == report["selected"]])
                label = (item["experiment"]+" / "+item["model"]).replace("|", "\\|")
                lines.append("| "+label+" | "+" | ".join(f"{value:.5f}" for value in values)+" |")
        lines += ["", "Выбрана модель: "+report["selected"]+". При равенстве сохраняется первый вариант."]
        (temporary/"comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        temporary.rename(destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
