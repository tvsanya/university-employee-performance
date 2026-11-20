"""Контроль качества перед выбором и обучением модели."""
import math
from .data import Dataset, basic_profile

MISSING_VALUES = {"", "na", "n/a", "null", "none"}


def is_missing(value: str) -> bool:
    return value.casefold() in MISSING_VALUES


def profile_dataset(dataset: Dataset, target: str | None = None) -> dict:
    report = basic_profile(dataset)
    report["duplicate_rows"] = len(dataset.rows) - len(set(dataset.rows))
    profiles = {}
    for index, column in enumerate(dataset.columns):
        values = [row[index] for row in dataset.rows]
        present = [value for value in values if not is_missing(value)]
        numbers = []
        nonfinite = 0
        textual = 0
        for value in present:
            try:
                number = float(value)
            except ValueError:
                textual += 1
            else:
                if math.isfinite(number):
                    numbers.append(number)
                else:
                    nonfinite += 1
        numeric = bool(present) and textual == 0
        profiles[column] = {
            "kind": "numeric" if numeric else "categorical",
            "missing": len(values) - len(present), "unique": len(set(present)),
            "nonfinite": nonfinite if numeric else 0,
            "min": min(numbers) if numeric and numbers else None,
            "max": max(numbers) if numeric and numbers else None,
        }
    report["column_profiles"] = profiles
    report["target"] = target
    return report


def validate_dataset(dataset: Dataset, target: str, *, min_rows=1000, min_columns=10) -> dict:
    """Проверяет структуру и пригодность меток, а не утверждает бизнес-модель."""
    report = profile_dataset(dataset, target)
    issues = []
    warnings = []
    if len(dataset.rows) < min_rows:
        issues.append(f"Требуется минимум {min_rows} строк; получено {len(dataset.rows)}.")
    if len(dataset.columns) < min_columns:
        issues.append(f"Требуется минимум {min_columns} колонок; получено {len(dataset.columns)}.")
    if target not in dataset.columns:
        issues.append(f"Целевая колонка {target!r} не найдена.")
    else:
        info = report["column_profiles"][target]
        if info["missing"]:
            issues.append("В целевой колонке есть пропуски.")
        if info["unique"] < 2 or (info["kind"] == "numeric" and info["min"] == info["max"]):
            issues.append("Целевая колонка должна содержать разные значения.")
    if report["duplicate_rows"]:
        warnings.append(f"Обнаружено повторов строк: {report['duplicate_rows']}. Учитывайте их при разделении выборок.")
    for column, info in report["column_profiles"].items():
        if info["nonfinite"]:
            issues.append(f"Колонка {column!r} содержит NaN или бесконечность.")
        if column != target and info["missing"]:
            warnings.append(f"Колонка {column!r}: пропусков {info['missing']}; нужен способ обработки.")
        if column != target and info["unique"] <= 1:
            warnings.append(f"Колонка {column!r} не содержит вариативных значений.")
    report.update({"ok": not issues, "issues": issues, "warnings": warnings})
    return report
