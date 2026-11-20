"""Воспроизводимое разбиение без пересечения групп и одинаковых строк."""
import csv
import hashlib
import json
import math
import random
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from .data import Dataset, DatasetError
from .quality import is_missing


@dataclass(frozen=True)
class SplitResult:
    partitions: dict[str, tuple[tuple[str, ...], ...]]
    group_counts: dict[str, int]
    seed: int
    train_fraction: float
    validation_fraction: float
    group_column: str | None


def split_dataset(dataset: Dataset, *, seed=42, train_fraction=0.7,
                  validation_fraction=0.15, group_column=None) -> SplitResult:
    fractions = (train_fraction, validation_fraction)
    if any(not math.isfinite(value) or not 0 < value < 1 for value in fractions) or sum(fractions) >= 1:
        raise DatasetError("Доли train и validation должны быть положительными и оставлять непустой test.")
    if group_column and group_column not in dataset.columns:
        raise DatasetError(f"Группирующая колонка {group_column!r} не найдена.")
    group_index = dataset.columns.index(group_column) if group_column else None
    groups = defaultdict(list)
    for row in dataset.rows:
        if group_index is not None:
            key = row[group_index]
            if is_missing(key):
                raise DatasetError("В группирующей колонке есть пропуски.")
        else:
            # Даже без идентификатора точные дубли остаются в одной выборке.
            key = json.dumps(row, ensure_ascii=True)
        groups[key].append(row)
    if len(groups) < 3:
        raise DatasetError("Для трёх непустых выборок нужны минимум три независимые группы.")
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    count = len(keys)
    train_count = min(max(1, int(count * train_fraction)), count - 2)
    validation_count = min(max(1, int(count * validation_fraction)), count - train_count - 1)
    selections = {
        "train": keys[:train_count],
        "validation": keys[train_count:train_count + validation_count],
        "test": keys[train_count + validation_count:],
    }
    partitions = {name: tuple(row for key in selected for row in groups[key]) for name, selected in selections.items()}
    return SplitResult(partitions, {name: len(selected) for name, selected in selections.items()}, seed,
                       train_fraction, validation_fraction, group_column)


def write_splits(dataset: Dataset, result: SplitResult, destination: Path, *, target=None) -> dict:
    """Сохраняет три CSV и манифест, не перезаписывая существующий результат."""
    destination = Path(destination)
    if destination.exists():
        raise DatasetError("Каталог результата уже существует; выберите новый путь.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".split-", dir=destination.parent))
    try:
        outputs = {}
        for name, rows in result.partitions.items():
            path = temporary / f"{name}.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream, lineterminator="\n")
                writer.writerow(dataset.columns)
                writer.writerows(rows)
            outputs[name] = {"rows": len(rows), "groups": result.group_counts[name],
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        manifest = {
            "algorithm": "group-shuffle-v1", "source": dataset.source, "source_sha256": dataset.sha256,
            "data_origin": dataset.origin_kind,
            "seed": result.seed, "group_column": result.group_column, "target": target,
            "non_feature_columns": [name for name in [target, result.group_column] if name],
            "requested_group_fractions": {"train": result.train_fraction, "validation": result.validation_fraction,
                                          "test": 1 - result.train_fraction - result.validation_fraction},
            "outputs": outputs,
            "notes": ["Доли задаются по независимым группам; доли строк могут отличаться.",
                      "Подбор параметров и преобразований выполняется на train, test сохраняется для итоговой проверки."],
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.rename(destination)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
