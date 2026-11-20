"""Явное описание задачи и допустимых признаков эксперимента."""
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from .data import Dataset, DatasetError


def _names(value, field):
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) or not item.strip() for item in value):
        raise DatasetError(f"{field}: требуется список непустых названий колонок.")
    return tuple(value)


@dataclass(frozen=True)
class ExperimentConfig:
    task: str
    target: str
    target_description: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...] = ()
    group_column: str | None = None
    excluded_columns: tuple[str, ...] = ()
    seed: int = 42

    @property
    def features(self):
        return tuple(self.numeric_features) + tuple(self.categorical_features)

    def validate(self):
        if self.task not in {"classification", "regression"}:
            raise DatasetError("Тип задачи: classification или regression.")
        if not isinstance(self.target, str) or not self.target.strip():
            raise DatasetError("Нужно указать целевую колонку.")
        if not isinstance(self.target_description, str) or not self.target_description.strip():
            raise DatasetError("Опишите смысл и способ получения целевого показателя.")
        for field in ["numeric_features", "categorical_features", "excluded_columns"]:
            _names(getattr(self, field), field)
        if not self.features or len(set(self.features)) != len(self.features):
            raise DatasetError("Список признаков должен быть непустым и без повторов.")
        if self.group_column is not None and (not isinstance(self.group_column, str) or not self.group_column.strip()):
            raise DatasetError("group_column должен быть непустым названием либо null.")
        forbidden = {self.target, *self.excluded_columns}
        if self.group_column:
            forbidden.add(self.group_column)
            if self.group_column == self.target:
                raise DatasetError("Идентификатор группы не может быть целевым показателем.")
        if set(self.features) & forbidden:
            raise DatasetError("Цель, идентификатор группы и исключённые колонки не могут быть признаками.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise DatasetError("seed должен быть целым числом.")
        return self

    def check_columns(self, dataset: Dataset):
        self.validate()
        required = {self.target, *self.features}
        if self.group_column:
            required.add(self.group_column)
        missing = required - set(dataset.columns)
        if missing:
            raise DatasetError("Отсутствуют колонки эксперимента: " + ", ".join(sorted(missing)))

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict):
            raise DatasetError("Конфигурация должна быть JSON-объектом.")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise DatasetError("Неизвестные параметры: " + ", ".join(sorted(unknown)))
        values = dict(values)
        for name in ["numeric_features", "categorical_features", "excluded_columns"]:
            values[name] = _names(values.get(name, []), name)
        try:
            return cls(**values).validate()
        except TypeError as exc:
            raise DatasetError("Не заданы обязательные параметры эксперимента.") from exc

    @classmethod
    def from_file(cls, path: Path):
        try:
            return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise DatasetError("Не удалось разобрать JSON-конфигурацию.") from exc
