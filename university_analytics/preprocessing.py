"""Подготовка признаков; статистики оцениваются только по train."""
import math
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import Dataset, DatasetError
from .experiment import ExperimentConfig
from .quality import is_missing


def feature_matrix(dataset: Dataset, config: ExperimentConfig):
    config.validate()
    missing = set(config.features) - set(dataset.columns)
    if missing:
        raise DatasetError("Отсутствуют входные признаки: " + ", ".join(sorted(missing)))
    matrix = np.empty((len(dataset.rows), len(config.features)), dtype=object)
    indices = [dataset.columns.index(name) for name in config.features]
    for number, row in enumerate(dataset.rows):
        for position, (name, index) in enumerate(zip(config.features, indices)):
            value = row[index]
            if is_missing(value) or value.casefold() == "nan":
                matrix[number, position] = np.nan
            elif name in config.numeric_features:
                try:
                    numeric = float(value)
                except ValueError as exc:
                    raise DatasetError(f"Колонка {name!r}, строка {number+2}: ожидалось число.") from exc
                if not math.isfinite(numeric):
                    raise DatasetError(f"Колонка {name!r} содержит недопустимое числовое значение.")
                matrix[number, position] = numeric
            else:
                matrix[number, position] = value
    return matrix


def target_vector(dataset: Dataset, config: ExperimentConfig):
    if config.target not in dataset.columns:
        raise DatasetError("Целевая колонка отсутствует.")
    index = dataset.columns.index(config.target)
    values = [row[index] for row in dataset.rows]
    if any(is_missing(value) or value.casefold() == "nan" for value in values):
        raise DatasetError("В целевом показателе есть пропуски.")
    if config.task == "regression":
        try:
            values = [float(value) for value in values]
        except ValueError as exc:
            raise DatasetError("Регрессия требует числового целевого показателя.") from exc
        if not all(math.isfinite(value) for value in values):
            raise DatasetError("Целевой показатель содержит бесконечность или NaN.")
    return np.asarray(values)


def build_preprocessor(config: ExperimentConfig):
    config.validate()
    transformers = []
    count = len(config.numeric_features)
    if count:
        numeric = Pipeline([
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ])
        transformers.append(("numeric", numeric, list(range(count))))
    if config.categorical_features:
        categorical = Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__", keep_empty_features=True)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical, list(range(count, len(config.features)))))
    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0)


def fit_preprocessor(train: Dataset, config: ExperimentConfig):
    if not train.rows:
        raise DatasetError("Обучающая выборка пуста.")
    config.check_columns(train)
    return build_preprocessor(config).fit(feature_matrix(train, config))


def transform_features(dataset: Dataset, config: ExperimentConfig, preprocessor):
    return preprocessor.transform(feature_matrix(dataset, config))
