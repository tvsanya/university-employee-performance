"""Применение MLP и классификационных baseline из проверенного JSON."""
import json
import hashlib
from pathlib import Path
import numpy as np
from .data import DatasetError
from .experiment import ExperimentConfig
from .preprocessing import feature_matrix

FORMAT = "university-mlp-json-v1"
LINEAR_FORMAT = "university-linear-json-v1"
DUMMY_FORMAT = "university-dummy-json-v1"
MAX_MODEL_BYTES = 16 * 1024 * 1024


def vector(values, size, name):
    if not isinstance(values, list) or len(values) != size or any(type(item) not in (int, float) for item in values):
        raise DatasetError(f"{name}: неверный числовой вектор.")
    try:
        result = np.asarray(values, dtype=float)
    except (ValueError, OverflowError) as exc:
        raise DatasetError(f"{name}: недопустимые числа.") from exc
    if not np.isfinite(result).all():
        raise DatasetError(f"{name}: NaN и бесконечность запрещены.")
    return result


def preprocessing_payload(model, config, data_origin):
    preprocess = model.named_steps["preprocess"]
    numeric = preprocess.named_transformers_.get("numeric")
    categorical = preprocess.named_transformers_.get("categorical")
    return {"config": config.as_dict(), "data_origin": data_origin,
        "intended_use": "development_demo_only" if data_origin in {"synthetic", "mixed"} else "research_requires_validation",
        "numeric": {"impute": numeric.named_steps["impute"].statistics_.tolist() if numeric else [],
            "mean": numeric.named_steps["scale"].mean_.tolist() if numeric else [],
            "scale": numeric.named_steps["scale"].scale_.tolist() if numeric else []},
        "categorical": {"categories": [item.tolist() for item in categorical.named_steps["encode"].categories_] if categorical else [],
            "missing_value": "__MISSING__"}}


def export_network(model, config, data_origin="unspecified"):
    classifier = model.named_steps["model"]
    payload = {**preprocessing_payload(model, config, data_origin), "format": FORMAT,
        "network": {"weights": [item.tolist() for item in classifier.coefs_],
            "biases": [item.tolist() for item in classifier.intercepts_], "activation": classifier.activation,
            "output_activation": classifier.out_activation_, "classes": classifier.classes_.tolist()}}
    PortableNetwork(payload)
    return payload


def export_baseline(model, config, data_origin="unspecified"):
    from sklearn.dummy import DummyClassifier
    from sklearn.linear_model import LogisticRegression
    if config.task != "classification":
        raise DatasetError("Перенос baseline в JSON сейчас поддерживает классификацию.")
    classifier = model.named_steps["model"]
    classes = classifier.classes_.tolist()
    payload = preprocessing_payload(model, config, data_origin)
    if isinstance(classifier, LogisticRegression):
        payload.update(format=LINEAR_FORMAT, network={"weights": [classifier.coef_.T.tolist()],
            "biases": [classifier.intercept_.tolist()], "classes": classes, "activation": "identity",
            "output_activation": "logistic" if len(classes) == 2 else "softmax"})
    elif isinstance(classifier, DummyClassifier) and classifier.strategy == "most_frequent":
        probabilities = [0.0]*len(classes)
        probabilities[int(np.argmax(classifier.class_prior_))] = 1.0
        payload.update(format=DUMMY_FORMAT, network={"classes": classes, "probabilities": probabilities})
    else:
        raise DatasetError("Неподдерживаемый тип baseline для JSON.")
    PortableNetwork(payload)
    return payload


class PortableNetwork:
    def __init__(self, payload):
        if not isinstance(payload, dict) or not isinstance(payload.get("format"), str) or payload["format"] not in {FORMAT, LINEAR_FORMAT, DUMMY_FORMAT}:
            raise DatasetError("Неподдерживаемый формат JSON-модели.")
        self.format = payload["format"]
        self.config = ExperimentConfig.from_dict(payload.get("config"))
        if self.config.task != "classification":
            raise DatasetError("JSON-модель поддерживает только классификацию.")
        self.data_origin = payload.get("data_origin")
        if not isinstance(self.data_origin, str) or self.data_origin not in {"synthetic", "mixed", "unspecified"}:
            raise DatasetError("Неизвестное происхождение обучающих данных.")
        expected_use = "development_demo_only" if self.data_origin in {"synthetic", "mixed"} else "research_requires_validation"
        if payload.get("intended_use") != expected_use:
            raise DatasetError("Назначение модели не соответствует происхождению данных.")
        self.intended_use = expected_use
        numeric, categorical, network = (payload.get(key) for key in ["numeric", "categorical", "network"])
        if not all(isinstance(item, dict) for item in [numeric, categorical, network]):
            raise DatasetError("В JSON отсутствуют параметры преобразования или сети.")
        count = len(self.config.numeric_features)
        self.impute = vector(numeric.get("impute"), count, "Медианы")
        self.mean = vector(numeric.get("mean"), count, "Средние")
        self.scale = vector(numeric.get("scale"), count, "Масштабы")
        if (self.scale <= 0).any():
            raise DatasetError("Масштаб должен быть положительным.")
        self.categories = categorical.get("categories")
        self.missing_value = categorical.get("missing_value")
        if not isinstance(self.categories, list) or len(self.categories) != len(self.config.categorical_features) or not isinstance(self.missing_value, str):
            raise DatasetError("Неверный словарь категорий.")
        for values in self.categories:
            if not isinstance(values, list) or not values or not all(isinstance(item, str) for item in values) or len(set(values)) != len(values):
                raise DatasetError("Категории должны быть непустыми уникальными строками.")
        width = count + sum(len(values) for values in self.categories)
        if not 1 <= width <= 10000:
            raise DatasetError("Недопустимое число входов сети.")
        self.classes = network.get("classes")
        if not isinstance(self.classes, list) or not 2 <= len(self.classes) <= 1000 or not all(isinstance(item, str) and item for item in self.classes) or len(set(self.classes)) != len(self.classes):
            raise DatasetError("Неверный список классов.")
        self.constant = None
        if self.format == DUMMY_FORMAT:
            self.constant = vector(network.get("probabilities"), len(self.classes), "Вероятности baseline")
            if (self.constant < 0).any() or (self.constant > 1).any() or not np.isclose(self.constant.sum(), 1, rtol=0, atol=1e-12):
                raise DatasetError("Некорректное распределение вероятностей baseline.")
            return
        self.activation, self.output_activation = network.get("activation"), network.get("output_activation")
        expected_output = "logistic" if len(self.classes) == 2 else "softmax"
        allowed = {"identity"} if self.format == LINEAR_FORMAT else {"relu", "tanh"}
        if not isinstance(self.activation, str) or self.activation not in allowed or self.output_activation != expected_output:
            raise DatasetError("Неподдерживаемая функция активации.")
        weights, biases = network.get("weights"), network.get("biases")
        valid_layers = isinstance(weights, list) and (len(weights) == 1 if self.format == LINEAR_FORMAT else 3 <= len(weights) <= 5)
        if not valid_layers or not isinstance(biases, list) or len(weights) != len(biases):
            raise DatasetError("Проверьте число скрытых слоёв и смещения.")
        self.weights, self.biases = [], []
        for index, (matrix, bias) in enumerate(zip(weights, biases)):
            outputs = 1 if expected_output == "logistic" else len(self.classes)
            if index < len(weights)-1:
                if not isinstance(bias, list) or not 2 <= len(bias) <= 256:
                    raise DatasetError("Скрытый слой должен содержать 2–256 нейронов.")
                outputs = len(bias)
            if not isinstance(matrix, list) or len(matrix) != width:
                raise DatasetError("Размеры матрицы весов не согласованы.")
            self.weights.append(np.vstack([vector(row, outputs, "Веса") for row in matrix]))
            self.biases.append(vector(bias, outputs, "Смещения"))
            width = outputs

    @classmethod
    def from_file(cls, path):
        with Path(path).open("rb") as stream:
            contents = stream.read(MAX_MODEL_BYTES+1)
        if len(contents) > MAX_MODEL_BYTES:
            raise DatasetError("JSON-модель превышает 16 МиБ.")
        try:
            payload = json.loads(contents)
        except (ValueError, UnicodeDecodeError, RecursionError) as exc:
            raise DatasetError("Не удалось прочитать JSON-модель.") from exc
        model = cls(payload)
        model.sha256 = hashlib.sha256(contents).hexdigest()
        return model

    def predict_proba(self, dataset):
        raw = feature_matrix(dataset, self.config)
        if self.constant is not None:
            return np.tile(self.constant, (len(dataset.rows), 1))
        numeric_count = len(self.config.numeric_features)
        numeric = raw[:, :numeric_count].astype(float)
        numeric = np.where(np.isnan(numeric), self.impute, numeric)
        with np.errstate(over="ignore", invalid="ignore"):
            chunks = [(numeric-self.mean)/self.scale]
            for offset, values in enumerate(self.categories, numeric_count):
                column = raw[:, offset]
                column = [self.missing_value if not isinstance(value, str) else value for value in column]
                chunks.append(np.asarray([[float(value == category) for category in values] for value in column]).reshape(len(column), len(values)))
            hidden = np.hstack(chunks)
            for index, (weights, bias) in enumerate(zip(self.weights, self.biases)):
                hidden = hidden @ weights + bias
                if not np.isfinite(hidden).all():
                    raise DatasetError("Слишком большие значения признаков: прогноз не является конечным числом.")
                if index < len(self.weights)-1:
                    hidden = np.maximum(hidden, 0) if self.activation == "relu" else np.tanh(hidden)
            if self.output_activation == "logistic":
                positive = 0.5*(1+np.tanh(hidden/2))
                return np.hstack([1-positive, positive])
            hidden = np.exp(hidden-hidden.max(axis=1, keepdims=True))
            return hidden/hidden.sum(axis=1, keepdims=True)

    def predict(self, dataset):
        return np.asarray(self.classes)[self.predict_proba(dataset).argmax(axis=1)]
