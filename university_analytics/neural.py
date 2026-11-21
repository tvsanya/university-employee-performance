"""Глубокая полносвязная сеть с внешней групповой validation-выборкой."""
import copy
import math
from dataclasses import asdict, dataclass
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from .baselines import scores
from .data import DatasetError
from .preprocessing import build_preprocessor, feature_matrix, target_vector


@dataclass(frozen=True)
class NetworkConfig:
    hidden_layers: tuple[int, ...] = (32, 16)
    activation: str = "relu"
    learning_rate: float = 0.001
    alpha: float = 0.0001
    batch_size: int = 64
    max_epochs: int = 100
    patience: int = 12
    min_delta: float = 0.0001
    balanced: bool = True
    seed: int = 42

    def validate(self):
        if not isinstance(self.hidden_layers, (tuple, list)) or not 2 <= len(self.hidden_layers) <= 4:
            raise DatasetError("DL-сеть должна содержать от двух до четырёх скрытых слоёв.")
        if any(type(width) is not int or not 2 <= width <= 256 for width in self.hidden_layers):
            raise DatasetError("В скрытом слое допустимо от 2 до 256 нейронов.")
        for name, low, high in [("batch_size", 1, 1024), ("max_epochs", 1, 1000), ("patience", 1, 1000), ("seed", 0, 2**32-1)]:
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise DatasetError(f"{name}: требуется целое число от {low} до {high}.")
        for name, low, high in [("learning_rate", 0, 1), ("alpha", 0, 10), ("min_delta", 0, 1)]:
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise DatasetError(f"Недопустимое значение {name}.")
        if self.learning_rate == 0 or self.activation not in {"relu", "tanh"} or type(self.balanced) is not bool:
            raise DatasetError("Проверьте learning_rate, activation и balanced.")
        return self

    def as_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict) or set(values) - set(cls.__dataclass_fields__):
            raise DatasetError("Неизвестные параметры конфигурации сети.")
        try:
            return cls(**values).validate()
        except TypeError as exc:
            raise DatasetError("Неверная конфигурация сети.") from exc


def train_network(train, validation, config, parameters=None):
    """Test не принимается: эпоха выбирается только по внешней validation."""
    parameters = (parameters or NetworkConfig(seed=config.seed)).validate()
    config.validate()
    if config.task != "classification":
        raise DatasetError("Текущая DL-архитектура предназначена для классификации; для регрессии доступны baseline.")
    for data in [train, validation]:
        config.check_columns(data)
        if not data.rows:
            raise DatasetError("Train и validation не должны быть пустыми.")
    train_y, validation_y = target_vector(train, config), target_vector(validation, config)
    classes = np.unique(train_y)
    if len(classes) < 2:
        raise DatasetError("Для обучения сети нужны минимум два класса в train.")
    if set(validation_y) - set(classes):
        raise DatasetError("В validation есть классы, отсутствующие в train.")
    preprocessor = build_preprocessor(config)
    train_x = preprocessor.fit_transform(feature_matrix(train, config))
    validation_x = preprocessor.transform(feature_matrix(validation, config))
    classifier = MLPClassifier(hidden_layer_sizes=tuple(parameters.hidden_layers), activation=parameters.activation,
        solver="adam", learning_rate_init=parameters.learning_rate, alpha=parameters.alpha,
        batch_size=min(parameters.batch_size, len(train_y)), random_state=parameters.seed,
        early_stopping=False, shuffle=True)
    weights = compute_sample_weight("balanced", train_y) if parameters.balanced else None
    history, best_model = [], None
    best_score, patience_reference, best_epoch, stale = -math.inf, -math.inf, 0, 0
    for epoch in range(1, parameters.max_epochs+1):
        try:
            classifier.partial_fit(train_x, train_y, classes=classes, sample_weight=weights)
        except ValueError as exc:
            raise DatasetError(f"Не удалось обучить сеть: {exc}") from exc
        if not math.isfinite(classifier.loss_) or not all(np.isfinite(layer).all() for layer in classifier.coefs_ + classifier.intercepts_):
            raise DatasetError("Обучение разошлось: нечисловые веса или loss.")
        train_score = scores("classification", train_y, classifier.predict(train_x))
        validation_score = scores("classification", validation_y, classifier.predict(validation_x))
        score = validation_score["macro_f1"]
        history.append({"epoch": epoch, "loss": float(classifier.loss_), "train": train_score, "validation": validation_score})
        if score > best_score:
            best_score, best_epoch, best_model = score, epoch, copy.deepcopy(classifier)
        if score > patience_reference + parameters.min_delta:
            patience_reference, stale = score, 0
        else:
            stale += 1
        if stale >= parameters.patience:
            break
    model = Pipeline([("preprocess", preprocessor), ("model", best_model)])
    architecture = {"input_units": int(train_x.shape[1]), "hidden_layers": list(parameters.hidden_layers),
        "output_units": int(best_model.n_outputs_), "output_activation": best_model.out_activation_,
        "class_labels": best_model.classes_.tolist(), "activation": parameters.activation,
        "parameter_count": int(sum(item.size for item in best_model.coefs_ + best_model.intercepts_))}
    return model, {"parameters": parameters.as_dict(), "architecture": architecture, "history": history,
        "best_epoch": best_epoch, "best_validation_macro_f1": best_score, "epochs_run": len(history),
        "stop_reason": "patience" if stale >= parameters.patience else "max_epochs",
        "selection_partition": "validation", "selection_metric": "macro_f1"}
