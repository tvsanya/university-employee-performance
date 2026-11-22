"""Перестановочная зависимость качества модели от исходных признаков."""
from dataclasses import replace
import numpy as np
from .analysis_context import save_analysis, validation_context
from .baselines import scores
from .data import DatasetError
from .preprocessing import target_vector


def permutation_importance(dataset, config, model, repeats=5, seed=42):
    config.check_columns(dataset)
    if type(repeats) is not int or not 1 <= repeats <= 20:
        raise DatasetError("Число перестановок должно быть от 1 до 20.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise DatasetError("Seed должен быть целым от 0 до 2^32−1.")
    n = len(dataset.rows)
    if n < 2 or n * len(config.features) * repeats > 2_000_000:
        raise DatasetError("Недостаточно строк или превышен бюджет анализа 2 млн строк-признаков.")
    truth = target_vector(dataset, config)
    if set(truth) - set(model.classes):
        raise DatasetError("В validation есть неизвестные модели классы.")
    baseline = scores("classification", truth, model.predict(dataset), labels=model.classes)
    random = np.random.default_rng(seed)
    results = []
    for feature in config.features:
        column = dataset.columns.index(feature)
        values = [row[column] for row in dataset.rows]
        drops = []
        for repeat in range(repeats):
            order = random.permutation(n)
            changed = []
            for row_number, row in enumerate(dataset.rows):
                updated = list(row)
                updated[column] = values[int(order[row_number])]
                changed.append(tuple(updated))
            altered = replace(dataset, rows=tuple(changed), source="derived-validation-permutation", sha256="")
            measured = scores("classification", truth, model.predict(altered), labels=model.classes)
            drops.append({metric:baseline[metric]-measured[metric] for metric in baseline})
        f1 = [drop["macro_f1"] for drop in drops]
        results.append({"feature":feature, "macro_f1_drop_mean":float(np.mean(f1)), "macro_f1_drop_std":float(np.std(f1, ddof=1)) if repeats > 1 else 0.0,
            "macro_f1_drop_min":float(min(f1)), "macro_f1_drop_max":float(max(f1)), "accuracy_drop_mean":float(np.mean([drop["accuracy"] for drop in drops])), "repeats":drops})
    results.sort(key=lambda item:(-item["macro_f1_drop_mean"],item["feature"]))
    return {"analysis":"permutation_importance", "baseline":baseline, "seed":seed, "repeats":repeats, "features":results,
        "limitation":"Перестановка оценивает зависимость данной модели от признака на validation. Коррелированные признаки и зависимые строки влияют на результат; это не причинный эффект и не доверительный интервал."}


def run_importance(split_directory, experiment_directory, destination, repeats=5, seed=42):
    dataset, config, model, provenance = validation_context(split_directory, experiment_directory)
    report = permutation_importance(dataset, config, model, repeats, seed)
    return save_analysis(destination, {**provenance, **report})
