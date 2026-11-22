"""Искусственные стресс-сценарии на validation без изменения модели."""
from dataclasses import replace
import numpy as np
from .analysis_context import save_analysis, validation_context
from .baselines import scores
from .data import DatasetError
from .preprocessing import feature_matrix, target_vector


def changed_dataset(dataset, rows):
    return replace(dataset,rows=tuple(tuple(row) for row in rows),source="derived-validation-stress",sha256="")


def robustness_analysis(dataset, config, model, repeats=3, seed=42):
    config.check_columns(dataset)
    if type(repeats) is not int or not 1 <= repeats <= 10 or type(seed) is not int or not 0 <= seed < 2**32:
        raise DatasetError("Недопустимое число повторов или seed анализа устойчивости.")
    n = len(dataset.rows)
    features = list(config.features)
    variants = 6 * repeats + len(features)
    if n < 2 or n * len(features) * variants > 2_000_000:
        raise DatasetError("Превышен бюджет анализа устойчивости или недостаточно строк.")
    truth = target_vector(dataset,config)
    if set(truth)-set(model.classes):
        raise DatasetError("Validation содержит неизвестный модели класс.")
    raw = feature_matrix(dataset,config)
    baseline_proba = model.predict_proba(dataset)
    labels = np.asarray(model.classes)
    baseline_prediction = labels[baseline_proba.argmax(axis=1)]
    baseline = scores("classification",truth,baseline_prediction,labels=model.classes)
    random = np.random.default_rng(seed)
    columns = [dataset.columns.index(feature) for feature in features]
    results = []
    def measure(rows, scenario, level, repeat, changed):
        altered = changed_dataset(dataset,rows)
        proba = model.predict_proba(altered)
        predicted = labels[proba.argmax(axis=1)]
        measured = scores("classification",truth,predicted,labels=model.classes)
        results.append({"scenario":scenario,"level":level,"repeat":repeat,"changed_cells":changed,"metrics":measured,
            "macro_f1_drop":baseline["macro_f1"]-measured["macro_f1"],"prediction_flip_rate":float(np.mean(predicted!=baseline_prediction)),
            "mean_probability_l1":float(np.abs(proba-baseline_proba).sum(axis=1).mean())})
    for fraction in [.05,.15,.30]:
        for repeat in range(repeats):
            rows = [list(row) for row in dataset.rows]
            mask = random.random((n,len(columns))) < fraction
            changed = 0
            for row_index, feature_index in zip(*np.nonzero(mask)):
                column = columns[int(feature_index)]
                if rows[int(row_index)][column] != "":
                    rows[int(row_index)][column] = ""
                    changed += 1
            measure(rows,"random_missing_fraction",fraction,repeat,changed)
    if config.numeric_features:
        for multiplier in [.05,.15,.30]:
            for repeat in range(repeats):
                rows = [list(row) for row in dataset.rows]
                changed = 0
                for feature_index,feature in enumerate(config.numeric_features):
                    column = dataset.columns.index(feature)
                    noise = random.normal(0,multiplier*model.scale[feature_index],n)
                    for row_index in range(n):
                        value = float(raw[row_index,feature_index])
                        if np.isnan(value):
                            continue
                        updated = value+float(noise[row_index])
                        if not np.isfinite(updated):
                            raise DatasetError("Возмущение создало неконечное значение.")
                        rows[row_index][column] = repr(updated)
                        changed += 1
                measure(rows,"numeric_noise_train_scale",multiplier,repeat,changed)
    for feature,column in zip(features,columns):
        rows = [list(row) for row in dataset.rows]
        changed = 0
        for row in rows:
            if row[column] != "":
                changed += 1
                row[column] = ""
        measure(rows,"complete_feature_missing",feature,0,changed)
    summaries = []
    keys = list(dict.fromkeys((row["scenario"],row["level"]) for row in results))
    for scenario,level in keys:
        subset = [row for row in results if row["scenario"]==scenario and row["level"]==level]
        drops = [row["macro_f1_drop"] for row in subset]
        summaries.append({"scenario":scenario,"level":level,"repeats":len(subset),"mean_macro_f1_drop":float(np.mean(drops)),
            "std_macro_f1_drop":float(np.std(drops,ddof=1)) if len(drops)>1 else 0.0,"mean_prediction_flip_rate":float(np.mean([row["prediction_flip_rate"] for row in subset]))})
    return {"analysis":"robustness","baseline":baseline,"seed":seed,"repeats":repeats,"scenarios":results,"summary":summaries,
        "limitation":"Искусственные пропуски и шум используют сохранённый train-масштаб, без переобучения. Шум может выходить за предметные диапазоны; это стресс-тест модели, не прогноз реальных изменений сотрудников. Разброс повторов не является доверительным интервалом."}


def run_robustness(split_directory, experiment_directory, destination, repeats=3, seed=42):
    dataset,config,model,provenance = validation_context(split_directory,experiment_directory)
    return save_analysis(destination,{**provenance,**robustness_analysis(dataset,config,model,repeats,seed)})
