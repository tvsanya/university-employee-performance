"""Групповой bootstrap фиксированной модели на validation."""
import numpy as np
from .analysis_context import save_analysis, validation_context
from .baselines import scores
from .data import DatasetError
from .preprocessing import target_vector


def grouped_bootstrap(dataset, config, model, repeats=500, confidence=.95, seed=42):
    config.check_columns(dataset)
    if type(repeats) is not int or not 50 <= repeats <= 2000:
        raise DatasetError("Bootstrap требует 50–2000 повторов.")
    if type(confidence) not in (float,int) or not .8 <= confidence <= .99:
        raise DatasetError("Уровень интервала должен быть от 0.8 до 0.99.")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise DatasetError("Недопустимый seed bootstrap.")
    n = len(dataset.rows)
    if n < 2 or n*repeats > 2_000_000:
        raise DatasetError("Недостаточно строк или превышен бюджет bootstrap.")
    truth = target_vector(dataset,config)
    predicted = model.predict(dataset)
    if set(truth)-set(model.classes):
        raise DatasetError("Неизвестный модели класс в validation.")
    groups = {}
    if config.group_column:
        column = dataset.columns.index(config.group_column)
        for row_index,row in enumerate(dataset.rows):
            key = row[column]
            if not key:
                raise DatasetError("Групповой ID не может быть пустым.")
            groups.setdefault(key,[]).append(row_index)
        unit = "group"
    else:
        groups = {str(index):[index] for index in range(n)}
        unit = "row"
    members = [np.asarray(indices,dtype=int) for indices in groups.values()]
    group_count = len(members)
    if group_count < 2:
        raise DatasetError("Для bootstrap нужны минимум две независимые группы.")
    baseline = scores("classification",truth,predicted,labels=model.classes)
    random = np.random.default_rng(seed)
    samples = []
    incomplete_classes = 0
    evaluated_rows = 0
    for repeat in range(repeats):
        chosen = random.integers(0,group_count,size=group_count)
        indices = np.concatenate([members[int(index)] for index in chosen])
        evaluated_rows += len(indices)
        if evaluated_rows > 2_000_000:
            raise DatasetError("Bootstrap превысил бюджет строк из-за неодинаковых размеров групп.")
        sample_truth = truth[indices]
        if set(model.classes)-set(sample_truth):
            incomplete_classes += 1
        measured = scores("classification",sample_truth,predicted[indices],labels=model.classes)
        samples.append({"rows":len(indices),**measured})
    alpha = (1-confidence)/2
    intervals = {}
    for metric,point in baseline.items():
        values = np.asarray([sample[metric] for sample in samples],dtype=float)
        intervals[metric] = {"point":point,"lower":float(np.quantile(values,alpha)),"upper":float(np.quantile(values,1-alpha)),
            "bootstrap_mean":float(values.mean()),"bootstrap_std":float(values.std(ddof=1))}
    return {"analysis":"grouped_bootstrap","resampling_unit":unit,"groups":group_count,"rows":n,"repeats":repeats,"seed":seed,"confidence":confidence,
        "interval_method":"percentile","metrics":intervals,"samples_missing_classes":incomplete_classes,"samples":samples,
        "limitation":"Интервалы условны для сохранённой модели и validation при предположении независимости групп. Они не включают неопределённость обучения, не являются оценкой причинного эффекта и не подтверждают перенос с симуляции на реальных сотрудников. При отсутствии group_column предполагается независимость строк."}


def run_uncertainty(split_directory,experiment_directory,destination,repeats=500,seed=42):
    dataset,config,model,provenance = validation_context(split_directory,experiment_directory)
    report = grouped_bootstrap(dataset,config,model,repeats=repeats,seed=seed)
    return save_analysis(destination,{**provenance,**report})
