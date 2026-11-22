"""Оценка вероятностей и уверенности без подбора на test."""
import numpy as np
from .analysis_context import save_analysis, validation_context
from .data import DatasetError
from .preprocessing import target_vector


def validate_probabilities(truth, probabilities, classes):
    if not isinstance(classes, (list, tuple)) or len(classes) < 2 or any(not isinstance(value,str) for value in classes) or len(set(classes)) != len(classes):
        raise DatasetError("Нужны уникальные классы вероятностной модели.")
    truth = np.asarray(truth)
    try:
        probabilities = np.asarray(probabilities, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DatasetError("Вероятности должны быть числами.") from exc
    if truth.ndim != 1 or not len(truth) or probabilities.shape != (len(truth),len(classes)):
        raise DatasetError("Размеры целей и вероятностей не согласованы.")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
        raise DatasetError("Вероятности должны быть конечными и лежать от 0 до 1.")
    if not np.allclose(probabilities.sum(axis=1),1,rtol=0,atol=1e-8):
        raise DatasetError("Сумма вероятностей каждой строки должна быть единицей.")
    mapping = {label:index for index,label in enumerate(classes)}
    if set(truth)-set(mapping):
        raise DatasetError("Цель содержит неизвестный модели класс.")
    indices = np.asarray([mapping[label] for label in truth],dtype=int)
    return truth, probabilities, indices


def probability_quality(truth, probabilities, classes, bins=10, thresholds=(0,.5,.7,.9,.95)):
    truth, probabilities, indices = validate_probabilities(truth,probabilities,classes)
    if type(bins) is not int or not 2 <= bins <= 50:
        raise DatasetError("Число интервалов должно быть от 2 до 50.")
    if not isinstance(thresholds,(list,tuple)) or not 1 <= len(thresholds) <= 20:
        raise DatasetError("Нужно от 1 до 20 порогов уверенности.")
    for threshold in thresholds:
        if type(threshold) not in (float,int) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise DatasetError("Пороги должны лежать от 0 до 1.")
    n = len(truth)
    predicted_indices = probabilities.argmax(axis=1)
    correct = predicted_indices == indices
    confidence = probabilities.max(axis=1)
    true_probability = probabilities[np.arange(n),indices]
    epsilon = 1e-15
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(n),indices] = 1
    squared = (probabilities-one_hot)**2
    assignments = np.minimum((confidence*bins).astype(int),bins-1)
    reliability = []
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        count = int(mask.sum())
        mean_confidence = float(confidence[mask].mean()) if count else None
        accuracy = float(correct[mask].mean()) if count else None
        gap = abs(mean_confidence-accuracy) if count else None
        if count:
            ece += count/n*gap
        reliability.append({"lower":index/bins,"upper":(index+1)/bins,"rows":count,"mean_confidence":mean_confidence,"accuracy":accuracy,"absolute_gap":gap})
    selective = []
    for threshold in sorted(set(thresholds)):
        mask = confidence >= threshold
        count = int(mask.sum())
        selective.append({"threshold":threshold,"rows":count,"coverage":count/n,"accuracy":float(correct[mask].mean()) if count else None})
    entropy = -np.sum(probabilities*np.log(np.clip(probabilities,epsilon,1)),axis=1)
    classwise = []
    for index,label in enumerate(classes):
        classwise.append({"label":label,"support":int((indices==index).sum()),"mean_predicted_probability":float(probabilities[:,index].mean()),"observed_frequency":float((indices==index).mean()),"binary_brier":float(squared[:,index].mean())})
    return {"analysis":"probability_quality","rows":n,"classes":list(classes),
        "log_loss":float(-np.log(np.clip(true_probability,epsilon,1)).mean()),"log_loss_epsilon":epsilon,"true_probabilities_below_epsilon":int((true_probability<epsilon).sum()),
        "multiclass_brier_sum":float(squared.sum(axis=1).mean()),"top_label_ece":float(ece),"maximum_bin_gap":max((row["absolute_gap"] for row in reliability if row["rows"]),default=0),
        "mean_confidence":float(confidence.mean()),"accuracy":float(correct.mean()),"mean_normalized_entropy":float(entropy.mean()/np.log(len(classes))),
        "reliability_bins":reliability,"classwise":classwise,"selective":selective,
        "limitation":"Это описательная оценка вероятностей на validation, а не калибровка модели. Brier и log-loss учитывают также различимость классов; ECE зависит от интервалов. Пороги не подбираются и не являются правилом кадрового решения."}


def run_probability_analysis(split_directory, experiment_directory, destination):
    dataset, config, model, provenance = validation_context(split_directory, experiment_directory)
    report = probability_quality(target_vector(dataset,config),model.predict_proba(dataset),model.classes)
    return save_analysis(destination,{**provenance,**report})
