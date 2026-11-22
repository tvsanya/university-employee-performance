"""Описательные расстояния распределений признаков train и validation."""
from collections import Counter
import numpy as np
from .analysis_context import save_analysis, validation_context
from .baselines import load_split_bundle
from .data import DatasetError
from .preprocessing import feature_matrix


def stable_moments(values):
    scale = float(np.max(np.abs(values))) or 1.0
    normalized = values/scale
    mean = float(normalized.mean())*scale
    deviation = float(normalized.std())*scale
    return mean,deviation,scale


def numeric_distance(reference, current, bins):
    reference = np.asarray(reference,dtype=float)
    current = np.asarray(current,dtype=float)
    reference_missing = np.isnan(reference)
    current_missing = np.isnan(current)
    left = reference[~reference_missing]
    right = current[~current_missing]
    result = {"kind":"numeric","reference_missing_rate":float(reference_missing.mean()),"current_missing_rate":float(current_missing.mean()),
        "reference_nonmissing":len(left),"current_nonmissing":len(right)}
    result["missing_rate_change"] = result["current_missing_rate"]-result["reference_missing_rate"]
    if not len(left) or not len(right):
        result.update(status="insufficient_nonmissing",ks_distance=None,psi=None)
        return result
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise DatasetError("Неконечный признак в анализе распределений.")
    left_sorted = np.sort(left)
    right_sorted = np.sort(right)
    points = np.unique(np.concatenate([left_sorted,right_sorted]))
    reference_cdf = np.searchsorted(left_sorted,points,side="right")/len(left)
    current_cdf = np.searchsorted(right_sorted,points,side="right")/len(right)
    result["ks_distance"] = float(np.max(np.abs(reference_cdf-current_cdf)))
    reference_mean,standard_deviation,scale = stable_moments(left)
    current_mean,_,_ = stable_moments(right)
    result["reference_mean"] = reference_mean
    result["current_mean"] = current_mean
    result["mean_change"] = result["current_mean"]-result["reference_mean"]
    if not np.isfinite(result["mean_change"]):
        raise DatasetError("Диапазон признака слишком велик для конечной разницы средних.")
    result["reference_std"] = standard_deviation
    result["standardized_mean_change"] = result["mean_change"]/standard_deviation if standard_deviation>0 else None
    result["outside_reference_range_rate"] = float(((right<left.min())|(right>left.max())).mean())
    cutpoints = np.unique(np.quantile(left/scale,np.linspace(0,1,bins+1)[1:-1])*scale)
    edges = np.concatenate([[-np.inf],cutpoints,[np.inf]])
    reference_counts = np.histogram(left,bins=edges)[0]
    current_counts = np.histogram(right,bins=edges)[0]
    smoothing = .5
    reference_frequency = (reference_counts+smoothing)/(len(left)+smoothing*len(reference_counts))
    current_frequency = (current_counts+smoothing)/(len(right)+smoothing*len(current_counts))
    contributions = (current_frequency-reference_frequency)*np.log(current_frequency/reference_frequency)
    result["psi"] = float(contributions.sum())
    result["psi_smoothing_count"] = smoothing
    histogram = []
    for index in range(len(reference_counts)):
        lower = float(edges[index]) if np.isfinite(edges[index]) else None
        upper = float(edges[index+1]) if np.isfinite(edges[index+1]) else None
        histogram.append({"lower":lower,"upper":upper,"reference_rows":int(reference_counts[index]),"current_rows":int(current_counts[index]),
            "reference_smoothed_frequency":float(reference_frequency[index]),"current_smoothed_frequency":float(current_frequency[index]),"psi_contribution":float(contributions[index])})
    result["histogram"] = histogram
    result["status"] = "completed"
    return result


def categorical_distance(reference,current):
    left = Counter(value if isinstance(value,str) else None for value in reference)
    right = Counter(value if isinstance(value,str) else None for value in current)
    n_left = sum(left.values())
    n_right = sum(right.values())
    if not n_left or not n_right:
        raise DatasetError("Нельзя сравнить пустые категориальные распределения.")
    categories = set(left)|set(right)
    contributions = []
    for value in categories:
        before = left[value]/n_left
        after = right[value]/n_right
        contributions.append({"value":value,"reference_rows":left[value],"current_rows":right[value],"reference_frequency":before,"current_frequency":after,"absolute_change":abs(after-before)})
    contributions.sort(key=lambda item:(-item["absolute_change"],str(item["value"])))
    unknown = sum(count for value,count in right.items() if value is not None and value not in left)
    distance = .5*sum(item["absolute_change"] for item in contributions)
    top = contributions[:20]
    return {"kind":"categorical","status":"completed","total_variation":distance,"unseen_category_rows":unknown,"unseen_category_rate":unknown/n_right,
        "reference_missing_rate":left[None]/n_left,"current_missing_rate":right[None]/n_right,"categories":len(categories),"largest_changes":top,"omitted_categories":max(0,len(contributions)-len(top))}


def distribution_shift(reference,current,config,bins=10):
    config.check_columns(reference)
    config.check_columns(current)
    if type(bins) is not int or not 2 <= bins <= 30:
        raise DatasetError("Для распределений допустимо 2–30 интервалов.")
    if not reference.rows or not current.rows:
        raise DatasetError("Нужны непустые train и validation.")
    if (len(reference.rows)+len(current.rows))*len(config.features)>2_000_000:
        raise DatasetError("Превышен ресурсный бюджет сравнения распределений.")
    left = feature_matrix(reference,config)
    right = feature_matrix(current,config)
    features = []
    numeric_count = len(config.numeric_features)
    for index,name in enumerate(config.features):
        if index < numeric_count:
            measured = numeric_distance(left[:,index],right[:,index],bins)
        else:
            measured = categorical_distance(left[:,index],right[:,index])
        features.append({"feature":name,**measured})
    return {"analysis":"input_distribution_shift","reference_partition":"train","current_partition":"validation","reference_rows":len(reference.rows),"current_rows":len(current.rows),
        "binning":"train_quantiles","requested_bins":bins,"features":features,
        "limitation":"Сравниваются только входные распределения. KS приводится как расстояние без p-value; PSI использует train-границы и сглаживание 0.5 наблюдения. Числа не доказывают деградацию модели, справедливость или причинность; пороги тревоги не подбирались. Test не используется."}


def run_distribution_shift(split_directory,experiment_directory,destination):
    current,config,model,provenance = validation_context(split_directory,experiment_directory)
    datasets,manifest = load_split_bundle(split_directory,config)
    report = distribution_shift(datasets["train"],current,config)
    return save_analysis(destination,{**provenance,**report})
