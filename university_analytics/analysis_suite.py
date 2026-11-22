"""Ограниченный набор анализов результатов GUI-эксперимента."""
from pathlib import Path
from .analysis_context import save_analysis, validation_context
from .data import DatasetError
from .importance import permutation_importance
from .preprocessing import target_vector
from .probabilities import probability_quality
from .robustness import robustness_analysis
from .uncertainty import grouped_bootstrap


def run_analysis_suite(split_directory,experiment_directory,destination):
    dataset,config,model,provenance = validation_context(split_directory,experiment_directory)
    truth = target_vector(dataset,config)
    tasks = {
        "importance":lambda:permutation_importance(dataset,config,model,repeats=3,seed=config.seed),
        "probabilities":lambda:probability_quality(truth,model.predict_proba(dataset),model.classes),
        "robustness":lambda:robustness_analysis(dataset,config,model,repeats=2,seed=config.seed),
        "uncertainty":lambda:grouped_bootstrap(dataset,config,model,repeats=200,seed=config.seed),
    }
    status = {}
    for name,calculate in tasks.items():
        try:
            report = {"status":"completed",**calculate()}
        except DatasetError as exc:
            report = {"status":"unavailable","analysis":name,"reason":str(exc)}
        save_analysis(Path(destination)/name,{**provenance,**report})
        status[name] = report["status"]
    return status
