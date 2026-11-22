"""Только разрешённые результаты своего законченного эксперимента."""
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from .jobs import JobNotFound


class JobNotReady(RuntimeError):
    pass


ARTIFACT_LABELS = {"model.json": "Выбранная модель JSON", "comparison.json": "Сравнение JSON",
    "comparison.csv": "Сравнение CSV", "history.json": "История нейросети JSON",
    "diagnostics.json": "Ошибки выбранной модели JSON", "training.png": "Обучение нейросети PNG",
    "neural-confusion.png": "Матрица ошибок нейросети PNG", "distribution.json": "Распределения признаков (train/validation)", "importance.json": "Важность признаков (validation)", "probabilities.json": "Вероятности (validation)", "robustness.json": "Устойчивость (validation)", "uncertainty.json": "Групповые интервалы (validation)", "experiment.zip": "Полный эксперимент с данными ZIP"}


def completed_job(manager, job_id, owner):
    state = manager.get(job_id, owner)
    if state["state"] != "completed":
        raise JobNotReady("Результаты доступны после успешного завершения эксперимента.")
    return manager.directory(job_id, owner), state


def artifact_path(manager, job_id, owner, name):
    directory, state = completed_job(manager, job_id, owner)
    family = state["result"]["selected_family"]
    if family not in {"baseline", "neural"}:
        raise JobNotFound
    mapping = {"model.json": f"{family}/model.json", "comparison.json": "comparison/comparison.json",
        "comparison.csv": "comparison/comparison.csv", "history.json": "neural/training.json",
        "diagnostics.json": "diagnostics-selected/diagnostics.json" if family == "baseline" else "diagnostics-neural/diagnostics.json",
        "distribution.json": "analysis/distribution/analysis.json", "importance.json": "analysis/importance/analysis.json", "probabilities.json": "analysis/probabilities/analysis.json", "robustness.json": "analysis/robustness/analysis.json", "uncertainty.json": "analysis/uncertainty/analysis.json",
        "training.png": "figures/training.png", "neural-confusion.png": "figures/test-confusion.png"}
    if name not in mapping:
        raise JobNotFound
    path = directory/mapping[name]
    if not path.is_file() or directory.resolve() not in path.resolve().parents:
        raise JobNotFound
    return path


def experiment_archive(manager, job_id, owner):
    directory, state = completed_job(manager, job_id, owner)
    files = [directory/"input.csv", directory/"config.json"]
    for folder in ["splits", "baseline", "neural", "comparison", "diagnostics-neural", "diagnostics-selected", "figures", "analysis"]:
        files += sorted((directory/folder).rglob("*"))
    output, hashes = io.BytesIO(), {}
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            if not path.is_file() or directory.resolve() not in path.resolve().parents:
                continue
            name = path.relative_to(directory).as_posix()
            payload = path.read_bytes()
            hashes[name] = hashlib.sha256(payload).hexdigest()
            archive.writestr(name, payload)
        requirements = manager.repository/"requirements.txt"
        if requirements.is_file():
            payload = requirements.read_bytes()
            hashes["requirements.txt"] = hashlib.sha256(payload).hexdigest()
            archive.writestr("requirements.txt", payload)
        archive.writestr("SHA256SUMS.json", json.dumps(hashes, ensure_ascii=False, indent=2)+"\n")
        archive.writestr("README.txt", "Эксперимент содержит исходный CSV и независимые выборки.\n"
            "Происхождение: "+state["result"]["data_origin"]+". Назначение: "+state["result"]["intended_use"]+".\n"
            "Модель выбирается по validation. Для применения используйте JSON выбранного семейства.\n"
            "Версии зависимостей и параметры сохранены; инструкции программы: https://github.com/tvsanya/university-employee-performance\n"
            "Служебные идентификаторы браузерной сессии в архив не включены.\n")
    output.seek(0)
    return output
