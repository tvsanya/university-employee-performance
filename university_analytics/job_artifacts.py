"""Только разрешённые результаты своего законченного эксперимента."""
import hashlib
import io
import json
import math
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from .jobs import JobNotFound


class JobNotReady(RuntimeError):
    pass


ARTIFACT_LABELS = {"preflight.json": "Проверка настроек JSON","research.md": "Исследовательский отчёт Markdown", "research-manifest.json": "Происхождение и SHA анализов JSON", "model.json": "Выбранная модель JSON", "comparison.json": "Сравнение JSON",
    "comparison.csv": "Сравнение CSV", "history.json": "История нейросети JSON",
    "diagnostics.json": "Ошибки выбранной модели JSON", "training.png": "Обучение нейросети PNG",
    "neural-confusion.png": "Матрица ошибок нейросети PNG", "distribution.json": "Распределения признаков (train/validation)", "importance.json": "Важность признаков (validation)", "probabilities.json": "Вероятности (validation)", "robustness.json": "Устойчивость (validation)", "uncertainty.json": "Групповые интервалы (validation)", "experiment.zip": "Полный эксперимент с данными ZIP"}


MAX_ARTIFACT=32*1024*1024
MAX_ARCHIVE=128*1024*1024


def checked_file(directory,path):
    chain=[path]
    while chain[-1]!=directory and directory in chain[-1].parents:chain.append(chain[-1].parent)
    if directory.resolve() not in path.resolve().parents or any(part.is_symlink() for part in chain):
        raise JobNotReady('Файл результата является ссылкой или выходит за каталог эксперимента.')
    if not path.is_file():raise JobNotReady('Файл результата отсутствует. Исходный эксперимент сохранён.')
    with path.open('rb') as stream:payload=stream.read(MAX_ARTIFACT+1)
    if len(payload)>MAX_ARTIFACT:raise JobNotReady('Файл результата превышает допустимый размер.')
    return payload


def artifact_json(directory,path):
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError
            result[key]=value
        return result
    def invalid(value):raise ValueError
    try:
        value=json.loads(checked_file(directory,path),object_pairs_hook=pairs,parse_constant=invalid)
        stack=[(value,0)]
        while stack:
            item,depth=stack.pop()
            if depth>64 or (isinstance(item,float) and not math.isfinite(item)):raise ValueError
            if isinstance(item,dict):stack.extend((v,depth+1) for v in item.values())
            elif isinstance(item,list):stack.extend((v,depth+1) for v in item)
        if not isinstance(value,dict):raise ValueError
        return value
    except (ValueError,UnicodeError,RecursionError):
        raise JobNotReady('Повреждён JSON результата. Файлы сохранены; создайте новый эксперимент.') from None


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
    mapping = {"preflight.json": "preflight.json","research.md": "research/research.md", "research-manifest.json": "research/manifest.json", "model.json": f"{family}/model.json", "comparison.json": "comparison/comparison.json",
        "comparison.csv": "comparison/comparison.csv", "history.json": "neural/training.json",
        "diagnostics.json": "diagnostics-selected/diagnostics.json" if family == "baseline" else "diagnostics-neural/diagnostics.json",
        "distribution.json": "analysis/distribution/analysis.json", "importance.json": "analysis/importance/analysis.json", "probabilities.json": "analysis/probabilities/analysis.json", "robustness.json": "analysis/robustness/analysis.json", "uncertainty.json": "analysis/uncertainty/analysis.json",
        "training.png": "figures/training.png", "neural-confusion.png": "figures/test-confusion.png"}
    if name not in mapping:
        raise JobNotFound
    path = directory/mapping[name]
    if not path.is_file() or directory.resolve() not in path.resolve().parents:
        raise JobNotFound
    checked_file(directory,path)
    return path


def experiment_archive(manager, job_id, owner):
    directory, state = completed_job(manager, job_id, owner)
    files = [directory/"input.csv", directory/"config.json", directory/"preflight.json"]
    for folder in ["splits", "baseline", "neural", "comparison", "diagnostics-neural", "diagnostics-selected", "figures", "analysis", "research"]:
        files += sorted((directory/folder).rglob("*"))
    for required in [directory/'input.csv',directory/'config.json',directory/'comparison/comparison.json']:
        checked_file(directory,required)
    if len(files)>512:raise JobNotReady('Слишком много файлов для архива.')
    output, hashes = io.BytesIO(), {}
    total=0
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for path in files:
            if not path.is_file() or directory.resolve() not in path.resolve().parents:
                continue
            name = path.relative_to(directory).as_posix()
            payload = checked_file(directory,path)
            total+=len(payload)
            if total>MAX_ARCHIVE:raise JobNotReady('Архив превышает допустимый объём.')
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
