"""Проверенный контекст анализа: только validation и безопасный JSON."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from .baselines import load_split_bundle
from .data import DatasetError
from .experiment import ExperimentConfig
from .portable import PortableNetwork
from .preprocessing import target_vector


def validation_context(split_directory, experiment_directory):
    root = Path(experiment_directory)
    raw = (root / "metrics.json").read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise DatasetError("Отчёт эксперимента слишком велик.")
    try:
        metrics = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise DatasetError("Некорректный JSON отчёта эксперимента.") from exc
    if not isinstance(metrics, dict) or not isinstance(metrics.get("portable_artifact"), dict):
        raise DatasetError("Нужен эксперимент с переносимой JSON-моделью.")
    artifact = metrics["portable_artifact"]
    filename = artifact.get("file")
    if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".json"):
        raise DatasetError("Недопустимое имя модели.")
    model = PortableNetwork.from_file(root / filename)
    if model.sha256 != artifact.get("sha256"):
        raise DatasetError("Контрольная сумма модели не совпала.")
    config = ExperimentConfig.from_file(root / "config.json")
    if config != model.config:
        raise DatasetError("Контракт модели не совпадает с конфигурацией.")
    datasets, manifest = load_split_bundle(split_directory, config)
    split_sha = hashlib.sha256((Path(split_directory)/"manifest.json").read_bytes()).hexdigest()
    if metrics.get("source_sha256") != manifest.get("source_sha256") or metrics.get("split_manifest_sha256") != split_sha:
        raise DatasetError("Данные и эксперимент имеют разные контрольные суммы.")
    if metrics.get("data_origin") != model.data_origin:
        raise DatasetError("Происхождение модели не совпадает с отчётом.")
    dataset = datasets["validation"]
    truth = target_vector(dataset, config)
    if not len(truth) or set(truth) - set(model.classes):
        raise DatasetError("Validation пуста или содержит класс, неизвестный модели.")
    provenance = {"partition":"validation", "rows":len(dataset.rows), "source_sha256":manifest["source_sha256"], "split_manifest_sha256":split_sha, "model_sha256":model.sha256, "data_origin":model.data_origin, "intended_use":model.intended_use}
    return dataset, config, model, provenance


def save_analysis(destination, report):
    destination = Path(destination)
    if destination.exists():
        raise DatasetError("Каталог анализа уже существует.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".analysis-", dir=destination.parent))
    try:
        payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
        (temporary/"analysis.json").write_text(payload, encoding="utf-8")
        temporary.rename(destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return report
