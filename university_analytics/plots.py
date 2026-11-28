"""Экспорт проверенных историй и матриц ошибок в самостоятельные PNG/SVG."""
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from .data import DatasetError


def read_plot_inputs(experiment, diagnostics):
    root = Path(experiment)
    try:
        metrics = json.loads((root/"metrics.json").read_text(encoding="utf-8"))
        diagnostic = json.loads((Path(diagnostics)/"diagnostics.json").read_text(encoding="utf-8"))
        artifact = metrics.get("training_artifact", {})
        name = artifact.get("file")
        if not isinstance(name, str) or Path(name).name != name:
            raise DatasetError("Не найден корректный файл истории обучения.")
        payload = (root/name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != artifact.get("sha256"):
            raise DatasetError("Контрольная сумма истории не совпадает.")
        training = json.loads(payload)
        if diagnostic.get("model_sha256") != metrics.get("portable_artifact", {}).get("sha256") or diagnostic.get("data_origin") != metrics.get("data_origin"):
            raise DatasetError("Диагностика и история относятся к разным моделям.")
        selected = training.get("selected_candidate")
        matches = [item for item in training.get("trials", []) if item.get("candidate") == selected]
        if len(matches) != 1:
            raise DatasetError("Не найдена выбранная конфигурация сети.")
        trial = matches[0]
        history = trial.get("history", [])
        if not history or len(history) > 1000:
            raise DatasetError("Некорректная длина истории.")
        for row in history:
            for value in [row.get("loss"), row.get("train", {}).get("macro_f1"), row.get("validation", {}).get("macro_f1")]:
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise DatasetError("История содержит нечисловое значение.")
        return metrics, diagnostic, trial
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        raise DatasetError("Не удалось разобрать историю и диагностику.") from exc


def export_plots(experiment, diagnostics, destination):
    destination = Path(destination)
    if destination.exists():
        raise DatasetError("Каталог графиков уже существует.")
    metrics, diagnostic, trial = read_plot_inputs(experiment, diagnostics)
    os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent/".local/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    import numpy as np
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".figures-", dir=destination.parent))
    try:
        history = trial["history"]
        epochs = [row["epoch"] for row in history]
        figure = Figure(figsize=(11, 4.3), layout="constrained")
        axes = figure.subplots(1, 2)
        axes[0].plot(epochs, [row["loss"] for row in history], color="#254ca0")
        axes[0].set(xlabel="Epoch", ylabel="Train objective", title="Optimization history")
        for partition, color in [("train", "#254ca0"), ("validation", "#188567")]:
            axes[1].plot(epochs, [row[partition]["macro_f1"] for row in history], label=partition, color=color)
        axes[1].axvline(trial["best_epoch"], color="#a74032", linestyle="--", label="Selected checkpoint")
        axes[1].set(xlabel="Epoch", ylabel="Macro-F1", ylim=(0, 1), title="Selection uses validation")
        axes[1].legend(loc="lower right")
        for axis in axes:
            axis.grid(alpha=.2)
        figure.suptitle("MLP training")
        for suffix in ["png", "svg"]:
            figure.savefig(temporary/f"training.{suffix}", dpi=180, metadata={"Date": None} if suffix == "svg" else None)
        test = diagnostic["partitions"]["test"]
        labels = test["labels"]
        if len(labels) <= 12:
            matrix = np.asarray(test["confusion_matrix"])
            if matrix.shape != (len(labels), len(labels)) or not np.isfinite(matrix).all() or (matrix < 0).any():
                raise DatasetError("Некорректная матрица ошибок.")
            confusion = Figure(figsize=(6, 5.5), layout="constrained")
            axis = confusion.subplots()
            image = axis.imshow(matrix, cmap="Blues", vmin=0)
            axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
                xlabel="Predicted class", ylabel="True class", title="Held-out test")
            for i in range(len(labels)):
                for j in range(len(labels)):
                    axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="white" if matrix[i, j] > matrix.max()/2 else "#142635")
            confusion.colorbar(image, ax=axis, label="Rows")
            for suffix in ["png", "svg"]:
                confusion.savefig(temporary/f"test-confusion.{suffix}", dpi=180, metadata={"Date": None} if suffix == "svg" else None)
        report = {"data_origin": metrics["data_origin"], "intended_use": metrics["intended_use"],
            "best_epoch": trial["best_epoch"], "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(temporary.iterdir())}}
        (temporary/"figures.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        temporary.rename(destination)
        return report
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
