"""Единый воспроизводимый пакет диагностик сохранённой модели."""
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import tempfile
from .analysis_context import validation_context
from .data import DatasetError
from .drift import run_distribution_shift
from .importance import run_importance
from .probabilities import run_probability_analysis
from .robustness import run_robustness
from .uncertainty import run_uncertainty


def markdown_text(value):
    return escape(str(value)).replace("\\", "\\\\").replace("|", "\\|").replace("`", "\\`").replace("[", "\\[").replace("]", "\\]").replace("\n", " ").replace("\r", " ")


def run_research_report(splits, experiment, destination):
    destination = Path(destination)
    if destination.exists():
        raise DatasetError("Каталог исследовательского пакета уже существует.")
    dataset, config, model, provenance = validation_context(splits, experiment)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".research-", dir=destination.parent))
    try:
        analyses = {}
        steps = [("distribution",run_distribution_shift),("importance",run_importance),
            ("probabilities",run_probability_analysis),("robustness",run_robustness),("uncertainty",run_uncertainty)]
        for name, function in steps:
            try:
                report = function(splits, experiment, temporary/name)
            except DatasetError as exc:
                analyses[name] = {"status":"unavailable", "reason":str(exc)}
                continue
            if any(report.get(key) != value for key,value in provenance.items()):
                raise DatasetError("Анализы относятся к разным данным или моделям; пакет не сохранён.")
            analyses[name] = {"status":"completed", "file":f"{name}/analysis.json", "report":report}
        lines = ["# Диагностика сохранённой модели", "",
            f"Происхождение: {markdown_text(provenance['data_origin'])}. Назначение: {markdown_text(provenance['intended_use'])}.",
            "Симуляционные результаты не оценивают реальных сотрудников. Test не используется для настройки анализов.",
            f"Validation: {len(dataset.rows)} строк. Модель SHA-256: {provenance['model_sha256']}.",
            f"Источник SHA-256: {provenance['source_sha256']}. Split SHA-256: {provenance['split_manifest_sha256']}.", ""]
        for name, item in analyses.items():
            lines += [f"## {name}", ""]
            if item["status"] != "completed":
                lines += ["Недоступно: "+markdown_text(item["reason"]), ""]
                continue
            report = item["report"]
            if name == "importance":
                lines += ["Признак | Среднее падение macro-F1 | SD повторов", "--- | ---: | ---:"]
                lines += [f"{markdown_text(row['feature'])} | {row['macro_f1_drop_mean']:.5f} | {row['macro_f1_drop_std']:.5f}" for row in report["features"]]
            elif name == "probabilities":
                lines += [f"Log-loss: {report['log_loss']:.5f}; Brier (сумма по классам): {report['multiclass_brier_sum']:.5f}; ECE: {report['top_label_ece']:.5f}."]
            elif name == "uncertainty":
                lines += [f"Единица: {report['resampling_unit']}; групп: {report['groups']}; повторов: {report['repeats']}."]
                for metric, interval in report["metrics"].items():
                    lines += [f"{markdown_text(metric)}: {interval['point']:.5f}; percentile {report['confidence']:.0%}: [{interval['lower']:.5f}, {interval['upper']:.5f}]."]
            elif name == "robustness":
                lines += ["Сценарий | Уровень | Среднее падение macro-F1", "--- | --- | ---:"]
                lines += [f"{markdown_text(row['scenario'])} | {markdown_text(row['level'])} | {row['mean_macro_f1_drop']:.5f}" for row in report["summary"]]
            else:
                lines += [f"Сравниваются train ({report['reference_rows']} строк) и validation ({report['current_rows']} строк). Числовые и категориальные расстояния сохранены в JSON."]
            lines += ["", markdown_text(report["limitation"]), "", f"Полный результат: [{name}/analysis.json]({name}/analysis.json).", ""]
        (temporary/"research.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
        summary = {**provenance, "schema_version":1,
            "status":"completed" if all(item["status"]=="completed" for item in analyses.values()) else "partial",
            "analyses":{name:{key:value for key,value in item.items() if key!="report"} for name,item in analyses.items()}}
        summary["files"] = {str(path.relative_to(temporary)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(temporary.rglob("*")) if path.is_file()}
        (temporary/"manifest.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+"\n", encoding="utf-8")
        temporary.rename(destination)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
