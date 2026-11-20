"""Синтетический сценарий для разработки, не данные университета."""
import csv
import hashlib
import json
import random
import shutil
import tempfile
from pathlib import Path
from .data import DatasetError
from .experiment import ExperimentConfig

NUMERIC_FEATURES = ("employment_fraction", "planned_teaching_hours", "teaching_hours",
                    "student_feedback_avg", "course_completion_rate", "publication_count",
                    "training_hours", "admin_delay_days", "digital_activity_rate")
COLUMNS = ("source_kind", "employee_id", "period_id", "department", "position",
           *NUMERIC_FEATURES, "efficiency_score", "efficiency_level")


def clip(value, low, high):
    return min(high, max(low, value))


def demo_rows(employees=1500, periods=2, seed=42):
    """Признаки и цель — шумные следствия искусственного скрытого фактора."""
    if not 3 <= employees <= 5000 or not 1 <= periods <= 5:
        raise DatasetError("Для демо допустимы 3–5000 сотрудников и 1–5 периодов.")
    generator = random.Random(seed)
    departments = ["Информатика", "Экономика", "Математика", "Психология"]
    for employee in range(employees):
        ability = generator.gauss(0, 1)
        department = generator.choice(departments)
        position = generator.choice(["Преподаватель", "Старший преподаватель", "Доцент"])
        fraction = generator.choice([0.5, 0.75, 1.0])
        for period in range(periods):
            latent = ability + generator.gauss(0, 0.25)
            planned = round(generator.uniform(120, 240)*fraction)
            score = clip(70+latent*15+generator.gauss(0, 4), 0, 100)
            label = "низкая" if score < 55 else "средняя" if score < 80 else "высокая"
            features = [f"{fraction:.2f}", str(planned),
                str(round(planned*clip(0.95+latent*0.06+generator.gauss(0, 0.07), 0.4, 1.3))),
                f"{clip(3.6+latent*0.55+generator.gauss(0, 0.35), 1, 5):.2f}",
                f"{clip(0.75+latent*0.09+generator.gauss(0, 0.06), 0, 1):.3f}",
                str(max(0, round(2+latent+generator.gauss(0, 1.2)))),
                str(round(clip(30+latent*5+generator.gauss(0, 8), 0, 150))),
                f"{clip(7-latent*2+generator.gauss(0, 2), 0, 30):.1f}",
                f"{clip(0.7+latent*0.1+generator.gauss(0, 0.08), 0, 1):.3f}"]
            # Пропуски есть только в признаках; метки сохраняются.
            for index in range(3, len(features)):
                if generator.random() < 0.02:
                    features[index] = ""
            yield ("synthetic", f"SIM-T{employee+1:05d}", f"SIM-P{period+1}", department, position,
                   *features, f"{score:.2f}", label)


def generate_demo(destination: Path, *, employees=1500, periods=2, seed=42):
    destination = Path(destination)
    if destination.exists():
        raise DatasetError("Каталог демонстрационных данных уже существует.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".simulation-", dir=destination.parent))
    try:
        path = temporary/"synthetic_employees.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(COLUMNS)
            writer.writerows(demo_rows(employees, periods, seed))
        config = ExperimentConfig("classification", "efficiency_level",
            "Синтетическая категория скрытого фактора симулятора; не реальная оценка сотрудника",
            NUMERIC_FEATURES, ("department", "position"), "employee_id",
            ("efficiency_score", "source_kind", "period_id"), seed).validate()
        provenance = {"data_origin": "synthetic", "intended_use": "development_demo_only",
            "generator": "university-demo-v1", "seed": seed, "employees": employees,
            "periods": periods, "rows": employees*periods, "columns": len(COLUMNS),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "limitations": "Искусственные зависимости проверяют код; метрики не доказывают качество оценки реальных сотрудников."}
        (temporary/"provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        (temporary/"experiment.json").write_text(json.dumps(config.as_dict(), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
        temporary.rename(destination)
        return provenance
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
