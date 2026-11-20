"""Командный интерфейс первого этапа подготовки данных."""
import argparse
import json
from pathlib import Path
from urllib.error import URLError
from .data import DatasetError, basic_profile, load_dataset
from .quality import profile_dataset, validate_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Проверка данных для оценки эффективности сотрудников университета")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Посмотреть структуру CSV")
    validate = commands.add_parser("validate", help="Проверить объём, признаки и целевую колонку")
    for command in [inspect, validate]:
        command.add_argument("source", help="HTTP/HTTPS-ссылка или локальный CSV")
        command.add_argument("--member", help="Имя CSV внутри ZIP-архива")
        command.add_argument("--delimiter", help="Явный разделитель вместо автоопределения")
        command.add_argument("--report", type=Path, help="Сохранить сводку JSON")
        command.add_argument("--target", required=command is validate, help="Название целевой колонки")
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.source, args.member, args.delimiter)
        report = validate_dataset(dataset, args.target) if args.command == "validate" else profile_dataset(dataset, args.target)
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if report.get("ok", True) else 2
    except (DatasetError, OSError, URLError) as exc:
        parser.exit(2, f"Ошибка данных: {exc}\n")
