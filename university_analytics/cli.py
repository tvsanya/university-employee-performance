"""Командный интерфейс первого этапа подготовки данных."""
import argparse
import json
from pathlib import Path
from urllib.error import URLError
from .data import DatasetError, basic_profile, load_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Проверка данных для оценки эффективности сотрудников университета")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Посмотреть структуру CSV")
    inspect.add_argument("source", help="HTTP/HTTPS-ссылка или локальный CSV")
    inspect.add_argument("--member", help="Имя CSV внутри ZIP-архива")
    inspect.add_argument("--report", type=Path, help="Сохранить сводку JSON")
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.source, args.member)
        report = basic_profile(dataset)
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0
    except (DatasetError, OSError, URLError) as exc:
        parser.exit(2, f"Ошибка данных: {exc}\n")
