"""Командный интерфейс первого этапа подготовки данных."""
import argparse
import json
from pathlib import Path
from urllib.error import URLError
from .data import DatasetError, load_dataset
from .quality import profile_dataset, validate_dataset
from .splitting import split_dataset, write_splits
from .experiment import ExperimentConfig


def main(argv=None):
    parser = argparse.ArgumentParser(description="Проверка данных для оценки эффективности сотрудников университета")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Посмотреть структуру CSV")
    validate = commands.add_parser("validate", help="Проверить объём, признаки и целевую колонку")
    split = commands.add_parser("split", help="Сохранить независимые train/validation/test")
    split.add_argument("--output", required=True, type=Path, help="Новый каталог для выборок и манифеста")
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--group-column", help="Идентификатор сотрудника или иной независимой группы")
    split.add_argument("--train-fraction", type=float, default=0.7)
    split.add_argument("--validation-fraction", type=float, default=0.15)
    check_spec = commands.add_parser("check-spec", help="Проверить описание задачи и признаки")
    check_spec.add_argument("--config", required=True, type=Path)
    for command in [inspect, validate, split, check_spec]:
        command.add_argument("source", help="HTTP/HTTPS-ссылка или локальный CSV")
        command.add_argument("--member", help="Имя CSV внутри ZIP-архива")
        command.add_argument("--delimiter", help="Явный разделитель вместо автоопределения")
        command.add_argument("--report", type=Path, help="Сохранить сводку JSON")
        if command is not check_spec:
            command.add_argument("--target", required=command is not inspect, help="Название целевой колонки")
    args = parser.parse_args(argv)
    try:
        dataset = load_dataset(args.source, args.member, args.delimiter)
        if args.command == "check-spec":
            config = ExperimentConfig.from_file(args.config)
            config.check_columns(dataset)
            report = validate_dataset(dataset, config.target)
            report["experiment"] = config.as_dict()
        else:
            report = profile_dataset(dataset, args.target) if args.command == "inspect" else validate_dataset(dataset, args.target)
        if args.command == "split" and report["ok"]:
            result = split_dataset(dataset, seed=args.seed, train_fraction=args.train_fraction,
                                   validation_fraction=args.validation_fraction, group_column=args.group_column)
            report["split"] = write_splits(dataset, result, args.output, target=args.target)
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(serialized, encoding="utf-8")
        print(serialized, end="")
        return 0 if report.get("ok", True) else 2
    except (DatasetError, OSError, URLError) as exc:
        parser.exit(2, f"Ошибка данных: {exc}\n")
