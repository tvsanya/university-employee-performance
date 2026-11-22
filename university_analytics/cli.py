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
    baseline = commands.add_parser("train-baseline", help="Обучить baseline только на train и сохранить модели")
    baseline.add_argument("--splits", required=True, type=Path)
    baseline.add_argument("--config", required=True, type=Path)
    baseline.add_argument("--output", required=True, type=Path)
    baseline.add_argument("--report", type=Path)
    neural = commands.add_parser("train-neural", help="Подобрать DL-сеть по validation и сохранить лучший checkpoint")
    neural.add_argument("--splits", required=True, type=Path)
    neural.add_argument("--config", required=True, type=Path)
    neural.add_argument("--grid", required=True, type=Path)
    neural.add_argument("--output", required=True, type=Path)
    neural.add_argument("--report", type=Path)
    predict = commands.add_parser("predict", help="Применить сохранённую JSON-сеть без повторного обучения")
    predict.add_argument("source", help="CSV с признаками, целевая колонка необязательна")
    predict.add_argument("--model", required=True, type=Path)
    predict.add_argument("--output", required=True, type=Path, help="Новый каталог для predictions.csv и report.json")
    predict.add_argument("--report", type=Path)
    diagnose = commands.add_parser("diagnose", help="Разобрать ошибки сохранённой JSON-сети без обучения")
    diagnose.add_argument("--splits", required=True, type=Path)
    diagnose.add_argument("--experiment", required=True, type=Path)
    diagnose.add_argument("--output", required=True, type=Path)
    diagnose.add_argument("--report", type=Path)
    compare = commands.add_parser("compare", help="Сравнить семейства моделей по validation")
    compare.add_argument("--experiments", nargs="+", required=True, type=Path)
    compare.add_argument("--output", required=True, type=Path)
    compare.add_argument("--report", type=Path)
    plot = commands.add_parser("plot", help="Экспортировать историю и матрицу ошибок в PNG/SVG")
    plot.add_argument("--experiment", required=True, type=Path)
    plot.add_argument("--diagnostics", required=True, type=Path)
    plot.add_argument("--output", required=True, type=Path)
    plot.add_argument("--report", type=Path)
    importance = commands.add_parser("importance", help="Оценить важность признаков на validation без обучения")
    importance.add_argument("--splits", required=True, type=Path)
    importance.add_argument("--experiment", required=True, type=Path)
    importance.add_argument("--output", required=True, type=Path)
    importance.add_argument("--repeats", type=int, default=5)
    importance.add_argument("--seed", type=int, default=42)
    importance.add_argument("--report", type=Path)
    serve = commands.add_parser("serve", help="Открыть локальный интерфейс данных и JSON-прогноза")
    serve.add_argument("--port", type=int, default=8033)
    demo = commands.add_parser("generate-demo", help="Создать явно помеченную симуляцию для разработки")
    demo.add_argument("--output", required=True, type=Path)
    demo.add_argument("--employees", type=int, default=1500)
    demo.add_argument("--periods", type=int, default=2)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--report", type=Path)
    for command in [inspect, validate, split, check_spec]:
        command.add_argument("source", help="HTTP/HTTPS-ссылка или локальный CSV")
        command.add_argument("--member", help="Имя CSV внутри ZIP-архива")
        command.add_argument("--delimiter", help="Явный разделитель вместо автоопределения")
        command.add_argument("--report", type=Path, help="Сохранить сводку JSON")
        if command is not check_spec:
            command.add_argument("--target", required=command is not inspect, help="Название целевой колонки")
    args = parser.parse_args(argv)
    try:
        if args.command == "serve":
            if not 1024 <= args.port <= 65535:
                parser.error("Порт должен быть от 1024 до 65535.")
            from .gui import create_app
            create_app().run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
            return 0
        elif args.command == "generate-demo":
            from .simulation import generate_demo
            report = generate_demo(args.output, employees=args.employees, periods=args.periods, seed=args.seed)
        elif args.command == "importance":
            from .importance import run_importance
            report = run_importance(args.splits, args.experiment, args.output, args.repeats, args.seed)
        elif args.command == "predict":
            from .prediction import run_prediction
            report = run_prediction(args.model, args.source, args.output)
        elif args.command == "diagnose":
            from .diagnostics import run_diagnostics
            report = run_diagnostics(args.splits, args.experiment, args.output)
        elif args.command == "compare":
            from .comparison import run_comparison
            report = run_comparison(args.experiments, args.output)
        elif args.command == "plot":
            from .plots import export_plots
            report = export_plots(args.experiment, args.diagnostics, args.output)
        elif args.command == "train-neural":
            from .tuning import load_network_grid, run_neural_experiment
            config = ExperimentConfig.from_file(args.config)
            report = run_neural_experiment(args.splits, config, load_network_grid(args.grid, config.seed), args.output)
        elif args.command == "train-baseline":
            # Зависимости обучения не требуются для обычного inspect.
            from .baselines import run_baseline_experiment
            report = run_baseline_experiment(args.splits, ExperimentConfig.from_file(args.config), args.output)
        else:
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
