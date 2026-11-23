"""Бакулин Александр Максимович.

Автоматизация процессов оценки эффективности работы сотрудника университета
с применением интеллектуального анализа данных.
"""
import sys


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"--check", "--port"}:
        from run_local import main as local_main
        return local_main(arguments)
    from university_analytics.cli import main as command_main
    return command_main(arguments)

if __name__ == "__main__":
    raise SystemExit(main())
