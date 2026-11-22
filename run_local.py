"""Local GUI entry point, also suitable for the Run action in a Python IDE."""
import argparse
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PORT = 8033


def main():
    parser = argparse.ArgumentParser(description="Локальный интерфейс анализа данных")
    parser.add_argument("--check", action="store_true", help="Проверить запуск интерфейса без сервера")
    parser.add_argument("--port", type=int, default=PORT, help="Локальный порт приложения")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Порт должен быть от 1024 до 65535")
    os.chdir(BASE_DIR)
    from university_analytics.gui import create_app
    app = create_app(BASE_DIR)
    manager = app.extensions["training_manager"]
    try:
        if args.check:
            with app.test_client() as client:
                response = client.get("/")
                if response.status_code != 200:
                    raise SystemExit("Интерфейс не прошёл проверку запуска")
            print("Локальный интерфейс проверен: http://127.0.0.1:" + str(args.port))
            return
        app.run(host="127.0.0.1", port=args.port, debug=False, use_reloader=False)
    finally:
        manager.close()


if __name__ == "__main__":
    main()
