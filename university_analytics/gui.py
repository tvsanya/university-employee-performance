"""Локальный интерфейс проверки CSV и применения безопасной JSON-модели."""
import csv
import hashlib
import hmac
import io
import json
import re
import secrets
import tempfile
from itertools import islice
from pathlib import Path
from flask import Flask, abort, jsonify, render_template, request, session, send_file
from werkzeug.exceptions import HTTPException
from .data import DatasetError, MAX_BYTES, load_dataset
from .prediction import run_prediction
from .quality import profile_dataset
from .jobs import TrainingManager, TrainingBusy, JobNotFound


def create_app(repository=None):
    repository = Path(repository or Path(__file__).resolve().parent.parent)
    models = repository/"models"
    web = Path(__file__).parent/"web"
    app = Flask(__name__, template_folder=str(web/"templates"), static_folder=str(web/"static"))
    app.config.update(SECRET_KEY=secrets.token_hex(32), MAX_CONTENT_LENGTH=38*1024*1024,
        MAX_FORM_MEMORY_SIZE=512*1024, MAX_FORM_PARTS=10, TRUSTED_HOSTS=["127.0.0.1", "localhost"],
        SESSION_COOKIE_NAME="university_analytics_session", SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict")
    app.json.sort_keys = False
    manager = TrainingManager(repository)
    app.extensions["training_manager"] = manager

    @app.before_request
    def protect_local_session():
        if request.method == "POST":
            expected = session.get("csrf", "")
            supplied = request.headers.get("X-CSRF-Token", "")
            if not expected or not hmac.compare_digest(expected, supplied):
                abort(403, description="Обновите страницу перед отправкой данных.")
            if request.headers.get("Origin") and request.headers["Origin"] != request.host_url.rstrip("/"):
                abort(403, description="Источник запроса не совпадает с приложением.")

    @app.after_request
    def response_headers(response):
        response.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; object-src 'none'; frame-ancestors 'none'",
            "Cache-Control": "no-store"})
        return response

    @app.errorhandler(DatasetError)
    def dataset_error(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(OSError)
    def file_error(error):
        return jsonify(error="Не удалось прочитать файл. Проверьте данные и повторите загрузку."), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(TrainingBusy)
    def busy(error):
        return jsonify(error=str(error)), 409

    @app.errorhandler(JobNotFound)
    def missing_job(error):
        return jsonify(error="Эксперимент не найден в этой браузерной сессии."), 404

    @app.get("/")
    def index():
        session.setdefault("csrf", secrets.token_urlsafe(32))
        session.setdefault("workspace_id", secrets.token_urlsafe(32))
        return render_template("index.html", csrf=session["csrf"])

    @app.get("/api/experiments")
    def experiments():
        result = []
        if models.is_dir():
            for root in sorted(models.iterdir()):
                if not root.is_dir() or root.resolve().parent != models.resolve():
                    continue
                try:
                    report = json.loads((root/"metrics.json").read_text(encoding="utf-8"))
                    if not isinstance(report, dict) or not isinstance(report.get("metrics"), dict):
                        continue
                    result.append({"id": root.name, "data_origin": report.get("data_origin", "unspecified"),
                        "has_portable": (root/"model.json").is_file(), "metrics": report["metrics"]})
                except (OSError, ValueError):
                    continue
        return jsonify(experiments=result)

    @app.get("/api/figures/<name>")
    def figure(name):
        if name not in {"training.png", "test-confusion.png"}:
            abort(404)
        path = repository/"reports/simulation-figures-v1"/name
        if not path.is_file():
            abort(404)
        return send_file(path, mimetype="image/png")

    def csv_upload(folder):
        uploaded = request.files.get("data")
        if uploaded is None:
            raise DatasetError("Выберите CSV с данными.")
        contents = uploaded.read(MAX_BYTES+1)
        if not contents or len(contents) > MAX_BYTES:
            raise DatasetError("CSV должен быть непустым и не превышать 20 МиБ.")
        path = folder/"input.csv"
        path.write_bytes(contents)
        return path

    @app.post("/api/inspect")
    def inspect():
        with tempfile.TemporaryDirectory(prefix="university-inspect-") as folder:
            dataset = load_dataset(str(csv_upload(Path(folder))))
            report = profile_dataset(dataset)
            report.pop("source", None)
            return jsonify(report)

    @app.post("/api/predict")
    def predict():
        with tempfile.TemporaryDirectory(prefix="university-predict-") as folder:
            root = Path(folder)
            source = csv_upload(root)
            upload = request.files.get("model")
            if upload is not None and upload.filename:
                model_path = root/"model.json"
                contents = upload.read(16*1024*1024+1)
                if len(contents) > 16*1024*1024:
                    raise DatasetError("Модель превышает 16 МиБ.")
                model_path.write_bytes(contents)
            else:
                name = request.form.get("experiment", "")
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,100}", name):
                    raise DatasetError("Выберите сохранённую модель или загрузите JSON-файл.")
                experiment = models/name
                if experiment.resolve().parent != models.resolve():
                    abort(400)
                model_path = experiment/"model.json"
                try:
                    metadata = json.loads((experiment/"metrics.json").read_text(encoding="utf-8"))
                except ValueError as exc:
                    raise DatasetError("Повреждены сведения о модели.") from exc
                if hashlib.sha256(model_path.read_bytes()).hexdigest() != metadata.get("portable_artifact", {}).get("sha256"):
                    raise DatasetError("Нарушена целостность сохранённой модели.")
            report = run_prediction(model_path, str(source), root/"result", max_rows=20000, max_classes=20)
            contents = (root/"result/predictions.csv").read_text(encoding="utf-8")
            preview = list(islice(csv.DictReader(io.StringIO(contents)), 30))
            return jsonify(report=report, preview=preview, csv=contents)

    @app.post("/api/train")
    def train():
        try:
            epochs = int(request.form.get("epochs", "60"))
        except ValueError as exc:
            raise DatasetError("Количество эпох должно быть целым числом.") from exc
        config_file = request.files.get("config")
        values = None
        if config_file is not None and config_file.filename:
            payload = config_file.read(65537)
            if len(payload) > 65536:
                raise DatasetError("Контракт превышает 64 КиБ.")
            try:
                values = json.loads(payload)
            except (ValueError, UnicodeDecodeError) as exc:
                raise DatasetError("Не удалось прочитать JSON-контракт.") from exc
        with tempfile.TemporaryDirectory(prefix="university-train-") as folder:
            source = csv_upload(Path(folder))
            job = manager.submit(source, values, session.setdefault("workspace_id", secrets.token_urlsafe(32)), epochs)
        return jsonify(job), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        return jsonify(manager.get(job_id, session.get("workspace_id", "")))

    return app
