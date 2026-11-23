"""Локальный интерфейс проверки CSV и применения безопасной JSON-модели."""
import csv
import hashlib
import hmac
import io
import json
import re
import secrets
import os
import tempfile
from itertools import islice
from pathlib import Path
from flask import Flask, abort, jsonify, render_template, request, session, send_file
from werkzeug.exceptions import HTTPException
from .data import DatasetError, MAX_BYTES, load_dataset, read_source
from urllib.parse import urlparse
from .prediction import run_prediction
from .quality import profile_dataset
from .jobs import TrainingManager, TrainingBusy, JobNotFound
from .job_artifacts import ARTIFACT_LABELS, JobNotReady, artifact_path, experiment_archive, completed_job


def local_secret(repository):
    directory = repository/".local"
    directory.mkdir(exist_ok=True)
    secret_path = directory/"gui-secret.key"
    if not secret_path.exists():
        with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=".gui-secret-", delete=False) as stream:
            stream.write(secrets.token_hex(32))
            temporary = Path(stream.name)
        try:
            os.link(temporary, secret_path)
        except FileExistsError:
            pass
        finally:
            temporary.unlink(missing_ok=True)
    secret = secret_path.read_text(encoding="utf-8").strip()
    if len(secret) < 64:
        raise DatasetError("Повреждён локальный ключ браузерной сессии.")
    return secret


def create_app(repository=None):
    repository = Path(repository or Path(__file__).resolve().parent.parent)
    models = repository/"models"
    web = Path(__file__).parent/"web"
    app = Flask(__name__, template_folder=str(web/"templates"), static_folder=str(web/"static"))
    app.config.update(SECRET_KEY=local_secret(repository), MAX_CONTENT_LENGTH=38*1024*1024,
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

    @app.errorhandler(JobNotReady)
    def job_not_ready(error):
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
        address = request.form.get("source_url", "").strip()
        has_file = uploaded is not None and bool(uploaded.filename)
        if has_file == bool(address):
            raise DatasetError("Выберите один источник: CSV-файл либо HTTP/HTTPS-ссылку.")
        if address:
            try:
                parsed = urlparse(address)
                valid = (len(address) <= 2048 and parsed.scheme in {"http", "https"} and parsed.hostname
                    and not parsed.username and not parsed.password and not parsed.fragment
                    and not any(ord(char) < 33 for char in address))
                parsed.port
            except ValueError as exc:
                raise DatasetError("Некорректный адрес CSV.") from exc
            if not valid:
                raise DatasetError("Укажите HTTP/HTTPS-ссылку до2048 символов без учётных данных и фрагмента.")
            try:
                contents = read_source(address)
            except DatasetError:
                raise
            except (OSError, ValueError) as exc:
                raise DatasetError("Не удалось загрузить CSV по ссылке. Проверьте адрес и доступность источника.") from exc
        else:
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
                if name.startswith("job:"):
                    model_path = artifact_path(manager, name[4:], session.get("workspace_id", ""), "model.json")
                    experiment = model_path.parent
                else:
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
            try:
                review_threshold = float(request.form.get("review_threshold", ".7"))
            except ValueError as exc:
                raise DatasetError("Укажите числовой порог просмотра от 0 до 1.") from exc
            report = run_prediction(model_path, str(source), root/"result", max_rows=20000, max_classes=20, review_threshold=review_threshold)
            contents = (root/"result/predictions.csv").read_text(encoding="utf-8")
            preview = list(islice(csv.DictReader(io.StringIO(contents)), 30))
            return jsonify(report=report, preview=preview, csv=contents)

    @app.post("/api/train")
    @app.post("/api/train/validate")
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
            except (ValueError, UnicodeDecodeError, RecursionError) as exc:
                raise DatasetError("Не удалось прочитать JSON-контракт.") from exc
        with tempfile.TemporaryDirectory(prefix="university-train-") as folder:
            source = csv_upload(Path(folder))
            if request.path.endswith('/validate'):
                _, _, summary = manager.prepare(source, values, epochs)
                return jsonify(summary)
            job = manager.submit(source, values, session.setdefault("workspace_id", secrets.token_urlsafe(32)), epochs)
        return jsonify(job), 202

    @app.get("/api/jobs/<job_id>")
    def job_status(job_id):
        return jsonify(manager.get(job_id, session.get("workspace_id", "")))

    @app.get("/api/jobs")
    def job_list():
        return jsonify(jobs=manager.list(session.get("workspace_id", "")))

    @app.get("/api/jobs/<job_id>/artifacts")
    def job_artifacts(job_id):
        directory, state = completed_job(manager, job_id, session.get("workspace_id", ""))
        comparison = json.loads((directory/"comparison/comparison.json").read_text(encoding="utf-8"))
        return jsonify(selected=state["result"], comparison=comparison,
            files=[{"name": name, "label": label} for name, label in ARTIFACT_LABELS.items()
                if (not name.startswith("research") or state["result"].get("research_status"))
                and (name != "preflight.json" or (directory/name).is_file())])

    @app.get("/api/jobs/<job_id>/download/<name>")
    def job_download(job_id, name):
        owner = session.get("workspace_id", "")
        if name == "experiment.zip":
            return send_file(experiment_archive(manager, job_id, owner), as_attachment=True, download_name="experiment.zip", mimetype="application/zip")
        path = artifact_path(manager, job_id, owner, name)
        return send_file(path, as_attachment=True, download_name=name)

    @app.get("/api/jobs/<job_id>/figure/<name>")
    def job_figure(job_id, name):
        if name not in {"training.png", "neural-confusion.png"}:
            raise JobNotFound
        return send_file(artifact_path(manager, job_id, session.get("workspace_id", ""), name), mimetype="image/png")

    return app
