from .job_lock import TrainingLease
"""Один ограниченный фоновый эксперимент с результатами браузерной сессии."""
import hashlib
import hmac
import json
import os
import shutil
import threading
import uuid
import traceback
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from datetime import datetime, timezone
from .analysis_suite import run_analysis_suite
from .research_report import run_research_report
from .baselines import run_baseline_experiment
from .comparison import run_comparison
from .data import Dataset, DatasetError, load_dataset
from .diagnostics import run_diagnostics
from .experiment import ExperimentConfig
from .neural import NetworkConfig
from .plots import export_plots
from .preprocessing import feature_matrix, target_vector
from .quality import validate_dataset
from .splitting import split_dataset, write_splits
from .tuning import run_neural_experiment


class TrainingBusy(RuntimeError):
    pass


class TrainingCancelled(RuntimeError):
    pass


class JobNotFound(LookupError):
    pass


def write_json(path, value):
    path=Path(path);temporary=None
    try:
        with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,prefix='.'+path.name+'-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n')
        temporary.replace(path)
    finally:
        if temporary is not None:temporary.unlink(missing_ok=True)


def write_state(directory, state):
    write_json(directory/'status.json',state)


def run_job(directory, dataset, config, epochs):
    state = json.loads((directory/"status.json").read_text(encoding="utf-8"))
    def check_cancel():
        if (directory/"cancel.requested").exists():
            raise TrainingCancelled
    def stage(name, progress):
        check_cancel()
        state.update(state="running", stage=name, progress=progress)
        write_state(directory, state)
    try:
        stage("Разделение данных", 10)
        split = split_dataset(dataset, seed=config.seed, group_column=config.group_column)
        write_splits(dataset, split, directory/"splits", target=config.target)
        stage("Базовые модели", 25)
        run_baseline_experiment(directory/"splits", config, directory/"baseline")
        stage("Обучение нейросетей", 45)
        candidates = [NetworkConfig(hidden_layers=(32, 16), learning_rate=rate, max_epochs=epochs,
            patience=min(12, epochs), seed=config.seed) for rate in [.001, .003]]
        run_neural_experiment(directory/"splits", config, candidates, directory/"neural", checkpoint=check_cancel)
        stage("Сравнение по validation", 75)
        comparison = run_comparison([directory/"baseline", directory/"neural"], directory/"comparison")
        selected = next(item for item in comparison["entries"] if item["id"] == comparison["selected"])
        family = "baseline" if selected["id"].startswith("1:") else "neural"
        stage("Диагностика и графики", 90)
        run_diagnostics(directory/"splits", directory/"neural", directory/"diagnostics-neural")
        if family == "baseline":
            run_diagnostics(directory/"splits", directory/"baseline", directory/"diagnostics-selected")
        export_plots(directory/"neural", directory/"diagnostics-neural", directory/"figures")
        stage("Анализ validation выбранной модели", 95)
        analyses = run_analysis_suite(directory/"splits", directory/family, directory/"analysis", checkpoint=check_cancel)
        package = run_research_report(directory/"splits", directory/family, directory/"research",
            existing_analyses=directory/"analysis")
        check_cancel()
        state.update(state="completed", stage="Готово", progress=100,
            completed_at=datetime.now(timezone.utc).isoformat(), result={"selected_family": family,
                "selected_model": selected["model"], "analysis_status": analyses, "research_status": package["status"], "metrics": selected["metrics"], "rows": comparison["rows"],
                "data_origin": comparison["data_origin"], "intended_use": comparison["intended_use"]})
        write_state(directory, state)
    except TrainingCancelled:
        state.update(state="cancelled", stage="Обучение остановлено", progress=0,
            completed_at=datetime.now(timezone.utc).isoformat())
        write_state(directory, state)
    except Exception as exc:
        (directory/"error.log").write_text(traceback.format_exc(), encoding="utf-8")
        state.update(state="failed", stage="Обучение не завершено", progress=0,
            error=str(exc) if isinstance(exc, DatasetError) else "Не удалось завершить эксперимент. Проверьте данные и повторите запуск.")
        write_state(directory, state)


class TrainingManager:
    def __init__(self, repository):
        self.repository = Path(repository)
        self.root = self.repository/".local/gui-jobs"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="university-training")
        self.active = None
        self.lease = TrainingLease(self.root.parent/"gui-training.lock")
        self.marker = self.root.parent/"gui-active.json"
        self.runner_id = uuid.uuid4().hex

    def prepare(self, source, values, epochs=60):
        if type(epochs) is not int or not 1 <= epochs <= 100:
            raise DatasetError('Для интерфейса допустимо от 1 до 100 эпох.')
        data = load_dataset(str(source))
        data = Dataset(data.columns, data.rows, "gui-upload", data.sha256, data.origin_kind)
        if values is None:
            if data.origin_kind != "synthetic":
                raise DatasetError("Для собственных данных загрузите JSON-контракт эксперимента.")
            config = ExperimentConfig.from_file(self.repository/"data/simulation-v1/experiment.json")
        else:
            config = ExperimentConfig.from_dict(values)
        config.check_columns(data)
        if config.task != "classification":
            raise DatasetError("Обучение в текущем интерфейсе поддерживает классификацию.")
        NetworkConfig(max_epochs=epochs, seed=config.seed).validate()
        checks = validate_dataset(data, config.target)
        if not checks["ok"]:
            raise DatasetError("; ".join(checks["issues"]))
        if len(data.rows) > 10000 or len(config.features) > 100:
            raise DatasetError("В интерфейсе обучение ограничено 10 000 строками и 100 входными признаками.")
        labels = target_vector(data, config)
        if not 2 <= len(set(labels)) <= 20:
            raise DatasetError("Для обучения требуется от 2 до 20 классов.")
        expanded = len(config.numeric_features)
        for feature in config.categorical_features:
            index = data.columns.index(feature)
            expanded += len({row[index] for row in data.rows})+1
        if expanded > 512:
            raise DatasetError("Слишком много категорий: после кодирования допустимо до 512 входов.")
        feature_matrix(data, config)
        split = split_dataset(data, seed=config.seed, group_column=config.group_column)
        target_index = data.columns.index(config.target)
        train_classes = {row[target_index] for row in split.partitions['train']}
        if len(train_classes) < 2:
            raise DatasetError('В train должно быть минимум два класса; проверьте независимые группы и распределение цели.')
        summary = {'schema_version': 1, 'source_sha256': data.sha256, 'data_origin': data.origin_kind,
            'rows': len(data.rows), 'columns': len(data.columns), 'target': config.target,
            'features': list(config.features), 'excluded_columns': list(config.excluded_columns),
            'group_column': config.group_column, 'seed': config.seed, 'epochs': epochs,
            'encoded_feature_upper_bound': expanded,
            'partitions': {name: {'rows': len(rows), 'groups': split.group_counts[name]}
                           for name, rows in split.partitions.items()},
            'warnings': checks['warnings'],
            'limitation': 'Проверка структуры не измеряет качество будущей модели. Test не используется для выбора параметров.'}
        return data, config, summary

    def submit(self, source, values, owner, epochs=60):
        if type(epochs) is not int or not 1 <= epochs <= 100:
            raise DatasetError("Для интерфейса допустимо от 1 до 100 эпох.")
        with self.lock:
            if self.active is not None and not self.active.done():
                raise TrainingBusy("Один эксперимент уже выполняется. Дождитесь его завершения.")
            if not self.lease.acquire():
                raise TrainingBusy('Обучение уже выполняется в другом окне или процессе приложения.')
            enqueued = False
            job_id = None
            try:
                data, config, preflight = self.prepare(source, values, epochs)
                job_id = uuid.uuid4().hex
                directory = self.root/job_id
                directory.mkdir(mode=0o700)
                try:
                    with Path(source).open('rb') as stream:
                        original = stream.read(20*1024*1024+1)
                    if hashlib.sha256(original).hexdigest() != data.sha256:
                        raise DatasetError('Исходный файл изменился после проверки. Повторите загрузку.')
                    (directory/'input.csv').write_bytes(original)
                    (directory/'preflight.json').write_text(json.dumps(preflight, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
                    (directory/"config.json").write_text(json.dumps(config.as_dict(), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
                    state = {"id": job_id, "owner": hashlib.sha256(owner.encode()).hexdigest(), "pid": os.getpid(), "runner_id": self.runner_id,
                        "state": "queued", "stage": "Ожидает запуска", "progress": 0,
                        "created_at": datetime.now(timezone.utc).isoformat(), "epochs": epochs}
                    write_json(self.marker,{"job_id":job_id,"runner_id":self.runner_id})
                    write_state(directory, state)
                    self.active = self.executor.submit(self._run, directory, data, config, epochs)
                    enqueued = True
                except BaseException:
                    shutil.rmtree(directory, ignore_errors=True)
                    raise
                return self.get(job_id, owner)
            finally:
                if not enqueued:
                    self._clear_marker(job_id)
                    self.lease.release()

    def _run(self, directory, data, config, epochs):
        try:
            run_job(directory, data, config, epochs)
        finally:
            self._clear_marker(directory.name)
            self.lease.release()

    def _clear_marker(self,job_id):
        try:
            marker=json.loads(self.marker.read_text())
            if marker.get('job_id')==job_id and marker.get('runner_id')==self.runner_id:
                self.marker.unlink(missing_ok=True)
        except (OSError,ValueError,AttributeError):
            pass

    def _runner_alive(self,job_id,state):
        probe=TrainingLease(self.lease.path)
        if probe.acquire():
            probe.release()
            return False
        try:
            marker=json.loads(self.marker.read_text())
        except (OSError,ValueError):
            return None
        return isinstance(marker,dict) and marker.get('job_id')==job_id and marker.get('runner_id')==state['runner_id']

    def directory(self, job_id, owner):
        if not isinstance(job_id, str) or len(job_id) != 32 or any(character not in "0123456789abcdef" for character in job_id):
            raise JobNotFound
        directory = self.root/job_id
        if directory.resolve().parent != self.root.resolve():
            raise JobNotFound
        try:
            state = json.loads((directory/"status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise JobNotFound from exc
        if not isinstance(state, dict) or not isinstance(state.get("owner"), str) or state.get("id") != job_id:
            raise JobNotFound
        if not hmac.compare_digest(state["owner"], hashlib.sha256(owner.encode()).hexdigest()):
            raise JobNotFound
        return directory

    def get(self, job_id, owner):
        directory = self.directory(job_id, owner)
        state = json.loads((directory/"status.json").read_text(encoding="utf-8"))
        if state.get('state') in {'queued','running'} and state.get('runner_id'):
            if self._runner_alive(job_id,state) is False:
                state.update(state='failed',stage='Запуск прерван',progress=0,
                    completed_at=datetime.now(timezone.utc).isoformat(),
                    error='Рабочий процесс завершился до окончания эксперимента. Исходные данные сохранены; создайте новый запуск.')
                write_state(directory,state)
        elif state.get("state") in {"queued", "running"} and state.get("pid") != os.getpid():
            try:
                pid = state.get("pid")
                if type(pid) is not int or pid <= 0:
                    raise ProcessLookupError
                os.kill(pid, 0)
            except ProcessLookupError:
                state.update(state="failed", stage="Запуск прерван", progress=0,
                    error="Предыдущий процесс завершился до окончания обучения. Создайте новый эксперимент.")
            except PermissionError:
                pass
        if state.get('state') in {'running', 'queued'} and (directory/'cancel.requested').exists():
            state.update(state='cancelling', stage='Остановка после текущего шага')
        return {key: value for key, value in state.items() if key not in {"owner", "pid", "runner_id"}}

    def cancel(self, job_id, owner):
        with self.lock:
            directory = self.directory(job_id, owner)
            state = self.get(job_id, owner)
            if state['state'] in {'running', 'queued', 'cancelling'}:
                (directory/'cancel.requested').touch(exist_ok=True)
            return self.get(job_id, owner)

    def list(self, owner):
        result = []
        for directory in self.root.iterdir():
            if not directory.is_dir():
                continue
            try:
                result.append(self.get(directory.name, owner))
            except JobNotFound:
                continue
        return sorted(result, key=lambda item: item["created_at"], reverse=True)[:20]

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.lease.release()
