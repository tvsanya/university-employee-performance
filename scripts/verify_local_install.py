"""Проверка GUI, notebook и модели в новом окружении из отслеживаемых исходников."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import urlopen
import venv


def verify(report_path):
    root = Path(__file__).resolve().parents[1]
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    sources = {}
    with tempfile.TemporaryDirectory(prefix="university-install-") as directory:
        target = Path(directory)
        for name in filter(None, names):
            original = root/name
            if original.is_symlink():
                raise RuntimeError("Символические ссылки в комплекте требуют отдельной проверки.")
            destination = target/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(original, destination)
            sources[name] = hashlib.sha256(original.read_bytes()).hexdigest()
        venv.EnvBuilder(with_pip=True).create(target/".venv")
        python = target/".venv"/("Scripts/python.exe" if os.name == "nt" else "bin/python")
        env = {key:value for key,value in os.environ.items() if key not in
            {"DATABASE_URL", "SECRET_KEY", "DEBUG", "ALLOWED_HOSTS", "DJANGO_SETTINGS_MODULE", "PYTHONPATH", "PYTHONHOME"}}
        env.update(DEBUG="1", ALLOWED_HOSTS="127.0.0.1,localhost,testserver")
        with (target/"verification.log").open("w+") as log:
            def run(*args):
                result = subprocess.run([str(python), *args], cwd=target, env=env, stdout=log, stderr=log, timeout=180)
                if result.returncode:
                    log.seek(0)
                    raise RuntimeError(log.read()[-6000:])
            run("-m", "pip", "install", "-r", "requirements.txt")
            run("-m", "pip", "check")
            run("run_local.py", "--check")
            run("-m", "unittest", "discover", "-q")
            run("scripts/execute_notebook.py")
            run("main.py", "predict", "data/simulation-v1/synthetic_employees.csv", "--model", "models/simulation-baseline-v1/model.json", "--output", ".local/acceptance-predictions")
            with socket.socket() as listener:
                listener.bind(("127.0.0.1", 0))
                port = listener.getsockname()[1]
            server = subprocess.Popen([str(python), str(target/"run_local.py"), "--port", str(port)], cwd=root.parent, env=env, stdout=log, stderr=log)
            try:
                deadline = time.monotonic()+30
                while True:
                    try:
                        with urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                            assert response.status == 200
                        break
                    except (URLError, TimeoutError):
                        if server.poll() is not None or time.monotonic() >= deadline:
                            raise RuntimeError("Чистый локальный сервер не запустился.")
                        time.sleep(.2)
                with urlopen(f"http://127.0.0.1:{port}/static/app.css", timeout=2) as response:
                    assert response.status == 200 and b"{" in response.read()
            finally:
                server.terminate()
                try: server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait(timeout=5)
            log.seek(0)
            transcript = log.read()
        versions = subprocess.check_output([str(python), "-m", "pip", "freeze"], cwd=target, env=env, text=True).splitlines()
    result = {"schema_version": 1, "status": "passed", "source_sha256": sources,
        "python": sys.version, "dependencies": versions,
        "checks": ["isolated_venv_install", "pip_check", "full_test_suite", "executed_research_notebook", "bundled_model_prediction", "loopback_http_from_other_cwd", "css_http"],
        "verification_log_tail": transcript[-2200:],
        "limitations": ["HTTP and Django clients do not replace visual browser acceptance"]}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    print(f"Чистая локальная установка проверена: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("reports/local-install-v1.json"))
    verify(parser.parse_args().report.resolve())
