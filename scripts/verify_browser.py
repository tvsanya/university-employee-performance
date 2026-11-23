"""Фактический Chrome: CSV по HTTP, обучение, прогноз и приватные артефакты."""
import argparse
import csv
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
import threading
import time
from urllib.error import URLError
from urllib.request import urlopen
from zipfile import ZipFile
from playwright.sync_api import sync_playwright, expect


def verify(app_python, output):
    root=Path(__file__).resolve().parents[1];app_python=Path(os.path.abspath(app_python));output.mkdir(parents=True,exist_ok=True)
    names=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
    sources={};checks=[];screens=[]
    env={key:value for key,value in os.environ.items() if key not in {'PYTHONPATH','PYTHONHOME'}}
    with tempfile.TemporaryDirectory(prefix='university-browser-') as directory:
        target=Path(directory)
        for name in filter(None,names):
            path=root/name
            if path.is_symlink():raise RuntimeError('Symlink in source kit')
            destination=target/name;destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
            sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        payload=(target/'data/simulation-v1/synthetic_employees.csv').read_bytes()
        source_requests=[]
        class Handler(BaseHTTPRequestHandler):
            def do_GET(inner):
                source_requests.append(inner.path);inner.send_response(200);inner.send_header('Content-Type','text/csv')
                inner.send_header('Content-Length',str(len(payload)));inner.end_headers();inner.wfile.write(payload)
            def log_message(inner,*args):pass
        fixture=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=fixture.serve_forever,daemon=True);thread.start()
        with socket.socket() as listener:listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
        url=f'http://127.0.0.1:{port}'
        with (target/'server.log').open('w') as log:
            server=subprocess.Popen([str(app_python),str(target/'run_local.py'),'--port',str(port)],cwd=root.parent,env=env,stdout=log,stderr=log)
            try:
                deadline=time.monotonic()+30
                while True:
                    try:
                        with urlopen(url,timeout=2) as response:assert response.status==200
                        break
                    except (URLError,TimeoutError):
                        if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Local GUI failed')
                        time.sleep(.1)
                with sync_playwright() as p:
                    browser=p.chromium.launch(channel='chrome',headless=True)
                    context=browser.new_context(viewport={'width':1440,'height':1000},locale='ru-RU',accept_downloads=True)
                    page=context.new_page();errors=[];page.on('pageerror',lambda error:errors.append(str(error)))
                    page.goto(url);page.wait_for_selector('#experiment option[value="simulation-baseline-v1"]',state='attached')
                    def shot(name,selector=None):
                        if selector:page.locator(selector).screenshot(path=str(output/name))
                        else:page.screenshot(path=str(output/name),full_page=True)
                        screens.append(name)
                    shot('initial-desktop.png')
                    page.locator('#source-url').fill(f'http://127.0.0.1:{fixture.server_port}/simulated.csv')
                    page.get_by_role('button',name='Проверить данные',exact=True).click()
                    expect(page.locator('#status')).to_contain_text('Строк: 3000',timeout=30000)
                    assert 'симуляционные данные' in page.locator('#summary').inner_text();checks.append('actual_http_csv_profile')
                    page.locator('#experiment').select_option('simulation-baseline-v1')
                    page.get_by_role('button',name='Получить прогноз',exact=True).click()
                    expect(page.locator('#result-heading')).to_have_text('Прогнозы готовы',timeout=30000)
                    assert page.locator('#table tbody tr').count()==30;shot('prediction-desktop.png','.workspace')
                    with page.expect_download() as result:page.get_by_role('button',name='Скачать все прогнозы CSV',exact=True).click()
                    predictions=output/'predictions-browser.csv';result.value.save_as(predictions)
                    records=list(csv.DictReader(io.StringIO(predictions.read_text(encoding='utf-8-sig'))))
                    assert len(records)==3000 and 'review_recommended' in records[0];checks.append('3000_predictions_and_csv_review_flags')
                    page.locator('#train-data').set_input_files(target/'data/simulation-v1/synthetic_employees.csv')
                    page.locator('#epochs').fill('2');page.get_by_role('button',name='Начать обучение',exact=True).click()
                    expect(page.locator('#job-stage')).to_have_text('Готово',timeout=120000)
                    expect(page.locator('#job-metrics tbody tr')).to_have_count(3,timeout=30000)
                    checks.append('actual_background_baseline_and_two_dl_candidates')
                    job_id=page.evaluate('sessionStorage.getItem("university-job-id")')
                    state=context.request.get(url+'/api/jobs/'+job_id).json()
                    assert state['state']=='completed' and state['result']['intended_use']=='development_demo_only'
                    assert len(state['result']['analysis_status'])==5;checks.append('five_validation_analyses')
                    shot('trained-desktop.png','#training-result')
                    with page.expect_download() as result:page.get_by_role('link',name='Полный эксперимент с данными ZIP',exact=True).click()
                    archive=target/'experiment.zip';result.value.save_as(archive)
                    with ZipFile(archive) as bundle:
                        hashes=json.loads(bundle.read('SHA256SUMS.json'))
                        assert all(hashlib.sha256(bundle.read(name)).hexdigest()==value for name,value in hashes.items())
                        assert bundle.read('input.csv')==payload and 'status.json' not in bundle.namelist()
                    checks.append('downloaded_archive_hashes_and_source')
                    page.get_by_role('button',name='Выбрать для прогноза',exact=True).click()
                    expect(page.locator('#experiment')).to_have_value('job:'+job_id)
                    page.get_by_role('button',name='Получить прогноз',exact=True).click()
                    expect(page.locator('#status')).to_contain_text('Обработано строк: 3000',timeout=30000)
                    checks.append('selected_job_applied')
                    stranger=browser.new_context();stranger.new_page().goto(url)
                    assert stranger.request.get(url+'/api/jobs/'+job_id).status==404;stranger.close();checks.append('other_session_cannot_read_job')
                    page.set_viewport_size({'width':390,'height':844})
                    for name,selector in [('form-mobile.png','#data-form'),('prediction-mobile.png','.results'),('trained-mobile.png','#training-result')]:
                        shot(name,selector)
                        dimensions=page.evaluate('({width:document.documentElement.clientWidth,scroll:document.documentElement.scrollWidth})')
                        assert dimensions['scroll']<=dimensions['width']+1,(name,dimensions)
                    checks.append('mobile_390_no_document_overflow');assert not errors,errors;checks.append('no_browser_page_errors')
                    browser_version=browser.version;context.close();browser.close()
            finally:
                server.terminate()
                try:server.wait(timeout=15)
                except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
                fixture.shutdown();fixture.server_close();thread.join(timeout=2)
        assert len(source_requests)>=3
    result={'status':'passed','browser':'Chrome '+browser_version,'playwright':version('playwright'),'source_sha256':sources,
        'checks':checks,'screenshots':screens,'viewports':[[1440,1000],[390,844]],'training_epochs_for_smoke':2,
        'limitations':['Only synthetic data; two smoke epochs do not replace the saved full research experiment.','One browser, not a full accessibility or cross-browser audit.']}
    (output/'browser.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('Chrome acceptance passed:',len(checks),'checks')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--app-python',default='.venv/bin/python')
    parser.add_argument('--output',type=Path,default=Path('reports/local-visual-v1'))
    args=parser.parse_args();verify(args.app_python,args.output.resolve())
