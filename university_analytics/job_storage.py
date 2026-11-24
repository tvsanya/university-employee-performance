"""Учёт места и явная очистка только своей корзины; результаты не удаляются автоматически."""
import hashlib
import hmac
import os
from pathlib import Path
import re
import shutil
from .data import DatasetError
from .job_state import read_state

JOB_NAME=re.compile(r'[0-9a-f]{32}\Z')
TRASH_NAME=re.compile(r'[0-9a-f]{32}-[0-9a-f]{32}\Z')
MAX_ENTRIES=10000


def own_directories(root,owner,trash=False):
    if root.is_symlink():raise DatasetError('Каталог результатов не должен быть ссылкой.')
    if not root.exists():return []
    expected=hashlib.sha256(owner.encode()).hexdigest()
    result=[]
    for number,path in enumerate(root.iterdir(),1):
        if number>2000:raise DatasetError('Больше 2000 каталогов результатов. Требуется ручная проверка хранилища.')
        if not (TRASH_NAME if trash else JOB_NAME).fullmatch(path.name) or path.is_symlink() or not path.is_dir():continue
        try:state=read_state(path/'status.json')
        except (OSError,ValueError,UnicodeError):continue
        if state['id']!=path.name[:32] or (trash and state['state'] not in {'completed','failed','cancelled'}):continue
        if hmac.compare_digest(state['owner'],expected):result.append(path)
    return result


def walk_error(error):
    raise DatasetError("Не удалось полностью прочитать каталог результатов; частичный объём не показан.") from error


def tree_bytes(paths,budget):
    total=0
    for path in paths:
        for folder,dirs,files in os.walk(path,followlinks=False,onerror=walk_error):
            dirs[:]=[name for name in dirs if not (Path(folder)/name).is_symlink()]
            budget[0]+=len(dirs)+len(files)
            if budget[0]>MAX_ENTRIES:raise DatasetError('Больше 10000 объектов в личных результатах. Полный подсчёт не выполнен; проверьте каталоги вручную.')
            for name in files:
                file=Path(folder)/name
                try:
                    if not file.is_symlink() and file.is_file():total+=file.stat().st_size
                except FileNotFoundError:continue
    return total


def storage_report(jobs,owner):
    jobs=Path(jobs);trash=jobs.parent/'gui-trash'
    own=own_directories(jobs,owner);removed=own_directories(trash,owner,trash=True)
    budget=[0]
    return {'schema_version':1,'jobs':len(own),'job_bytes':tree_bytes(own,budget),
        'trash':len(removed),'trash_bytes':tree_bytes(removed,budget),
        'disk_free_bytes':shutil.disk_usage(jobs).free,
        'limitations':['Logical file sizes, not allocated disk blocks; files may change during calculation.',
            'Only valid states owned by this browser session are counted; damaged states and symlinks are left for manual inspection.']}


def purge_own_trash(jobs,owner):
    paths=own_directories(Path(jobs).parent/'gui-trash',owner,trash=True)
    deleted=failed=0
    for path in paths:
        try:shutil.rmtree(path);deleted+=1
        except FileNotFoundError:continue
        except OSError:failed+=1
    return {'deleted_directories':deleted,'failed_directories':failed}
