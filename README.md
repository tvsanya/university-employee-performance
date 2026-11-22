# Оценка эффективности сотрудника университета

Тема: «Автоматизация процессов оценки эффективности работы сотрудника университета с применением интеллектуального анализа данных».
Автор: Бакулин Александр Максимович (`tvsanya`).

## Текущий этап

Реализованы загрузка и проверка данных, контракт задачи, подготовка признаков и сравнение baseline. По выбору пользователя для разработки используется явно помеченная симуляция: 1500 условных преподавателей × 2 периода, 3000 строк и 16 колонок. Обе baseline-модели обучены на этой симуляции и сохранены в `models/simulation-baseline-v1`. Это демонстрация программного конвейера; результаты не являются оценкой реальных сотрудников и не подтверждают требования к реальным данным ВКР.

## Запуск

Python 3.12+. Для подготовки признаков установите зависимости с зафиксированными версиями:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

```sh
python3 main.py inspect https://example.org/dataset.csv --report .local/profile.json
python3 main.py inspect data.csv
python3 main.py validate data.csv --target score --report .local/validation.json
python3 main.py check-spec data.csv --config .local/experiment.json
python3 main.py split data.csv --target score --group-column employee_id --output .local/split-v1 --seed 42
python3 main.py train-baseline --splits .local/split-v1 --config .local/experiment.json --output .local/experiments/baseline-v1
python3 main.py inspect https://example.org/dataset.zip --member data.csv
python3 -m unittest discover -v
```

Адреса example.org показывают синтаксис, их нужно заменить действительным источником. Локальные файлы служат разработке и проверкам; итоговый сценарий загрузки набора должен работать по ссылке.

`split` сначала проверяет минимальный объём и целевую колонку, затем создаёт `train.csv`, `validation.csv`, `test.csv` и `manifest.json` с контрольными суммами. Один сотрудник остаётся в одной выборке при указании `--group-column`. Без неё вместе остаются точные дубли строк. Доли 70/15/15 относятся к группам; фактические размеры строк записаны в манифесте. Результат воспроизводим по исходным данным и seed. Существующий каталог не перезаписывается.

План: `PLAN.md`. Требования: `docs/REQUIREMENTS.md`. Учебные фикстуры в тестах не используются для обучения или результатов ВКР.

Кандидаты и их ограничения: `docs/DATA_SOURCES.md`. Для повторения проверки реального источника:

```sh
python3 main.py inspect 'https://archive.ics.uci.edu/static/public/262/turkiye%2Bstudent%2Bevaluation.zip' --member turkiye-student-evaluation_generic.csv --report .local/uci-candidate.json
```

Контракт задачи и правила подготовки признаков описаны в `docs/MODELING.md`. Медианы, масштаб и словарь категорий оцениваются на train; неизвестные категории при применении не расширяют словарь. Целевой столбец, идентификатор группы и явно исключённые поля не входят в признаки.

`train-baseline` проверяет контрольные суммы и независимость выборок, обучает простой и линейный baseline только на train, сравнивает по validation, оценивает все три выборки и сохраняет обе модели, конфигурацию, метрики, версии библиотек и контрольные суммы. Сохранённые joblib-модели следует загружать только из доверенных собственных экспериментов. Происхождение synthetic сохраняется в CSV, манифесте и метриках.

## Симуляционный сценарий

```sh
python main.py generate-demo --output .local/simulation-new
python main.py split data/simulation-v1/synthetic_employees.csv --target efficiency_level --group-column employee_id --output .local/simulation-splits-new
python main.py train-baseline --splits .local/simulation-splits-new --config data/simulation-v1/experiment.json --output .local/simulation-baseline-new
```

Готовые данные, provenance и контракт находятся в `data/simulation-v1`. Признаки и метки генерируются из искусственного скрытого фактора с шумом. Поле efficiency_score исключено из признаков классификации efficiency_level. ФИО реальных людей отсутствуют. Модель, демонстрационный интерфейс и DL можно разрабатывать на этом наборе; требование задания о реальных исследовательских данных остаётся невыполненным.

Обучение и подбор DL по внешней validation:

```sh
.venv/bin/python main.py train-neural --splits .local/simulation-splits-v1 --config data/simulation-v1/experiment.json --grid configs/network-grid-v1.json --output models/new-neural-experiment
```

Сохранённый демонстрационный запуск — models/simulation-neural-v1. Его метрики относятся к синтетическому сценарию; детали и сравнение с baseline приведены в docs/MODELING.md.

Применить сеть без обучения и без pickle-десериализации:

```sh
.venv/bin/python main.py predict path/to/new.csv --model models/simulation-neural-v1/model.json --output .local/new-predictions
```

CSV должен содержать признаки из config.json; цель необязательна. В новом каталоге появятся predictions.csv и report.json. Происхождение synthetic и назначение development_demo_only сохраняются в JSON-артефакте и отчёте.

Локальный интерфейс:

```sh
.venv/bin/python main.py serve --port 8033
```

Откройте http://127.0.0.1:8033. Доступны проверка CSV, просмотр экспериментов/графиков, загрузка JSON-модели, прогноз и CSV-результат. Обучение запускается командами CLI и фоновым заданием в GUI; доступны собственные результаты, выбранная модель и архив.

Дополнительные команды:

```sh
.venv/bin/python main.py diagnose --splits .local/simulation-splits-v1 --experiment models/simulation-neural-v1 --output reports/new-diagnostics
.venv/bin/python main.py compare --experiments models/simulation-baseline-v1 models/simulation-neural-v1 --output reports/new-comparison
.venv/bin/python main.py plot --experiment models/simulation-neural-v1 --diagnostics reports/simulation-neural-v1 --output reports/new-figures
```

Существующий результат не перезаписывается. JSON/CSV/Markdown сравнения и PNG/SVG графиков сохранены в reports/. Эти материалы относятся только к симуляции.

GUI теперь обучает собственный эксперимент в фоне: CSV с целью, JSON-контракт (для помеченной комплектной симуляции необязателен), до 100 эпох. Сохраняются обе семьи, сравнение по validation, диагностика и графики. Результаты доступны только своей браузерной сессии; выбранную модель можно применить, скачать либо забрать полный ZIP с данными/параметрами/версиями и хешами. Локальные результаты и ключ сессии находятся в игнорируемом `.local`.

Воспроизводимый notebook с результатами: `notebooks/research.ipynb`. Повторное исполнение проектным Python:

```sh
.venv/bin/python scripts/execute_notebook.py
```

Скрипт создаёт ядро внутри `.local`, выполняет все ячейки и сохраняет notebook только после успеха. При интерактивном открытии выберите Python из `.venv`. Математика и ограничения — docs/MATHEMATICS.md; проверки и открытые критерии — docs/TESTING.md. JSON-применение доступно также для выбранных классификационных baseline.

Анализ признаков: `main.py importance --splits КАТАЛОГ_ВЫБОРОК --experiment КАТАЛОГ_МОДЕЛИ --output НОВЫЙ_КАТАЛОГ`. Используется только validation; test не участвует в выборе признаков.

Дополнительный анализ: команды `probabilities`, `robustness` и `uncertainty` принимают --splits, --experiment и --output. После GUI-обучения выбранная модель автоматически получает пять отчётов: распределения train/validation, важность, вероятности, устойчивость и групповые интервалы. Они скачиваются отдельно и входят в архив. Ресурсное ограничение или неприменимость отдельного анализа отражаются status=unavailable с причиной и не выдаются за успешную оценку.

Команда `distribution` сравнивает входные распределения train/validation: числовые KS/PSI, неизвестные категории, TV и пропуски. Границы гистограмм определяет train; цель и test в анализе не используются.

## Запуск из IDE

Выберите интерпретатор `.venv/bin/python`, откройте `run_local.py` и нажмите Run без параметров. Адрес: http://127.0.0.1:8033 . Подготовка и проверка: `python run_local.py --check`. Пошаговая инструкция — [docs/LOCAL_RUN.md](docs/LOCAL_RUN.md).
