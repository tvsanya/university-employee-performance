# Оценка эффективности сотрудника университета

Тема: «Автоматизация процессов оценки эффективности работы сотрудника университета с применением интеллектуального анализа данных».
Автор: Бакулин Александр Максимович (`tvsanya`).

## Текущий этап

Реализована загрузка CSV по HTTP/HTTPS, из локального файла для разработки и из явно выбранного файла ZIP-архива. Определяется разделитель, формируется сводка пропусков, дублей, числовых значений и контрольная сумма. Команда `validate` проверяет 1000+ строк, 10+ колонок и заданную целевую колонку. Итоговый источник и показатель эффективности ещё не утверждены; модель не обучалась.

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

`train-baseline` проверяет контрольные суммы и независимость выборок, обучает простой и линейный baseline только на train, сравнивает по validation, оценивает все три выборки и сохраняет обе модели, конфигурацию, метрики, версии библиотек и контрольные суммы. Сохранённые joblib-модели следует загружать только из доверенных собственных экспериментов. Обучение в автоматических тестах использует исключительно программные фикстуры; это не метрики и не результаты диплома. Реальный датасет и показатель эффективности ещё требуют определения.
