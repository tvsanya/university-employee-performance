# Оценка эффективности сотрудника университета

Тема: «Автоматизация процессов оценки эффективности работы сотрудника университета с применением интеллектуального анализа данных».
Автор: Бакулин Александр Максимович (`tvsanya`).

## Текущий этап

Реализована загрузка CSV по HTTP/HTTPS, из локального файла для разработки и из явно выбранного файла ZIP-архива. Определяется разделитель, формируется сводка пропусков, дублей, числовых значений и контрольная сумма. Команда `validate` проверяет 1000+ строк, 10+ колонок и заданную целевую колонку. Итоговый источник и показатель эффективности ещё не утверждены; модель не обучалась.

## Запуск

Python 3.12+; первый этап не требует сторонних библиотек.

```sh
python3 main.py inspect https://example.org/dataset.csv --report .local/profile.json
python3 main.py inspect data.csv
python3 main.py validate data.csv --target score --report .local/validation.json
python3 main.py inspect https://example.org/dataset.zip --member data.csv
python3 -m unittest discover -v
```

Адреса example.org показывают синтаксис, их нужно заменить действительным источником. Локальные файлы служат разработке и проверкам; итоговый сценарий загрузки набора должен работать по ссылке.

План: `PLAN.md`. Требования: `docs/REQUIREMENTS.md`. Учебные фикстуры в тестах не используются для обучения или результатов ВКР.

Кандидаты и их ограничения: `docs/DATA_SOURCES.md`. Для повторения проверки реального источника:

```sh
python3 main.py inspect 'https://archive.ics.uci.edu/static/public/262/turkiye%2Bstudent%2Bevaluation.zip' --member turkiye-student-evaluation_generic.csv --report .local/uci-candidate.json
```
