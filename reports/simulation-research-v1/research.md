# Диагностика сохранённой модели

Происхождение: synthetic. Назначение: development_demo_only.
Симуляционные результаты не оценивают реальных сотрудников. Test не используется для настройки анализов.
Validation: 450 строк. Модель SHA-256: 43088d11acd7029e95dc8ad7424eb5d9a1a020c61a176f0f4524636bfcb1af4b.
Источник SHA-256: 698dbfc7709f492a58f8ee1dcba83d8355173ad0ebd0ae8d257e49c5dd1e75af. Split SHA-256: 7ac643421eb3b8b655b1f7344b5e1687960e62005cb3c00f83d878ccaebff45e.

## distribution

Сравниваются train (2100 строк) и validation (450 строк). Числовые и категориальные расстояния сохранены в JSON.

Сравниваются только входные распределения. KS приводится как расстояние без p-value; PSI использует train-границы и сглаживание 0.5 наблюдения. Числа не доказывают деградацию модели, справедливость или причинность; пороги тревоги не подбирались. Test не используется.

Полный результат: [distribution/analysis.json](distribution/analysis.json).

## importance

Признак | Среднее падение macro-F1 | SD повторов
--- | ---: | ---:
planned_teaching_hours | 0.15575 | 0.01611
teaching_hours | 0.15547 | 0.00723
course_completion_rate | 0.09226 | 0.01681
student_feedback_avg | 0.08038 | 0.01528
digital_activity_rate | 0.03240 | 0.00956
admin_delay_days | 0.01164 | 0.00595
employment_fraction | 0.00667 | 0.00349
department | 0.00435 | 0.00441
publication_count | 0.00330 | 0.00466
training_hours | -0.00366 | 0.00652
position | -0.00482 | 0.00557

Перестановка оценивает зависимость данной модели от признака на validation. Коррелированные признаки и зависимые строки влияют на результат; это не причинный эффект и не доверительный интервал.

Полный результат: [importance/analysis.json](importance/analysis.json).

## probabilities

Log-loss: 0.47100; Brier (сумма по классам): 0.29115; ECE: 0.02792.

Это описательная оценка вероятностей на validation, а не калибровка модели. Brier и log-loss учитывают также различимость классов; ECE зависит от интервалов. Пороги не подбираются и не являются правилом кадрового решения.

Полный результат: [probabilities/analysis.json](probabilities/analysis.json).

## robustness

Сценарий | Уровень | Среднее падение macro-F1
--- | --- | ---:
random_missing_fraction | 0.05 | 0.01177
random_missing_fraction | 0.15 | 0.04324
random_missing_fraction | 0.3 | 0.09001
numeric_noise_train_scale | 0.05 | 0.00016
numeric_noise_train_scale | 0.15 | 0.01166
numeric_noise_train_scale | 0.3 | 0.02752
complete_feature_missing | employment_fraction | 0.00472
complete_feature_missing | planned_teaching_hours | 0.11017
complete_feature_missing | teaching_hours | 0.09950
complete_feature_missing | student_feedback_avg | 0.01739
complete_feature_missing | course_completion_rate | 0.02197
complete_feature_missing | publication_count | 0.00381
complete_feature_missing | training_hours | -0.01889
complete_feature_missing | admin_delay_days | 0.00517
complete_feature_missing | digital_activity_rate | 0.00351
complete_feature_missing | department | -0.00237
complete_feature_missing | position | -0.00606

Искусственные пропуски и шум используют сохранённый train-масштаб, без переобучения. Шум может выходить за предметные диапазоны; это стресс-тест модели, не прогноз реальных изменений сотрудников. Разброс повторов не является доверительным интервалом.

Полный результат: [robustness/analysis.json](robustness/analysis.json).

## uncertainty

Единица: group; групп: 225; повторов: 500.
accuracy: 0.80222; percentile 95%: [0.76000, 0.84000].
macro_f1: 0.79764; percentile 95%: [0.75099, 0.83745].

Интервалы условны для сохранённой модели и validation при предположении независимости групп. Они не включают неопределённость обучения, не являются оценкой причинного эффекта и не подтверждают перенос с симуляции на реальных сотрудников. При отсутствии group_column предполагается независимость строк.

Полный результат: [uncertainty/analysis.json](uncertainty/analysis.json).

