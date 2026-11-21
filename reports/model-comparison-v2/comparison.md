# Сравнение моделей

Происхождение: synthetic. Назначение: development_demo_only.
Выбор по validation; test не участвует в ранжировании.

| Модель | Train macro_f1 | Validation | Test |
| --- | ---: | ---: | ---: |
| simulation-baseline-v1 / dummy | 0.24713 | 0.23625 | 0.24709 |
| simulation-baseline-v1 / linear | 0.75426 | 0.79764 | 0.74705 |
| simulation-neural-v1 / neural | 0.76106 | 0.79580 | 0.77140 |

Выбрана модель: 1:linear. При равенстве сохраняется первый вариант.
