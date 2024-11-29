## step 1 / собираем контексты для батчей

Делаем нарезку из 4-х крупных датасетов (варианты запуска):

Генерация 1000 контекстов, по 100 в файле

`python 01_context_preparation.py --total 1000 --batch_size 100 # всего 1000 контекстов, по 100 штук в файле, всего 10 файлов`


Тестовый прогон с маленьким количеством

`python 01_context_preparation.py --total 50 --batch_size 10 # всего 50 контекстов, по 10 штук в файле, всего 5 файлов`

Полный прогон с параметрами по умолчанию (200к контекстов)

`python 01_context_preparation.py`

== файлы с контекстами падают в папку `clean_contexts`

## step 2 / пакуем контексты в батчи

```python
python3 02_full_batch_generator.py \
    --total 50 \ # полный размер всего батча
    --batch_size 1000 \ # количество в одном файле
    --input_dir clean_contexts \ # тут понятно
    --output_dir api_batches
```

== файлы с батчами падают в папку `api_batches`

## step 3 / загружаем батчи руками 

`https://platform.openai.com/batches`

чем крупнее файл тем быстрее обработка, окно обработки 24 часа, но по факту быстрее, зависит от загрузки серверов openai.
После обработки сгружаем и складываем например в папку 
`api_responses`

## step 4 / выполняем очистку готовых файлов и собираем в кучу

```python
python 03_response-processor.py \
    --input_dir api_responses \
    --output_dir processed_data
```
на выходе получаем готовые csv в папке processed_data
