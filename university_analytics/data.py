"""Чтение исходного набора с сохранением происхождения и контрольной суммы."""
import csv
import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile

MAX_BYTES = 20 * 1024 * 1024


class DatasetError(ValueError):
    """Ошибка входных данных, которую можно показать пользователю."""


@dataclass(frozen=True)
class Dataset:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    source: str
    sha256: str


def read_source(source: str) -> bytes:
    """Загружает публичную HTTP-ссылку либо явно заданный локальный CSV."""
    parsed = urlparse(source)
    if parsed.scheme in {"http", "https"}:
        if parsed.username or parsed.password:
            raise DatasetError("Не передавайте учётные данные внутри URL датасета.")
        request = Request(source, headers={"User-Agent": "UniversityAnalytics/0.1"})
        with urlopen(request, timeout=30) as response:
            if urlparse(response.geturl()).scheme not in {"http", "https"}:
                raise DatasetError("Недопустимый адрес после перенаправления.")
            payload = response.read(MAX_BYTES + 1)
    elif parsed.scheme:
        raise DatasetError("Разрешены HTTP/HTTPS-ссылки и пути к локальным CSV.")
    else:
        with Path(source).open("rb") as stream:
            payload = stream.read(MAX_BYTES + 1)
    if len(payload) > MAX_BYTES:
        raise DatasetError("Размер набора превышает допустимые 20 МиБ.")
    if not payload:
        raise DatasetError("Источник данных пуст.")
    return payload


def load_dataset(source: str, member: str | None = None, delimiter: str | None = None) -> Dataset:
    payload = read_source(source)
    if member:
        try:
            with ZipFile(io.BytesIO(payload)) as archive:
                info = archive.getinfo(member)
                if info.file_size > MAX_BYTES:
                    raise DatasetError("CSV внутри архива превышает 20 МиБ.")
                # Читаем выбранный файл в память, не распаковывая пути на диск.
                with archive.open(info) as stream:
                    payload = stream.read(MAX_BYTES + 1)
        except (BadZipFile, KeyError) as exc:
            raise DatasetError("Не найден выбранный CSV в корректном ZIP-архиве.") from exc
    if len(payload) > MAX_BYTES:
        raise DatasetError("CSV превышает 20 МиБ.")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise DatasetError("CSV должен быть в кодировке UTF-8.") from exc
    try:
        if delimiter is None:
            try:
                delimiter = csv.Sniffer().sniff(text[:8192], delimiters=",;\t|").delimiter
            except csv.Error:
                delimiter = ","
        if delimiter not in {",", ";", "\t", "|"}:
            raise DatasetError("Разделитель должен быть запятой, точкой с запятой, табуляцией или |.")
        records = list(csv.reader(io.StringIO(text), delimiter=delimiter, strict=True))
    except csv.Error as exc:
        raise DatasetError("Не удалось разобрать CSV.") from exc
    records = [row for row in records if row]
    if not records:
        raise DatasetError("CSV не содержит заголовка.")
    columns = tuple(name.strip() for name in records[0])
    if any(not name for name in columns) or len(set(columns)) != len(columns):
        raise DatasetError("Названия колонок должны быть непустыми и уникальными.")
    rows = []
    for number, row in enumerate(records[1:], start=2):
        if len(row) != len(columns):
            raise DatasetError(f"Строка {number}: ожидалось {len(columns)} значений, получено {len(row)}.")
        rows.append(tuple(value.strip() for value in row))
    return Dataset(columns, tuple(rows), source + (f"#{member}" if member else ""), hashlib.sha256(payload).hexdigest())


def basic_profile(dataset: Dataset) -> dict:
    return {
        "source": dataset.source, "sha256": dataset.sha256,
        "row_count": len(dataset.rows), "column_count": len(dataset.columns),
        "columns": list(dataset.columns),
    }
