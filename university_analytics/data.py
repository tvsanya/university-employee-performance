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
MAX_ROWS = 50000
MAX_COLUMNS = 256
MAX_CELLS = 2_000_000
MAX_RECORDS = 100000
MAX_RECORD_CHARS = 256 * 1024


class DatasetError(ValueError):
    """Ошибка входных данных, которую можно показать пользователю."""


@dataclass(frozen=True)
class Dataset:
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    source: str
    sha256: str
    origin_kind: str = "unspecified"


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
        rows = []
        columns = None
        stream = io.StringIO(text, newline='')
        record_chars = 0
        def bounded_lines():
            nonlocal record_chars
            while line := stream.readline(MAX_RECORD_CHARS+1):
                record_chars += len(line)
                if record_chars > MAX_RECORD_CHARS:
                    raise DatasetError('Одна CSV-запись превышает 256 Ки символов.')
                yield line
        reader = csv.reader(bounded_lines(), delimiter=delimiter, strict=True)
        seen = 0
        while True:
            record_chars = 0
            try:
                row = next(reader)
            except StopIteration:
                break
            seen += 1
            if seen > MAX_RECORDS:
                raise DatasetError('CSV содержит больше 100 000 записей, включая пустые.')
            if not row:
                continue
            if columns is None:
                if len(row) > MAX_COLUMNS:
                    raise DatasetError('CSV содержит больше 256 колонок.')
                columns = tuple(name.strip() for name in row)
                if any(not name for name in columns) or len(set(columns)) != len(columns):
                    raise DatasetError('Названия колонок должны быть непустыми и уникальными.')
                continue
            if len(row) != len(columns):
                raise DatasetError(f'После физической строки {reader.line_num}: ожидалось {len(columns)} значений, получено {len(row)}.')
            if len(rows) >= MAX_ROWS or (len(rows)+1)*len(columns) > MAX_CELLS:
                raise DatasetError('Предел CSV: 50 000 строк данных и 2 000 000 ячеек; сократите набор.')
            rows.append(tuple(value.strip() for value in row))
    except csv.Error as exc:
        raise DatasetError('Не удалось разобрать CSV: проверьте кавычки и длину полей.') from exc
    if columns is None:
        raise DatasetError('CSV не содержит заголовка.')
    origin = "unspecified"
    if "source_kind" in columns and rows:
        flags = {row[columns.index("source_kind")] for row in rows}
        if flags == {"synthetic"}:
            origin = "synthetic"
        elif "synthetic" in flags:
            origin = "mixed"
    return Dataset(columns, tuple(rows), source + (f"#{member}" if member else ""), hashlib.sha256(payload).hexdigest(), origin)


def basic_profile(dataset: Dataset) -> dict:
    return {
        "source": dataset.source, "sha256": dataset.sha256,
        "row_count": len(dataset.rows), "column_count": len(dataset.columns),
        "columns": list(dataset.columns),
        "data_origin": dataset.origin_kind,
    }
