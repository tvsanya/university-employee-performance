import csv
import contextlib
import io
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from university_analytics.data import Dataset, DatasetError
from university_analytics.splitting import split_dataset, write_splits
from university_analytics.cli import main


def fixture():
    # Генерация нужна только для проверки свойств алгоритма; это не датасет ВКР.
    rows = tuple((str(employee), str(observation), str(employee % 3))
                 for employee in range(30) for observation in range(2))
    return Dataset(("employee_id", "observation", "score"), rows, "unit-test-fixture", "0" * 64)


class SplitTests(unittest.TestCase):
    def test_cli_creates_complete_split_from_valid_test_fixture(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "fixture.csv"
            destination = Path(folder) / "output"
            with source.open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["employee_id", "score", *[f"feature_{n}" for n in range(8)]])
                for n in range(1000):
                    writer.writerow([str(n), str(n % 5), *[str(n + offset) for offset in range(8)]])
            with contextlib.redirect_stdout(io.StringIO()):
                code = main(["split", str(source), "--target", "score", "--group-column", "employee_id", "--output", str(destination)])
            self.assertEqual(code, 0)
            manifest = json.loads((destination / "manifest.json").read_text())
            self.assertEqual(sum(info["rows"] for info in manifest["outputs"].values()), 1000)

    def test_employee_groups_do_not_cross_splits(self):
        result = split_dataset(fixture(), group_column="employee_id")
        sets = [{row[0] for row in part} for part in result.partitions.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(map(len, result.partitions.values())), 60)
        self.assertEqual(result.group_counts, {"train": 21, "validation": 4, "test": 5})

    def test_seed_is_reproducible_without_mutating_global_random(self):
        state = random.getstate()
        first = split_dataset(fixture(), seed=9, group_column="employee_id")
        self.assertEqual(first, split_dataset(fixture(), seed=9, group_column="employee_id"))
        self.assertNotEqual(first.partitions, split_dataset(fixture(), seed=10, group_column="employee_id").partitions)
        self.assertEqual(state, random.getstate())

    def test_exact_duplicates_stay_together_without_group_column(self):
        data = fixture()
        duplicated = Dataset(data.columns, data.rows + data.rows, data.source, data.sha256)
        result = split_dataset(duplicated)
        sets = [set(part) for part in result.partitions.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(map(len, result.partitions.values())), 120)

    def test_bad_ratios_missing_groups_and_few_groups_rejected(self):
        for train, validation in [(1, 0.1), (0.8, 0.3), (float("nan"), 0.2)]:
            with self.assertRaises(DatasetError):
                split_dataset(fixture(), train_fraction=train, validation_fraction=validation)
        with self.assertRaises(DatasetError):
            split_dataset(fixture(), group_column="missing")
        data = Dataset(("id",), (("1",), ("2",)), "fixture", "0" * 64)
        with self.assertRaises(DatasetError):
            split_dataset(data)
        with self.assertRaises(DatasetError):
            split_dataset(Dataset(("id",), (("1",), ("2",), ("",)), "fixture", "0" * 64), group_column="id")

    def test_exports_retain_all_rows_and_verified_fingerprints(self):
        data = fixture()
        result = split_dataset(data, group_column="employee_id")
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "splits"
            manifest = write_splits(data, result, out, target="score")
            self.assertEqual(json.loads((out / "manifest.json").read_text()), manifest)
            self.assertEqual(manifest["non_feature_columns"], ["score", "employee_id"])
            for name, info in manifest["outputs"].items():
                path = out / f"{name}.csv"
                with path.open(newline="") as stream:
                    rows = list(csv.reader(stream))
                self.assertEqual(tuple(rows[0]), data.columns)
                self.assertEqual(len(rows) - 1, info["rows"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), info["sha256"])
            with self.assertRaises(DatasetError):
                write_splits(data, result, out)
