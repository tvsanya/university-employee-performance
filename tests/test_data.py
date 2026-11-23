import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipFile
from university_analytics.data import DatasetError, basic_profile, load_dataset, read_source


class DatasetLoadingTests(unittest.TestCase):
    """Маленькие фикстуры проверяют парсер, а не обучают модель диплома."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "sample.csv"

    def write(self, text):
        self.path.write_text(text, encoding="utf-8")
        return str(self.path)

    def test_utf8_bom_and_quoted_values(self):
        dataset = load_dataset(self.write('\ufeffid,note\n1,"текст, с запятой"\n'))
        self.assertEqual(dataset.columns, ("id", "note"))
        self.assertEqual(dataset.rows[0][1], "текст, с запятой")
        self.assertEqual(basic_profile(dataset)["row_count"], 1)
        self.assertEqual(len(dataset.sha256), 64)

    def test_bad_headers_and_ragged_rows_rejected(self):
        for text in ["id,id\n1,2\n", "id,\n1,2\n", "id,score\n1\n", ""]:
            with self.subTest(text=text), self.assertRaises(DatasetError):
                load_dataset(self.write(text))

    def test_explicit_zip_member_is_read_without_extraction(self):
        archive = Path(self.temp.name) / "data.zip"
        with ZipFile(archive, "w") as zipped:
            zipped.writestr("nested/data.csv", "id,score\n1,5\n")
        self.assertEqual(load_dataset(str(archive), "nested/data.csv").rows[0], ("1", "5"))
        with self.assertRaises(DatasetError):
            load_dataset(str(archive), "missing.csv")

    def test_remote_source_is_downloaded(self):
        response = io.BytesIO(b"id,score\n1,5\n")
        response.geturl = lambda: "https://example.org/data.csv"
        with patch("university_analytics.data.urlopen", return_value=response) as download:
            dataset = load_dataset("https://example.org/data.csv")
            self.assertEqual(dataset.rows, (("1", "5"),))
            self.assertEqual(download.call_args.kwargs["timeout"], 30)

    def test_source_limits_and_credentials(self):
        with patch("university_analytics.data.MAX_BYTES", 4), self.assertRaises(DatasetError):
            read_source(self.write("12345"))
        for source in ["ftp://example.org/data.csv", "https://user:password@example.org/data.csv"]:
            with self.assertRaises(DatasetError):
                read_source(source)


    def test_row_column_cell_and_blank_record_limits(self):
        for setting, limit, text in [
            ('MAX_ROWS', 2, 'id\n1\n2\n3\n'),
            ('MAX_COLUMNS', 2, 'a,b,c\n1,2,3\n'),
            ('MAX_CELLS', 3, 'a,b\n1,2\n3,4\n'),
            ('MAX_RECORDS', 3, 'a\n\n\n\n')]:
            with self.subTest(setting=setting), patch('university_analytics.data.'+setting, limit):
                with self.assertRaises(DatasetError):load_dataset(self.write(text), delimiter=',')
        with patch('university_analytics.data.MAX_ROWS', 2):
            self.assertEqual(len(load_dataset(self.write('id\n1\n2\n')).rows), 2)

    def test_bounded_multiline_and_unterminated_record(self):
        dataset=load_dataset(self.write('a,b\r\n1,"line1\r\nline2"\r\n'), delimiter=',')
        self.assertEqual(dataset.rows[0], ('1','line1\r\nline2'))
        for text in ['a,b\n1,"xxxxxxxxxxxxxxxx"\n', 'a,b\n1,"a\nb\nc\nd\ne\nf\ng"\n']:
            with patch('university_analytics.data.MAX_RECORD_CHARS', 12), self.assertRaises(DatasetError):
                load_dataset(self.write(text),delimiter=',')
        with self.assertRaises(DatasetError):load_dataset(self.write('a,b\n1,"unclosed\n'),delimiter=',')
