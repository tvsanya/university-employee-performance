import tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from university_analytics.job_artifacts import artifact_json,checked_file,experiment_archive,JobNotReady


class ArtifactDamageTests(unittest.TestCase):
    def test_ambiguous_and_nonfinite_json_fail_without_deleting_source(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'result.json'
            for raw in ['{bad','{"x":1,"x":2}','{"x":1e999}','{"x":NaN}','[]']:
                path.write_text(raw)
                with self.assertRaises(JobNotReady):artifact_json(root,path)
                self.assertEqual(path.read_text(),raw)

    def test_archive_refuses_missing_source_and_internal_symlink(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);job=root/'job';job.mkdir()
            manager=SimpleNamespace(repository=root,get=lambda *args:{'state':'completed','result':{'data_origin':'synthetic','intended_use':'development_demo_only'}},directory=lambda *args:job)
            with self.assertRaises(JobNotReady):experiment_archive(manager,'id','owner')
            (job/'input.csv').write_text('x\n1');(job/'config.json').write_text('{}')
            (job/'comparison').mkdir();(job/'comparison/comparison.json').write_text('{}')
            (job/'status.json').write_text('private-owner')
            (job/'comparison/leak.txt').symlink_to(job/'status.json')
            with self.assertRaises(JobNotReady):experiment_archive(manager,'id','owner')
            self.assertTrue((job/'status.json').exists())

    def test_file_and_total_size_limits_refuse_complete_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);path=root/'result.json';path.write_text('{}')
            with patch('university_analytics.job_artifacts.MAX_ARTIFACT',1):
                with self.assertRaises(JobNotReady):checked_file(root,path)

    def test_direct_download_refuses_internal_alias_to_session_metadata(self):
        from university_analytics.job_artifacts import artifact_path
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'neural').mkdir();(root/'status.json').write_text('private-owner')
            (root/'neural/model.json').symlink_to(root/'status.json')
            manager=SimpleNamespace(get=lambda *args:{'state':'completed','result':{'selected_family':'neural'}},directory=lambda *args:root)
            with self.assertRaises(JobNotReady):artifact_path(manager,'id','owner','model.json')
