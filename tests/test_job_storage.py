import hashlib,json,tempfile,unittest,uuid
from pathlib import Path
from unittest.mock import patch
from university_analytics.gui import create_app
from university_analytics.jobs import TrainingManager,write_state
from university_analytics.data import DatasetError
from university_analytics.job_storage import storage_report,purge_own_trash,tree_bytes


class JobStorageTests(unittest.TestCase):
    def seed(self,root,owner,trash=False,status='failed'):
        key=uuid.uuid4().hex
        directory=root/(key+'-'+uuid.uuid4().hex if trash else key);directory.mkdir(parents=True)
        write_state(directory,{'id':key,'owner':hashlib.sha256(owner.encode()).hexdigest(),'state':status,
            'stage':'Fixture','progress':0,'created_at':'2026-01-01T00:00:00+00:00'})
        (directory/'input.csv').write_bytes(b'x'*100)
        return directory

    def test_size_and_explicit_purge_are_owned_and_do_not_touch_live_results(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs=Path(folder)/'gui-jobs';jobs.mkdir();trash=jobs.parent/'gui-trash'
            kept=self.seed(jobs,'owner');own=self.seed(trash,'owner',True);foreign=self.seed(trash,'foreign',True)
            active=self.seed(trash,'owner',True,'running')
            damaged=self.seed(trash,'owner',True);(damaged/'status.json').write_text('{bad')
            outside=Path(folder)/'outside';outside.write_bytes(b'x'*9999)
            (kept/'link').symlink_to(outside)
            expected=sum(p.stat().st_size for p in kept.iterdir() if not p.is_symlink())
            report=storage_report(jobs,'owner')
            self.assertEqual((report['jobs'],report['job_bytes'],report['trash']),(1,expected,1))
            self.assertEqual(purge_own_trash(jobs,'owner'),{'deleted_directories':1,'failed_directories':0})
            self.assertFalse(own.exists())
            for path in [kept,foreign,active,damaged,outside]:self.assertTrue(path.exists())
            self.assertEqual(purge_own_trash(jobs,'owner')['deleted_directories'],0)

    def test_limits_and_failed_cleanup_do_not_claim_success(self):
        with tempfile.TemporaryDirectory() as folder:
            jobs=Path(folder)/'gui-jobs';jobs.mkdir();own=self.seed(jobs.parent/'gui-trash','owner',True)
            with patch('university_analytics.job_storage.shutil.rmtree',side_effect=PermissionError):
                self.assertEqual(purge_own_trash(jobs,'owner')['failed_directories'],1)
            self.assertTrue(own.exists())
            with patch('university_analytics.job_storage.MAX_ENTRIES',0):
                with self.assertRaises(DatasetError):storage_report(jobs,'owner')

    def test_http_csrf_confirmation_and_browser_isolation(self):
        with tempfile.TemporaryDirectory() as folder:
            app=create_app(Path(folder));app.config['TESTING']=True
            manager=app.extensions['training_manager'];self.addCleanup(manager.close)
            client=app.test_client();client.get('/')
            with client.session_transaction() as state:owner=state['workspace_id'];token=state['csrf']
            own=self.seed(manager.root.parent/'gui-trash',owner,True)
            self.assertEqual(client.get('/api/storage').json['trash'],1)
            stranger=app.test_client();stranger.get('/')
            self.assertEqual(stranger.get('/api/storage').json['trash'],0)
            path='/api/storage/purge-trash'
            self.assertEqual(client.post(path,json={'confirm':'purge-own-trash'}).status_code,403)
            self.assertEqual(client.post(path,json={'confirm':'wrong'},headers={'X-CSRF-Token':token}).status_code,400)
            self.assertTrue(own.exists())
            self.assertEqual(client.post(path,json={'confirm':'purge-own-trash'},headers={'X-CSRF-Token':token}).json['deleted_directories'],1)
            self.assertFalse(own.exists())
