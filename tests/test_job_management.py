import hashlib,json,tempfile,unittest,uuid
from datetime import datetime,timezone,timedelta
from pathlib import Path
from university_analytics.gui import create_app
from university_analytics.jobs import TrainingManager,JobNotFound,TrainingBusy,write_state,write_json


class JobManagementTests(unittest.TestCase):
    def state(self,manager,owner,offset=0,status='failed'):
        key=uuid.uuid4().hex;directory=manager.root/key;directory.mkdir()
        write_state(directory,{'runner_id':manager.runner_id,'id':key,'owner':hashlib.sha256(owner.encode()).hexdigest(),
            'state':status,'stage':'Fixture','progress':0,'created_at':(datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(seconds=offset)).isoformat()})
        (directory/'input.csv').write_text('x\n1\n')
        return key,directory

    def test_owner_cursor_exposes_older_records_without_foreign_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=TrainingManager(Path(folder));self.addCleanup(manager.close)
            own=[self.state(manager,'owner',i)[0] for i in range(21)]
            foreign,_=self.state(manager,'stranger',100)
            first=manager.list_page('owner');self.assertEqual(len(first['jobs']),20);self.assertEqual(first['total'],21)
            second=manager.list_page('owner',first['next']);self.assertEqual([row['id'] for row in second['jobs']],[own[0]])
            self.assertIsNone(second['next'])
            with self.assertRaises(JobNotFound):manager.list_page('owner',foreign)

    def test_terminal_deletion_is_private_and_active_result_is_protected(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=TrainingManager(Path(folder));self.addCleanup(manager.close)
            key,directory=self.state(manager,'owner')
            with self.assertRaises(JobNotFound):manager.delete(key,'stranger')
            self.assertTrue(directory.exists())
            active,_=self.state(manager,'owner',status='running')
            self.assertTrue(manager.lease.acquire())
            write_json(manager.marker,{'job_id':active,'runner_id':manager.runner_id})
            try:
                with self.assertRaises(TrainingBusy):manager.delete(active,'owner')
            finally:manager.lease.release();manager.marker.unlink(missing_ok=True)
            self.assertEqual(manager.delete(key,'owner'),{'deleted':True,'cleanup_pending':False})
            self.assertFalse(directory.exists())
            with self.assertRaises(JobNotFound):manager.get(key,'owner')

    def test_http_requires_csrf_and_explicit_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            app=create_app(Path(folder));app.config['TESTING']=True
            manager=app.extensions['training_manager'];self.addCleanup(manager.close)
            client=app.test_client();client.get('/')
            with client.session_transaction() as state:token=state['csrf'];owner=state['workspace_id']
            key,directory=self.state(manager,owner)
            url='/api/jobs/'+key+'/delete'
            self.assertEqual(client.post(url,json={'confirm':key}).status_code,403)
            self.assertEqual(client.post(url,json={'confirm':'wrong'},headers={'X-CSRF-Token':token}).status_code,400)
            self.assertTrue(directory.exists())
            self.assertEqual(client.post(url,json={'confirm':key},headers={'X-CSRF-Token':token}).status_code,200)
            self.assertFalse(directory.exists())

    def test_cleanup_failure_is_reported_and_result_is_no_longer_public(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as folder:
            manager=TrainingManager(Path(folder));self.addCleanup(manager.close)
            key,_=self.state(manager,'owner')
            with patch('university_analytics.jobs.shutil.rmtree',side_effect=OSError('busy')):
                result=manager.delete(key,'owner')
            self.assertTrue(result['cleanup_pending'])
            with self.assertRaises(JobNotFound):manager.get(key,'owner')
            self.assertEqual(len(list((manager.root.parent/'gui-trash').iterdir())),1)
