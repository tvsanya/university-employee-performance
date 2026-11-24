import hashlib,json,os,tempfile,threading,unittest,uuid
from datetime import datetime,timezone
from pathlib import Path
from unittest.mock import patch
from university_analytics.jobs import TrainingManager,write_state,write_json
from university_analytics.job_lock import TrainingLease


class JobRecoveryTests(unittest.TestCase):
    def state(self,manager):
        key=uuid.uuid4().hex;directory=manager.root/key;directory.mkdir()
        state={'id':key,'owner':hashlib.sha256(b'owner').hexdigest(),'runner_id':'old-runner',
               'pid':os.getpid(),'state':'running','stage':'Old stage','progress':45,
               'created_at':datetime.now(timezone.utc).isoformat(),'epochs':3}
        write_state(directory,state)
        return key,directory,state

    def test_lost_lease_recovers_even_if_pid_is_current_process(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=TrainingManager(Path(folder));self.addCleanup(manager.close)
            key,directory,state=self.state(manager)
            with patch('university_analytics.jobs.os.kill',side_effect=AssertionError('PID must not determine ownership')):
                result=manager.get(key,'owner')
            self.assertEqual(result['state'],'failed')
            self.assertNotIn('runner_id',result)
            self.assertEqual(json.loads((directory/'status.json').read_text())['state'],'failed')
            self.assertEqual(manager.get(key,'owner')['state'],'failed')

    def test_live_other_manager_is_not_failed_and_new_owner_does_not_revive_old_job(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=TrainingManager(Path(folder));self.addCleanup(manager.close)
            key,directory,state=self.state(manager)
            lease=TrainingLease(manager.lease.path)
            self.assertTrue(lease.acquire())
            try:
                write_json(manager.marker,{'job_id':key,'runner_id':'old-runner'})
                self.assertEqual(manager.get(key,'owner')['state'],'running')
                write_json(manager.marker,{'job_id':uuid.uuid4().hex,'runner_id':'new-runner'})
                self.assertEqual(manager.get(key,'owner')['state'],'failed')
            finally:lease.release()

    def test_concurrent_atomic_status_writers_do_not_share_temporary_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);barrier=threading.Barrier(2);errors=[]
            def write(number):
                try:
                    barrier.wait(timeout=5)
                    for count in range(20):write_state(root,{'writer':number,'count':count,'payload':'x'*1000})
                except Exception as exc:errors.append(exc)
            threads=[threading.Thread(target=write,args=(number,)) for number in range(2)]
            for thread in threads:thread.start()
            for thread in threads:thread.join(timeout=10)
            self.assertFalse(any(thread.is_alive() for thread in threads));self.assertEqual(errors,[])
            self.assertEqual(json.loads((root/'status.json').read_text())['count'],19)
            self.assertFalse(list(root.glob('*.tmp')))
