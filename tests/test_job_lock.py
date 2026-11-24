import json,subprocess,sys,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
from university_analytics.data import DatasetError
from university_analytics.job_lock import TrainingLease
from university_analytics.jobs import TrainingBusy,TrainingManager
from university_analytics.simulation import generate_demo


class JobLockTests(unittest.TestCase):
    def test_lock_excludes_another_process_and_is_released_after_death(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'training.lock'
            code="from university_analytics.job_lock import TrainingLease;import sys;lock=TrainingLease(sys.argv[1]);assert lock.acquire();print('locked',flush=True);sys.stdin.read()"
            process=subprocess.Popen([sys.executable,'-c',code,str(path)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            lease=TrainingLease(path)
            try:
                self.assertEqual(process.stdout.readline().strip(),'locked')
                self.assertFalse(lease.acquire())
                process.terminate();process.wait(timeout=5)
                self.assertTrue(lease.acquire())
            finally:
                lease.release()
                if process.poll() is None:process.terminate();process.wait(timeout=5)
                for stream in [process.stdin,process.stdout,process.stderr]:stream.close()

    def test_two_managers_share_one_training_slot(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);generate_demo(root/'data',employees=500,periods=2)
            source=root/'data/synthetic_employees.csv';config=json.loads((root/'data/experiment.json').read_text())
            first,second=TrainingManager(root),TrainingManager(root)
            entered,released=threading.Event(),threading.Event()
            def worker(*args):entered.set();released.wait(10)
            try:
                with patch('university_analytics.jobs.run_job',side_effect=worker):
                    first.submit(source,config,'one',2);self.assertTrue(entered.wait(5))
                    with self.assertRaises(TrainingBusy):second.submit(source,config,'two',2)
                    self.assertEqual(len(list(first.root.iterdir())),1)
                    released.set();first.active.result(timeout=5)
                    second.submit(source,config,'two',2);second.active.result(timeout=5)
            finally:
                released.set();first.close();second.close()

    def test_validation_failure_releases_slot_without_creating_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'bad.csv';source.write_text('x,y\n1,2\n')
            manager=TrainingManager(root)
            try:
                with self.assertRaises(DatasetError):manager.submit(source,{},'owner',2)
                probe=TrainingLease(manager.lease.path)
                try:self.assertTrue(probe.acquire())
                finally:probe.release()
                self.assertEqual(list(manager.root.iterdir()),[])
            finally:manager.close()
