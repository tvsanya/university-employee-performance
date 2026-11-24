import copy
import hashlib
import json
import tempfile
import unittest
import uuid
from datetime import datetime,timezone
from pathlib import Path
from university_analytics.jobs import JobNotFound,TrainingManager,write_state


class JobStateTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.manager=TrainingManager(Path(self.temp.name));self.addCleanup(self.manager.close)
        self.key=uuid.uuid4().hex;self.directory=self.manager.root/self.key;self.directory.mkdir()
        self.state={'id':self.key,'owner':hashlib.sha256(b'owner').hexdigest(),'state':'failed','stage':'Fixture',
            'progress':0,'created_at':datetime.now(timezone.utc).isoformat()}

    def test_invalid_metadata_never_breaks_history_or_modifies_files(self):
        cases=[{},[],{'state':[]},{'progress':True},{'progress':-1},{'progress':101},{'created_at':None},
            {'created_at':'2026-01-01'}, {'stage':[]}, {'state':'completed','result':{}},{'owner':'я'*64}]
        for change in cases:
            state=change if isinstance(change,list) or not change else {**self.state,**change}
            original=json.dumps(state).encode();(self.directory/'status.json').write_bytes(original)
            with self.subTest(change=change):
                with self.assertRaises(JobNotFound):self.manager.get(self.key,'owner')
                self.assertEqual(self.manager.list_page('owner')['total'],0)
                self.assertEqual((self.directory/'status.json').read_bytes(),original)
        for payload in [b'{bad',b'\xff',b' '*65537]:
            (self.directory/'status.json').write_bytes(payload)
            with self.assertRaises(JobNotFound):self.manager.get(self.key,'owner')

    def test_unknown_fields_are_not_public_and_offset_order_uses_utc(self):
        write_state(self.directory,{**self.state,'secret_note':'private','created_at':'2026-01-01T12:00:00+04:00'})
        result=self.manager.get(self.key,'owner')
        self.assertNotIn('secret_note',result);self.assertNotIn('owner',result)
        self.assertEqual(result['created_at'],'2026-01-01T08:00:00+00:00')
        nextkey=uuid.uuid4().hex;other=self.manager.root/nextkey;other.mkdir()
        write_state(other,{**self.state,'id':nextkey,'created_at':'2026-01-01T10:00:00+00:00'})
        self.assertEqual([row['id'] for row in self.manager.list_page('owner')['jobs']],[nextkey,self.key])

    def test_state_symlinks_and_nonfinite_metrics_are_not_accepted(self):
        outside=Path(self.temp.name)/'outside.json';write_state(outside.parent,self.state)
        outside=(Path(self.temp.name)/'status.json')
        (self.directory/'status.json').symlink_to(outside)
        with self.assertRaises(JobNotFound):self.manager.get(self.key,'owner')
        (self.directory/'status.json').unlink()
        invalid={**self.state,'state':'completed','result':{'selected_family':'neural','selected_model':'neural','data_origin':'synthetic','metrics':{'train':{'accuracy':float('nan'),'macro_f1':1}}}}
        (self.directory/'status.json').write_text(json.dumps(invalid))
        with self.assertRaises(JobNotFound):self.manager.get(self.key,'owner')
