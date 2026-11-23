from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import threading
import unittest
from unittest.mock import patch
from pathlib import Path
from university_analytics.gui import create_app


class UrlGuiTests(unittest.TestCase):
    def setUp(self):
        self.app=create_app(Path(__file__).resolve().parents[1]);self.app.config['TESTING']=True
        self.client=self.app.test_client();self.client.get('/')
        with self.client.session_transaction() as state:self.token=state['csrf']
        self.headers={'X-CSRF-Token':self.token}
        class Handler(BaseHTTPRequestHandler):
            def do_GET(inner):
                payload=b'x,y\n1,a\n2,b\n'
                inner.send_response(200);inner.send_header('Content-Type','text/csv');inner.send_header('Content-Length',str(len(payload)));inner.end_headers();inner.wfile.write(payload)
            def log_message(inner,*args):pass
        self.server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.url=f'http://127.0.0.1:{self.server.server_port}/demo.csv'

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=2)
        self.app.extensions['training_manager'].executor.shutdown(wait=True)

    def test_actual_http_csv_profile_and_training_receives_download(self):
        result=self.client.post('/api/inspect',data={'source_url':self.url},headers=self.headers)
        self.assertEqual(result.status_code,200);self.assertEqual(result.json['row_count'],2)
        self.assertNotIn('source',result.json)
        manager=self.app.extensions['training_manager'];contents=[]
        def submit(source,*args):contents.append(source.read_bytes());return {'id':'fake-transport-only'}
        with patch.object(manager,'submit',side_effect=submit):
            result=self.client.post('/api/train',data={'source_url':self.url,'epochs':'1'},headers=self.headers)
        self.assertEqual(result.status_code,202);self.assertEqual(contents,[b'x,y\n1,a\n2,b\n'])

    def test_ambiguous_and_non_http_sources_are_rejected(self):
        result=self.client.post('/api/inspect',data={'source_url':self.url,'data':(io.BytesIO(b'x\n1\n'),'data.csv')},headers=self.headers)
        self.assertEqual(result.status_code,400)
        for address in ['http://[broken','https://example.org/a\nb','file:///etc/passwd','/tmp/input.csv','https://u:p@example.org/a.csv','http://localhost:bad/a','https://example.org/a#secret']:
            result=self.client.post('/api/inspect',data={'source_url':address},headers=self.headers)
            self.assertEqual(result.status_code,400,address)
        self.assertEqual(self.client.post('/api/inspect',data={'source_url':self.url}).status_code,403)
