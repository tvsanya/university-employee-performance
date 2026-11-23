import io
import json
import tempfile
import threading
import unittest
from zipfile import ZipFile
from pathlib import Path
from unittest.mock import patch
from university_analytics.data import DatasetError
from university_analytics.gui import create_app
from university_analytics.jobs import TrainingManager, TrainingBusy
from university_analytics.simulation import generate_demo


class TrainingJobTests(unittest.TestCase):
    def inputs(self, root):
        generate_demo(root/"data", employees=500, periods=2)
        return root/"data/synthetic_employees.csv", json.loads((root/"data/experiment.json").read_text())

    def test_gui_training_completes_and_result_is_private_to_session(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);source, config = self.inputs(root)
            app = create_app(root);app.config["TESTING"] = True
            manager = app.extensions["training_manager"]
            self.addCleanup(manager.close)
            client = app.test_client();client.get("/")
            with client.session_transaction() as state:
                token = state["csrf"]
            response = client.post("/api/train", data={"data": (io.BytesIO(source.read_bytes()), "input.csv"),
                "config": (io.BytesIO(json.dumps(config).encode()), "config.json"), "epochs": "3"}, headers={"X-CSRF-Token": token})
            self.assertEqual(response.status_code, 202, response.json)
            job_id = response.json["id"]
            manager.active.result(timeout=30)
            status = client.get("/api/jobs/"+job_id)
            if status.json["state"] == "failed":
                self.fail((manager.root/job_id/"error.log").read_text())
            self.assertEqual(status.json["state"], "completed")
            self.assertEqual(status.json["result"]["intended_use"], "development_demo_only")
            self.assertTrue((manager.root/job_id/"comparison/comparison.json").exists())
            artifacts = client.get("/api/jobs/"+job_id+"/artifacts")
            self.assertEqual(artifacts.status_code, 200)
            self.assertEqual(status.json["result"]["analysis_status"]["uncertainty"],"completed")
            analysis = client.get("/api/jobs/"+job_id+"/download/importance.json")
            self.assertEqual(analysis.json["partition"],"validation")
            self.assertEqual(analysis.json["status"],"completed")
            analysis.close()
            markdown = client.get("/api/jobs/"+job_id+"/download/research.md")
            self.assertIn("Диагностика сохранённой модели", markdown.data.decode());markdown.close()
            manifest = client.get("/api/jobs/"+job_id+"/download/research-manifest.json")
            self.assertEqual(manifest.json["analysis_source"], "existing_snapshot")
            self.assertEqual(manifest.json["model_sha256"], analysis.json["model_sha256"])
            manifest.close()
            model = client.get("/api/jobs/"+job_id+"/download/model.json")
            self.assertIn("config", json.loads(model.data));model.close()
            archive = client.get("/api/jobs/"+job_id+"/download/experiment.zip")
            with ZipFile(io.BytesIO(archive.data)) as bundle:
                self.assertIn("input.csv", bundle.namelist())
                self.assertNotIn("status.json", bundle.namelist())
                self.assertIn("SHA256SUMS.json", bundle.namelist())
                self.assertIn("analysis/uncertainty/analysis.json", bundle.namelist())
                self.assertIn("research/research.md", bundle.namelist())
                self.assertEqual(bundle.read("analysis/uncertainty/analysis.json"), bundle.read("research/uncertainty/analysis.json"))
            archive.close()
            self.assertEqual(client.get("/api/jobs/"+job_id+"/download/status.json").status_code, 404)
            self.assertEqual(len(client.get("/api/jobs").json["jobs"]), 1)
            prediction = client.post("/api/predict", data={"data": (io.BytesIO(source.read_bytes()), "predict.csv"),
                "experiment": "job:"+job_id}, headers={"X-CSRF-Token": token})
            self.assertEqual(prediction.status_code, 200)
            self.assertEqual(prediction.json["report"]["rows"], 1000)
            other = app.test_client();other.get("/")
            self.assertEqual(other.get("/api/jobs/"+job_id).status_code, 404)
            self.assertEqual(other.get("/api/jobs/"+job_id+"/download/model.json").status_code, 404)
            self.assertEqual(other.get("/api/jobs").json["jobs"], [])
            self.assertNotIn("owner", status.json)
            manager.close()

    def test_second_active_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);source, config = self.inputs(root)
            manager = TrainingManager(root)
            released = threading.Event()
            try:
                with patch("university_analytics.jobs.run_job", side_effect=lambda *args: released.wait(10)):
                    manager.submit(source, config, "owner", 2)
                    with self.assertRaises(TrainingBusy):
                        manager.submit(source, config, "owner", 2)
            finally:
                released.set();manager.close()

    def test_invalid_dataset_never_creates_a_job(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);source, config = self.inputs(root)
            source.write_text("x,y\n1,2\n")
            manager = TrainingManager(root)
            try:
                with self.assertRaises(DatasetError):
                    manager.submit(source, config, "owner", 2)
                self.assertEqual(list(manager.root.iterdir()), [])
            finally:
                manager.close()
