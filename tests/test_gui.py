import csv
import io
import unittest
from pathlib import Path
from university_analytics.gui import create_app


class GuiTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parent.parent
        self.app = create_app(self.root)
        self.app.config["TESTING"] = True
        self.client = self.app.test_client()
        self.client.get("/")
        with self.client.session_transaction() as state:
            self.token = state["csrf"]

    def test_profile_upload_and_frontend_assets(self):
        response = self.client.post("/api/inspect", data={"data": (io.BytesIO(b"x,y\n1,a\n2,b\n"), "input.csv")}, headers={"X-CSRF-Token": self.token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["row_count"], 2)
        self.assertNotIn("source", response.json)
        for path in ["/static/app.js", "/api/figures/training.png"]:
            asset = self.client.get(path)
            try:
                self.assertEqual(asset.status_code, 200)
            finally:
                asset.close()

    def test_csrf_origin_and_host_are_restricted(self):
        self.assertEqual(self.client.post("/api/inspect").status_code, 403)
        self.assertEqual(self.client.post("/api/inspect", headers={"X-CSRF-Token": self.token, "Origin": "https://example.com"}).status_code, 403)
        self.assertEqual(self.client.get("/", headers={"Host": "example.com"}).status_code, 400)

    def test_saved_model_predicts_without_target_and_keeps_synthetic_label(self):
        with (self.root/"data/simulation-v1/synthetic_employees.csv").open(encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            row = next(reader)
            columns = [name for name in reader.fieldnames if name not in {"efficiency_level", "efficiency_score"}]
        row["employee_id"] = "=1+1"
        output = io.StringIO();writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader();writer.writerow(row)
        response = self.client.post("/api/predict", data={"data": (io.BytesIO(output.getvalue().encode()), "data.csv"),
            "experiment": "simulation-neural-v1"}, headers={"X-CSRF-Token": self.token})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["report"]["intended_use"], "development_demo_only")
        self.assertEqual(response.json["preview"][0]["employee_id"], "'=1+1")
        self.assertEqual(response.json["report"]["rows"], 1)

    def test_bad_inputs_are_reported_without_filesystem_details(self):
        response = self.client.post("/api/predict", data={"data": (io.BytesIO(b"x,y\n1,2\n"), "data.csv"),
            "experiment": "../private"}, headers={"X-CSRF-Token": self.token})
        self.assertEqual(response.status_code, 400)
        self.assertIn("error", response.json)
        self.assertNotIn(str(self.root), response.json["error"])
        malformed = self.client.post("/api/train", data={"data": (io.BytesIO(b"x,y\n1,2\n"), "data.csv"),
            "config": (io.BytesIO(b"["*2000+b"]"*2000), "deep.json")}, headers={"X-CSRF-Token": self.token})
        self.assertEqual(malformed.status_code, 400)

    def test_prediction_row_limit_prevents_unbounded_response(self):
        contents = b"x,y\n"+b"1,2\n"*20001
        response = self.client.post("/api/predict", data={"data": (io.BytesIO(contents), "large.csv"),
            "experiment": "simulation-neural-v1"}, headers={"X-CSRF-Token": self.token})
        self.assertEqual(response.status_code, 400)
        self.assertIn("20000", response.json["error"])
