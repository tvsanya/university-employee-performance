import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from university_analytics.data import DatasetError
from university_analytics.research_report import run_research_report


class ResearchPackageTests(unittest.TestCase):
    def test_partial_report_is_explicit_and_existing_output_not_replaced(self):
        provenance={"partition":"validation","rows":2,"source_sha256":"a"*64,"split_manifest_sha256":"b"*64,"model_sha256":"c"*64,"data_origin":"synthetic","intended_use":"development_demo_only"}
        with tempfile.TemporaryDirectory() as folder, ExitStack() as stack:
            stack.enter_context(patch("university_analytics.research_report.validation_context", return_value=(SimpleNamespace(rows=[1,2]),None,None,provenance)))
            for name in ["run_distribution_shift","run_importance","run_probability_analysis","run_robustness","run_uncertainty"]:
                stack.enter_context(patch("university_analytics.research_report."+name, side_effect=DatasetError("Недостаточно групп <script>")))
            target=Path(folder)/"result"
            report=run_research_report("splits","model",target)
            self.assertEqual(report["status"],"partial")
            self.assertNotIn("<script>",(target/"research.md").read_text())
            self.assertTrue(all(item["status"]=="unavailable" for item in report["analyses"].values()))
            with self.assertRaises(DatasetError): run_research_report("splits","model",target)

    def test_changed_provenance_aborts_whole_package(self):
        with tempfile.TemporaryDirectory() as folder:
            target=Path(folder)/"result"
            with patch("university_analytics.research_report.validation_context",return_value=(None,None,None,{"model_sha256":"a"})), patch("university_analytics.research_report.run_distribution_shift",return_value={"model_sha256":"b"}):
                with self.assertRaises(DatasetError): run_research_report("splits","model",target)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(folder).iterdir()),[])
