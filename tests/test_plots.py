import json
import tempfile
import unittest
from pathlib import Path
from university_analytics.data import DatasetError
from university_analytics.plots import read_plot_inputs


class PlotInputTests(unittest.TestCase):
    def test_modified_history_is_rejected_before_plotting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/"metrics.json").write_text(json.dumps({"training_artifact": {"file": "training.json", "sha256": "0"*64}}))
            (root/"training.json").write_text("{}")
            (root/"diagnostics.json").write_text("{}")
            with self.assertRaises(DatasetError):
                read_plot_inputs(root, root)
