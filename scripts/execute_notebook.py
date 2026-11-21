"""Выполнить ноутбук проектным Python, не меняя глобальные ядра Jupyter."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
runtime = ROOT/".local/notebook-runtime/share/jupyter"
kernel = runtime/"kernels/university-project"
kernel.mkdir(parents=True, exist_ok=True)
(kernel/"kernel.json").write_text(json.dumps({"argv": [sys.executable, "-Xfrozen_modules=off", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
    "display_name": "University project", "language": "python"}), encoding="utf-8")
os.environ["JUPYTER_PATH"] = str(runtime) + (os.pathsep+os.environ["JUPYTER_PATH"] if os.environ.get("JUPYTER_PATH") else "")
os.environ["IPYTHONDIR"] = str(ROOT/".local/ipython")
os.environ["JUPYTER_RUNTIME_DIR"] = str(ROOT/".local/jupyter-runtime")
from nbclient import NotebookClient
import nbformat

path = ROOT/"notebooks/research.ipynb"
notebook = nbformat.read(path, as_version=4)
NotebookClient(notebook, timeout=120, kernel_name="university-project", resources={"metadata": {"path": str(ROOT)}}).execute()
nbformat.validate(notebook)
cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
assert all(cell.execution_count is not None for cell in cells)
assert not any(output.output_type == "error" for cell in cells for output in cell.outputs)
temporary = ROOT/".local/executed-notebook.ipynb"
nbformat.write(notebook, temporary)
temporary.replace(path)
print(f"Executed notebook successfully: {len(cells)} code cells")
