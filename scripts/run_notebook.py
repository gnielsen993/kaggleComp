"""Execute the report in a fresh kernel; save outputs only after every cell passes."""

import os
from pathlib import Path
import sys
import tempfile

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter


def main():
    root = Path(__file__).resolve().parents[1]
    # Resolve the fresh kernel to the same virtual environment as this script.
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"]
    with tempfile.TemporaryDirectory(prefix="titanic-jupyter-") as temporary:
        for key in ["JUPYTER_CONFIG_DIR", "JUPYTER_RUNTIME_DIR", "IPYTHONDIR", "MPLCONFIGDIR"]:
            os.environ[key] = str(Path(temporary) / key.lower())
        path = root / "titanic.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        client = NotebookClient(
            notebook, timeout=600, kernel_name="python3",
            resources={"metadata": {"path": str(root)}},
        )
        client.execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        output = root / "outputs"
        output.mkdir(exist_ok=True)
        html, _ = HTMLExporter(embed_images=True).from_notebook_node(
            notebook, resources={"metadata": {"path": str(root)}}
        )
        (output / "titanic.html").write_text(html)
        count = sum(cell.cell_type == "code" for cell in notebook.cells)
        print(f"Executed {count} code cells successfully; rendered outputs/titanic.html")


if __name__ == "__main__":
    main()
