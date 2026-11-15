"""Verify the final notebook without repo helpers/assets and package its inputs."""

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

import nbformat
from bs4 import BeautifulSoup


def definitions(source):
    """Compare duplicated portable definitions with the tested source modules."""
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef):
            result[node.name] = ast.dump(node, include_attributes=False)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "FEATURE_SETS":
                    result[target.id] = ast.dump(node, include_attributes=False)
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    notebook = nbformat.read(root / "titanic.ipynb", as_version=4)
    nbformat.validate(notebook)
    expected = {}
    for filename in ["titanic_features.py", "titanic_models.py", "titanic_inspection.py"]:
        expected.update(definitions((root / filename).read_text()))
    embedded = next(cell for cell in notebook.cells if cell.id == "final-helper-definitions")
    assert definitions(embedded.source) == expected, "Notebook helpers diverge from tested modules"
    for cell in notebook.cells:
        if cell.cell_type == "code":
            assert not re.search(r"^(?:from|import) titanic_", cell.source, re.M)
        else:
            for target in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", cell.source):
                assert target.startswith("attachment:"), target
                assert target.removeprefix("attachment:") in cell.get("attachments", {})

    with tempfile.TemporaryDirectory(prefix="titanic-final-") as temporary:
        stage = Path(temporary)
        inputs = ["titanic.ipynb", "train.csv", "test.csv", "requirements.txt"]
        for name in inputs:
            shutil.copy2(root / name, stage / name)
        # Reuse the normal execution path, but its root is an isolated directory.
        (stage / "scripts").mkdir()
        runner = stage / "scripts/run_notebook.py"
        shutil.copy2(root / "scripts/run_notebook.py", runner)
        env = os.environ.copy()
        env.pop("PYTHONPATH", None)
        subprocess.run([sys.executable, str(runner)], cwd=stage, env=env, check=True)

        executed = nbformat.read(stage / "titanic.ipynb", as_version=4)
        code = [cell for cell in executed.cells if cell.cell_type == "code"]
        assert [cell.execution_count for cell in code] == list(range(1, len(code) + 1))
        assert not any(output.output_type == "error" for cell in code for output in cell.outputs)
        for number, score in [("03", 0.76315), ("04", 0.77511), ("05", 0.77033), ("06", 0.77990)]:
            record = json.loads((stage / f"outputs/experiment_{number}.json").read_text())
            actual = hashlib.sha256((stage / record["submission_file"]).read_bytes()).hexdigest()
            assert record["kaggle_score"] == score
            assert actual == record["submission_sha256"] == record["recorded_submission"]["submission_sha256"]
        html = (stage / "outputs/titanic.html").read_text()
        assert "attachment:" not in html, "An embedded image/link did not render"
        rendered = BeautifulSoup(html, "html.parser")
        assert rendered.find("details").find("img") is not None, "Original evidence escaped its collapsed section"
        assert all(image.get("src", "").startswith("data:image/") for image in rendered.find_all("img"))
        assert "## 12. Consolidated results" in next(c.source for c in executed.cells if c.id == "final-results")

        output = root / "outputs"
        bundle = output / "final_submission"
        bundle.mkdir(parents=True, exist_ok=True)
        # Publish artifacts only after isolated execution and integrity checks pass.
        shutil.copy2(stage / "titanic.ipynb", root / "titanic.ipynb")
        for artifact in (stage / "outputs").iterdir():
            if artifact.is_file():
                shutil.copy2(artifact, output / artifact.name)
        for name in inputs:
            shutil.copy2(stage / name, bundle / name)
        shutil.copy2(stage / "outputs/titanic.html", bundle / "titanic.html")
        (bundle / "README.txt").write_text(
            "CS 4320 Kaggle final report — Gabriel Nielsen and Drew Marks\n\n"
            "Main deliverable: titanic.ipynb. Saved outputs and score images are embedded.\n"
            "titanic.html is an optional read-only preview.\n\n"
            "To reproduce, use Python 3.13 and install requirements.txt in a virtual environment.\n"
            "Open titanic.ipynb in Jupyter, select that environment, then Restart Kernel and Run All.\n"
            "Keep train.csv and test.csv in the working directory. No local Python helpers,\n"
            "external images, GPU, Kaggle credentials, or network calls are needed while running.\n"
            "The complete pinned environment was verified on macOS. Other platforms may need\n"
            "compatible package wheels. Execution generates prediction CSVs and outputs/ metadata.\n\n"
            "Six non-dummy uploads are documented; best observed Kaggle accuracy: 0.77990.\n"
            "The separate project-wide AI usage document is outside this Kaggle bundle.\n"
        )
        members = inputs + ["titanic.html", "README.txt"]
        package = output / "kaggle_final_submission.zip"
        with zipfile.ZipFile(package, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in members:
                archive.write(bundle / name, arcname=name)
        with zipfile.ZipFile(package) as archive:
            assert archive.testzip() is None
            assert set(archive.namelist()) == set(members)
        print(f"Verified {len(code)} cells in isolation without repo helpers or images.")
        print("All four reproducible scored CSV hashes match their recorded uploads.")
        print(f"Final notebook: {bundle / 'titanic.ipynb'}")
        print(f"Companion bundle: {package}")


if __name__ == "__main__":
    main()
