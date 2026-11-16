"""Execute trusted Python template cells offline with small, varied datasets."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from updater import apply_template_replacements


def run_notebook_code(tmp_path: Path, code: str) -> None:
    """Run notebook code as a script with a headless plotting backend."""
    script = tmp_path / "notebook.py"
    code = apply_template_replacements(
        code, {"USER_AGENT_LITERAL": repr("Test/1.0"), "TIMEOUT_LITERAL": "5"}
    )
    script.write_text(code, encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env={**os.environ, "MPLBACKEND": "Agg", "MPLCONFIGDIR": str(tmp_path / "mpl")},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "data",
    [
        "pd.DataFrame({'value': [1, None], 'label': ['a', 'a']}).convert_dtypes()",
        "pd.DataFrame({'value': [1, 2, 2]})",
        "pd.DataFrame({'label': ['a', 'b']})",
        "pd.DataFrame({'value': pd.Series(dtype='Float64')})",
        "pd.DataFrame({'value': pd.Series([pd.NA, pd.NA], dtype='Float64')})",
        "pd.DataFrame({'value': [float('inf'), -float('inf'), 1.0]})",
        "pd.DataFrame()",
    ],
)
def test_python_eda_handles_small_and_missing_data(tmp_path: Path, data: str) -> None:
    """EDA keeps the source intact and handles empty, text-only and non-finite data."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "_templates/template_python.ipynb").read_text())
    cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    code = "\n\n".join(cells)
    assert "missingno" not in code
    assert "encoding_errors='ignore'" not in code
    assert "except Exception" not in code
    assert "sns.barplot" in code
    code = apply_template_replacements(
        code,
        {
            "USER_AGENT_LITERAL": repr("Test/1.0"),
            "TIMEOUT_LITERAL": "5",
            "DISTRIBUTION": f"df = {data}\noriginal = df.copy(deep=True)",
        },
    )
    code = "def display(value):\n    print(value)\n\n" + code
    code += "\n\npd.testing.assert_frame_equal(df, original)\n"
    code += "assert profile['missing'].to_dict() == df.isna().sum().to_dict()\n"
    code += "assert set(profile.index) == set(df.columns)\n"
    run_notebook_code(tmp_path, code)


def test_python_csv_loader_preserves_data_and_rejects_corruption(tmp_path: Path) -> None:
    """CSV dialects and nullable values work without suppressing parsing failures."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "_templates/template_python.ipynb").read_text())
    helper = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def get_dataset(" in "".join(cell["source"])
    )
    (tmp_path / "comma.txt").write_text('label,value\n"a,b",1\nother,\n')
    (tmp_path / "semicolon.csv").write_text('label;value\n"a;b";1\nother;\n')
    (tmp_path / "single.csv").write_text("value\n1\n2\n")
    (tmp_path / "broken.csv").write_text("a,b\n1,2\n3,4,5\n")
    (tmp_path / "encoding.csv").write_bytes(b"a,b\n1,\xff\n")
    code = (
        "import pandas as pd\n"
        + helper
        + """
for filename, label in [("comma.txt", "a,b"), ("semicolon.csv", "a;b")]:
    data = get_dataset(filename)
    assert data.shape == (2, 2)
    assert data.loc[0, "label"] == label
    assert data.loc[0, "value"] == 1
    assert pd.isna(data.loc[1, "value"])
assert get_dataset("single.csv", sep=",").shape == (2, 1)
failures = [("broken.csv", pd.errors.ParserError), ("encoding.csv", UnicodeDecodeError)]
for filename, error in failures:
    try:
        get_dataset(filename, sep=",")
    except error:
        pass
    else:
        raise AssertionError(f"{filename} was silently accepted")
"""
    )
    run_notebook_code(tmp_path, code)


def test_csv_loader_reads_gzip_files_over_http(tmp_path: Path) -> None:
    """A gzip CSV file must load over HTTP just as it does from a local path."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "_templates/template_python.ipynb").read_text())
    helper = next("".join(c["source"]) for c in notebook["cells"] if c.get("id") == "csv-reader")
    code = (
        "import gzip\nfrom io import BytesIO\nfrom pathlib import Path\n"
        "from unittest.mock import patch\nimport pandas as pd\nimport pytest\n"
        + helper
        + """
contents = gzip.compress(b"label,value\\nfirst,1\\nsecond,2\\n")
Path("data.csv.gz").write_bytes(contents)
expected = {"label": ["first", "second"], "value": [1, 2]}
assert get_dataset("data.csv.gz").to_dict("list") == expected
for options in ({}, {"compression": "infer"}, {"compression": "gzip"},
                {"compression": {"method": "gzip"}}):
    response = requests.Response()
    response.status_code = 200
    response.raw = BytesIO(contents)
    with patch("requests.get", return_value=response):
        data = get_dataset("https://example.invalid/data.csv.gz", **options)
        assert data.to_dict("list") == expected
    assert response.raw.closed
response = requests.Response()
response.status_code = 200
response.raw = BytesIO(contents)
with patch("requests.get", return_value=response), pytest.raises(UnicodeDecodeError):
    get_dataset("https://example.invalid/data.csv.gz", compression=None)
assert response.raw.closed
"""
    )
    run_notebook_code(tmp_path, code)


@pytest.mark.parametrize("separator", [None, ","], ids=["inferred", "explicit"])
@pytest.mark.parametrize(
    "contents",
    ["a,b\n1,2,\n3,4,\n", "a,b\n1,2,3\n4,5,6\n"],
    ids=["trailing-empty-field", "extra-value"],
)
def test_csv_loader_rejects_implicit_index(
    tmp_path: Path, separator: str | None, contents: str
) -> None:
    """Consistently oversized rows must not silently shift values under headers."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "_templates/template_python.ipynb").read_text())
    helper = next("".join(c["source"]) for c in notebook["cells"] if c.get("id") == "csv-reader")
    (tmp_path / "oversized.csv").write_text(contents, encoding="utf-8")
    code = (
        "import pandas as pd\nimport pytest\n"
        + helper
        + f"\nseparator = {separator!r}\n"
        + """
for options in ({}, {"index_col": None}):
    with pytest.raises(pd.errors.ParserError, match="index_col"):
        get_dataset("oversized.csv", sep=separator, **options)
"""
    )
    run_notebook_code(tmp_path, code)


def test_csv_loader_preserves_explicit_index(tmp_path: Path) -> None:
    """Callers may deliberately load an index absent from the CSV header."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "_templates/template_python.ipynb").read_text())
    helper = next("".join(c["source"]) for c in notebook["cells"] if c.get("id") == "csv-reader")
    (tmp_path / "indexed.csv").write_text("a,b\nrow1,1,2\nrow2,3,4\n", encoding="utf-8")
    code = (
        "import pandas as pd\n"
        + helper
        + """
data = get_dataset("indexed.csv", sep=",", index_col=0)
assert data.index.tolist() == ["row1", "row2"]
assert data.to_dict("list") == {"a": [1, 3], "b": [2, 4]}
"""
    )
    run_notebook_code(tmp_path, code)


@pytest.mark.parametrize("separator", [",", ";"])
def test_csv_loader_removes_only_empty_unnamed_columns(tmp_path: Path, separator: str) -> None:
    """Trailing delimiters must not create phantom variables in EDA."""
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads((root / "_templates/template_python.ipynb").read_text())
    helper = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def get_dataset(" in "".join(cell["source"])
    )
    rows = [
        ["value", "named_empty", "", ""],
        ["1", "", "42", ""],
        ["2", "", "", ""],
    ]
    (tmp_path / "trailing.csv").write_text(
        "\n".join(separator.join(row) for row in rows) + "\n", encoding="utf-8"
    )
    (tmp_path / "header_only.csv").write_text("value,\n", encoding="utf-8")
    code = (
        "import pandas as pd\n"
        + helper
        + """
data = get_dataset("trailing.csv")
assert data.columns.tolist() == ["value", "named_empty", "Unnamed: 2"]
assert data["value"].tolist() == [1, 2]
assert data["named_empty"].isna().all()
assert data.loc[0, "Unnamed: 2"] == 42
assert pd.isna(data.loc[1, "Unnamed: 2"])
original = get_dataset("trailing.csv", drop_empty_unnamed=False)
assert original.columns.tolist() == ["value", "named_empty", "Unnamed: 2", "Unnamed: 3"]
assert original["Unnamed: 3"].isna().all()
# With no rows, there is no evidence that the column is an import artifact.
assert get_dataset("header_only.csv", sep=",").columns.tolist() == ["value", "Unnamed: 1"]
"""
    )
    run_notebook_code(tmp_path, code)
