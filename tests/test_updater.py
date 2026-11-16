"""CKAN metadata and command-line regression tests."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

import updater


def test_script_help_works_outside_checkout(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "updater.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout
    assert "--verify-output" in result.stdout


def test_default_config_uses_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load((root / "config.yaml").read_text())
    settings["provider"]["name"] = "Local provider"
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(settings))
    monkeypatch.chdir(tmp_path)
    assert updater.Config.from_yaml().provider == "Local provider"


def test_missing_default_config_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError):
        updater.Config.from_yaml()


def test_generation_and_verification_use_utf8_in_non_utf8_locale(
    tmp_path: Path, catalogue: pd.DataFrame
) -> None:
    """Locale encoding must not affect template text or published Unicode metadata."""
    catalogue.at[0, "title"] = {"de": "Grüsse aus Zürich 東京"}
    code = """
import json
import locale
import sys
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import updater

assert locale.getencoding().lower().replace("-", "") != "utf8"
rows = json.load(sys.stdin)
config = replace(updater.Config.from_yaml(), temp_prefix=Path(sys.argv[1]))
with patch("updater.get_full_package_list", return_value=pd.DataFrame(rows)):
    updater.run_update(config)
updater.verify_output(config)
manifest = json.loads((config.temp_prefix / "manifest.json").read_text(encoding="utf-8"))
for name in manifest["files"]:
    text = (config.temp_prefix / name).read_text(encoding="utf-8")
    assert rows[0]["title"]["de"] in text, name
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "output")],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "LC_ALL": "C", "PYTHONUTF8": "0", "PYTHONCOERCECLOCALE": "0"},
        input=catalogue.to_json(orient="records", force_ascii=True),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("title", [{"it": "Solo italiano"}, "Plain title"])
def test_optional_ckan_metadata_and_multilingual_titles(
    config: updater.Config, title: dict[str, str] | str
) -> None:
    data = pd.DataFrame(
        [
            {
                "id": "dataset-one",
                "name": "dataset-slug",
                "title": title,
                "resources": [
                    {"id": "resource", "format": "CSV", "url": "https://example.invalid/data"}
                ],
            }
        ]
    )
    prepared = updater.prepare_data_for_codebooks(data, config)
    assert prepared.iloc[0]["title"] == ("Solo italiano" if isinstance(title, dict) else title)
    assert prepared.iloc[0]["catalogue_id"] == "dataset-slug"
    assert prepared.iloc[0]["filename"] == "dataset-one--resource"


@pytest.mark.parametrize("field", ["download_url", "url"])
@pytest.mark.parametrize(
    ("raw_url", "expected_url"),
    [
        ("www.example.invalid/data.csv", "https://www.example.invalid/data.csv"),
        ("https://example.invalid/data.csv ", "https://example.invalid/data.csv"),
    ],
    ids=["missing-scheme", "trailing-whitespace"],
)
def test_resource_urls_normalize_known_catalogue_defects(
    config: updater.Config, field: str, raw_url: str, expected_url: str
) -> None:
    catalogue = pd.DataFrame(
        [
            {
                "id": "dataset-one",
                "resources": [{"id": "resource", "format": "CSV", field: raw_url}],
            }
        ]
    )

    prepared = updater.prepare_data_for_codebooks(catalogue, config)

    assert prepared.iloc[0]["url"] == expected_url


def test_quoted_contact_does_not_silently_omit_notebooks(
    generated_release: updater.Config,
) -> None:
    config = generated_release
    manifest = json.loads((config.temp_prefix / "manifest.json").read_text())
    assert len(manifest["resource_files"]) == 2
    for entry in manifest["resource_files"]:
        notebook = json.loads((config.temp_prefix / entry["python"]).read_text())
        text = "\n".join("".join(cell["source"]) for cell in notebook["cells"])
        assert "Example &quot;quoted&quot;" in text


def test_verify_cli_is_offline_read_only_and_rejects_corruption(
    generated_release: updater.Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = generated_release
    root = Path(__file__).resolve().parents[1]
    settings = yaml.safe_load((root / "config.yaml").read_text())
    settings["paths"]["temp_prefix"] = str(config.temp_prefix)
    config_path = tmp_path / "operator.yaml"
    config_path.write_text(yaml.safe_dump(settings))

    def unexpected_fetch(_: updater.Config) -> None:
        pytest.fail("--verify-output fetched catalogue metadata")

    monkeypatch.setattr(updater, "get_full_package_list", unexpected_fetch)
    monkeypatch.chdir(tmp_path)
    before = {
        p.relative_to(config.temp_prefix): p.read_bytes()
        for p in config.temp_prefix.rglob("*")
        if p.is_file()
    }
    command = ["--config", str(config_path), "--verify-output"]
    updater.main(command)
    assert before == {
        p.relative_to(config.temp_prefix): p.read_bytes()
        for p in config.temp_prefix.rglob("*")
        if p.is_file()
    }
    (config.temp_prefix / "index.md").unlink()
    with pytest.raises(ValueError, match="integrity"):
        updater.main(command)
    assert not (config.temp_prefix / "index.md").exists()
