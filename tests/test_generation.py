"""Resource generation and publication contracts, without external services."""

import ast
import html
import json
import re
import shutil
from dataclasses import replace
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import pytest

import updater


def release_files(root: Path) -> dict[Path, bytes]:
    """Capture all release content, including operator-owned files."""
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def assert_release_preserved(config: updater.Config, before: dict[Path, bytes]) -> None:
    """Failed updates must preserve all bytes and clean up transient releases."""
    assert release_files(config.temp_prefix) == before
    assert not list(config.temp_prefix.parent.glob(".starter-release-*"))
    assert not config.temp_prefix.with_name(f".{config.temp_prefix.name}.previous").exists()


def generate(
    monkeypatch: pytest.MonkeyPatch, config: updater.Config, catalogue: pd.DataFrame
) -> None:
    """Replace only the external catalogue boundary."""
    monkeypatch.setattr(updater, "get_full_package_list", lambda _: catalogue)
    updater.run_update(config)


@pytest.mark.parametrize(
    ("catalogue_id", "expected_id"),
    [(262, "262"), ("resource-one", "resource-one"), (None, "url-e001b9d33811")],
)
def test_resource_filenames_use_catalogue_ids_with_short_url_fallback(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    catalogue_id: int | str | None,
    expected_id: str,
) -> None:
    """Paths, manifest and overview links expose the resource identity directly."""
    resource = catalogue.at[0, "resources"][0]
    resource["download_url"] = "https://example.invalid/no-id.csv"
    if catalogue_id is None:
        del resource["id"]
    else:
        resource["id"] = catalogue_id
    catalogue.at[0, "resources"] = [resource]
    generate(monkeypatch, config, catalogue)
    manifest = json.loads((config.temp_prefix / "manifest.json").read_text())
    paths = {
        "python": f"{config.python_output}/123@example--{expected_id}.ipynb",
        "r": f"{config.r_markdown_output}/123@example--{expected_id}.Rmd",
    }
    assert manifest["resource_files"] == [
        {"dataset_id": "123@example", "resource_id": expected_id, **paths}
    ]
    for path in paths.values():
        assert (config.temp_prefix / path).is_file()
        for page in ("README.md", "index.md"):
            assert quote(path) in (config.temp_prefix / page).read_text()


@pytest.mark.parametrize("collision", ["delimiter", "case", "fallback"])
def test_readable_filename_collisions_preserve_previous_release(
    generated_release: updater.Config,
    monkeypatch: pytest.MonkeyPatch,
    collision: str,
) -> None:
    """Ambiguous IDs and fallback collisions must fail instead of overwriting files."""
    identities = {
        "delimiter": [("a--b", "c"), ("a", "b--c")],
        "case": [("a", "one"), ("a", "ONE")],
        "fallback": [("a", None), ("a", "url-e001b9d33811")],
    }[collision]
    rows: dict[str, dict] = {}
    for dataset_id, resource_id in identities:
        row = rows.setdefault(dataset_id, {"id": dataset_id, "resources": []})
        row["resources"].append(
            {
                "id": resource_id,
                "format": "csv",
                "download_url": "https://example.invalid/no-id.csv",
            }
        )
    before = release_files(generated_release.temp_prefix)
    with pytest.raises(ValueError, match="Duplicate resource identifier|filename collision"):
        generate(monkeypatch, generated_release, pd.DataFrame(rows.values()))
    assert_release_preserved(generated_release, before)


@pytest.mark.parametrize("resource_count", [1, 2])
def test_generation_removes_legacy_selection_files_and_emits_only_resources(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    resource_count: int,
) -> None:
    """Each CSV has exactly one file per language, including after migration."""
    catalogue.at[0, "resources"] = catalogue.at[0, "resources"][:resource_count]
    for directory, extension in (
        (config.python_output, ".ipynb"),
        (config.r_markdown_output, ".Rmd"),
    ):
        output = config.temp_prefix / directory
        output.mkdir(parents=True)
        (output / f"123@example{extension}").write_text("legacy selection page")
        (output / f"123@example--610ac157b090344dc441{extension}").write_text(
            "legacy hashed resource notebook"
        )
    generate(monkeypatch, config, catalogue)
    manifest = json.loads((config.temp_prefix / "manifest.json").read_text())
    assert len(manifest["files"]) == 2 + 2 * resource_count
    for directory, key in ((config.python_output, "python"), (config.r_markdown_output, "r")):
        actual = {
            str(path.relative_to(config.temp_prefix))
            for path in (config.temp_prefix / directory).iterdir()
        }
        assert len(actual) == resource_count
        assert actual == {entry[key] for entry in manifest["resource_files"]}
    updater.verify_output(config)


@pytest.mark.parametrize(
    "value",
    ["Zurich's population", "['bevoelkerung', 'kanton_zuerich']", "Literal &#x27; & <tag>"],
)
def test_markdown_entities_decode_once_to_original_text(value: str) -> None:
    """Escaping must not corrupt its own entities or interpret source entities."""
    escaped = updater.markdown(value)
    assert html.unescape(escaped) == value
    assert "<tag>" not in escaped


@pytest.mark.parametrize("spacing", ["", " ", "  \t"])
def test_template_markers_allow_optional_whitespace(spacing: str) -> None:
    """Formatting marker whitespace must not change substitution or reprocess values."""
    template = "before {{" + spacing + "VALUE" + spacing + "}} after"
    assert updater.apply_template_replacements(template, {"VALUE": "{{OTHER}}"}) == (
        "before {{OTHER}} after"
    )
    with pytest.raises(ValueError, match="Unknown template marker: VALUE"):
        updater.apply_template_replacements(template, {})


@pytest.mark.parametrize("spacing", ["", " "])
def test_generation_accepts_compact_and_spaced_template_markers(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    spacing: str,
) -> None:
    """Generate all artifacts using either spelling of every template marker."""
    templates = tmp_path / "templates"
    shutil.copytree(config.template_folder, templates)
    for path in templates.iterdir():
        path.write_text(
            re.sub(r"{{\s*([A-Z_]+)\s*}}", rf"{{{{{spacing}\1{spacing}}}}}", path.read_text())
        )
    generate(monkeypatch, replace(config, template_folder=templates), catalogue)
    manifest = json.loads((config.temp_prefix / "manifest.json").read_text())
    for entry in manifest["resource_files"]:
        notebook = json.loads((config.temp_prefix / entry["python"]).read_text())
        code = "\n".join(
            "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
        )
        assert "{{" not in code
        assert code.count("df = get_dataset(") == 1
        assert f"USER_AGENT = {json.dumps(config.user_agent_python)}" in code
        assert f"HTTP_TIMEOUT = {config.http_timeout}" in code
        r_code = (config.temp_prefix / entry["r"]).read_text()
        assert "{{" not in r_code
        assert r_code.count("df <- get_dataset(") == 1
    updater.verify_output(config)


def test_notebooks_include_catalogue_link(generated_release: updater.Config) -> None:
    """Notebook formats retain a readable link to the encoded catalogue URL."""
    expected = (
        "[View this dataset in the data catalogue]("
        + generated_release.base_link
        + "catalogue-slug)"
    )
    for path in (generated_release.temp_prefix / generated_release.python_output).glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        assert any(expected in "".join(cell["source"]) for cell in notebook["cells"])
    for path in (generated_release.temp_prefix / generated_release.r_markdown_output).glob("*.Rmd"):
        assert expected in path.read_text()


def test_readme_uses_configured_template(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """README layout can change while sharing the rendered header and table."""
    templates = tmp_path / "templates"
    shutil.copytree(config.template_folder, templates)
    (templates / "custom_readme.md").write_text(
        "{{ HEADER }}\nCustom navigation: {{ PAGES_URL }}\n{{ RESOURCE_TABLE }}"
    )
    config = replace(config, template_folder=templates, template_readme="custom_readme.md")
    generate(monkeypatch, config, catalogue)
    readme = (config.temp_prefix / "README.md").read_text()
    index = (config.temp_prefix / "index.md").read_text()
    header, table = index.split("\n| Notebooks", maxsplit=1)
    assert readme == (
        header
        + "\nCustom navigation: https://rnckp.github.io/starter-code_opendataswiss/\n"
        + "\n| Notebooks"
        + table
    )


def test_overviews_use_configured_header(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Both entry points render the configured header and its dynamic values."""
    templates = tmp_path / "templates"
    shutil.copytree(config.template_folder, templates)
    (templates / "custom_header.md").write_text(
        "# Custom catalogue\n\n"
        "{{ DATASET_COUNT }} datasets / {{ RESOURCE_COUNT }} resources\n"
        "Updated {{ TODAY_DATE }}\n"
    )
    config = replace(config, template_folder=templates, template_header="custom_header.md")
    generate(monkeypatch, config, catalogue)
    headers = []
    for page in ("README.md", "index.md"):
        overview = (config.temp_prefix / page).read_text()
        assert overview.startswith("# Custom catalogue\n\n1 datasets / 2 resources\n")
        assert re.search(r"^Updated \d{4}-\d{2}-\d{2}$", overview, re.MULTILINE)
        assert "{{" not in overview
        headers.append(overview.splitlines()[:4])
    assert headers[0] == headers[1]


@pytest.mark.parametrize("page", ["README.md", "index.md"])
def test_overviews_put_notebook_links_first(generated_release: updater.Config, page: str) -> None:
    """Both entry points put notebook links before dataset and resource labels."""
    root = generated_release.temp_prefix
    overview = (root / page).read_text()
    table = [line for line in overview.splitlines() if line.startswith("|")]
    assert table[:2] == [
        "| Notebooks | Dataset |",
        "| :-- | :-- |",
    ]
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(table[2:]) == len(manifest["resource_files"]) == 2
    github = "https://github.com/rnckp/starter-code_opendataswiss/blob/main/"
    colab = "https://colab.research.google.com/github/rnckp/starter-code_opendataswiss/blob/main/"
    resources = {"one": "CSV &#124; resource", "two": "Second CSV resource"}
    for entry in manifest["resource_files"]:
        resource = resources[entry["resource_id"]]
        row = next(line for line in table[2:] if f"Resource: {resource} |" in line)
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert len(cells) == 2
        assert cells[1] == (
            "[Title &quot;quoted&quot; &#124; &lt;script&gt;bad&lt;/script&gt;]("
            "https://opendata.swiss/de/dataset/catalogue-slug)<br>"
            f"Resource: {resource}"
        )
        assert cells[0] == (
            f"[Colab]({colab}{quote(entry['python'])})<br>"
            f"[Python]({github}{quote(entry['python'])})<br>"
            f"[R]({github}{quote(entry['r'])})"
        )


def test_resource_outputs_are_safe_complete_and_individually_loadable(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generate(monkeypatch, config, catalogue)
    notebooks = list((config.temp_prefix / config.python_output).glob("*.ipynb"))
    resource_code = []
    for path in notebooks:
        notebook = json.loads(path.read_text())
        code = "\n".join(
            "".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"
        )
        ast.parse(code)
        if "df = get_dataset(" in code:
            resource_code.append(code)
            assert code.count("df = get_dataset(") == 1
            assert 'stop("bad")' not in code
    assert len(resource_code) == 2
    assert all("file_format" not in code and "read_parquet" not in code for code in resource_code)
    assert any("a'b&x=1" in code for code in resource_code)
    assert not (config.temp_prefix / config.python_output / "123@example.ipynb").exists()
    overview = (config.temp_prefix / "index.md").read_text()
    assert "CSV &#124; resource" in overview
    assert "<script>" not in overview
    assert "Second CSV resource" in overview
    assert "[R online](" not in overview
    for path in (config.temp_prefix / config.r_markdown_output).glob("*.Rmd"):
        text = path.read_text()
        assert '\nstop("bad")\n' not in text
    updater.verify_output(config)


def test_only_csv_resources_are_published(
    config: updater.Config, catalogue: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Filter by normalized catalogue format, never by the download URL suffix."""
    catalogue.at[0, "resources"].extend(
        [
            {
                "id": f"ignored-{index}",
                "format": file_format,
                "download_url": "https://example.invalid/misleading.csv",
            }
            for index, file_format in enumerate(["PARQUET", "pdf", "json", "", None])
        ]
    )
    non_csv_dataset = pd.DataFrame([{"id": "non-csv", "resources": [{"format": "parquet"}]}])
    generate(monkeypatch, config, pd.concat([catalogue, non_csv_dataset], ignore_index=True))
    manifest = json.loads((config.temp_prefix / "manifest.json").read_text())
    assert manifest["datasets"] == 1
    assert manifest["resources"] == 2
    assert {entry["resource_id"] for entry in manifest["resource_files"]} == {"one", "two"}
    for page in ("README.md", "index.md"):
        overview = (config.temp_prefix / page).read_text()
        assert "non-csv" not in overview
        assert "misleading.csv" not in overview
    assert not (config.temp_prefix / config.python_output / "non-csv.ipynb").exists()
    assert not (config.temp_prefix / config.r_markdown_output / "non-csv.Rmd").exists()


@pytest.mark.parametrize("fields", [{}, {"resources": None}], ids=["omitted", "null"])
def test_datasets_without_distributions_do_not_block_csv_generation(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    fields: dict[str, object],
) -> None:
    """Missing optional distributions cannot suppress other datasets' CSV resources."""
    rows = [*catalogue.to_dict("records"), {"id": "unpublished", **fields}]
    generate(monkeypatch, config, pd.DataFrame(rows))
    manifest = json.loads((config.temp_prefix / "manifest.json").read_text())
    assert manifest["datasets"] == 1
    assert manifest["resources"] == 2
    assert {
        (entry["dataset_id"], entry["resource_id"]) for entry in manifest["resource_files"]
    } == {("123@example", "one"), ("123@example", "two")}


@pytest.mark.parametrize("distributions", [{}, "invalid", 1], ids=["object", "string", "number"])
def test_non_null_malformed_distributions_preserve_previous_release(
    generated_release: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    distributions: object,
) -> None:
    """Missing-value handling must retain validation of malformed external data."""
    before = release_files(generated_release.temp_prefix)
    catalogue.at[0, "resources"] = distributions
    with pytest.raises(ValueError, match="Dataset distributions must be a list"):
        generate(monkeypatch, generated_release, catalogue)
    assert_release_preserved(generated_release, before)


def test_catalogue_without_csv_preserves_previous_release(
    generated_release: updater.Config, catalogue: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unsupported resources cannot replace an existing CSV publication."""
    before = release_files(generated_release.temp_prefix)
    for resource in catalogue.at[0, "resources"]:
        resource["format"] = "parquet"
    with pytest.raises(ValueError, match="No CSV resources"):
        generate(monkeypatch, generated_release, catalogue)
    assert_release_preserved(generated_release, before)


def test_failed_generation_preserves_previous_release(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A real R rendering failure after Python generation leaves no partial release."""
    generate(monkeypatch, config, catalogue)
    before = release_files(config.temp_prefix)
    templates = tmp_path / "broken-templates"
    shutil.copytree(config.template_folder, templates)
    template = templates / config.template_rmarkdown
    template.write_text(template.read_text() + "\n{{ UNKNOWN }}\n")
    with pytest.raises(ValueError, match="Unknown template marker: UNKNOWN"):
        updater.run_update(replace(config, template_folder=templates))
    assert_release_preserved(config, before)


@pytest.mark.parametrize("omitted", [False, True], ids=["empty-list", "omitted"])
def test_empty_catalogue_fails_without_replacing_output(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    omitted: bool,
) -> None:
    generate(monkeypatch, config, catalogue)
    before = release_files(config.temp_prefix)
    if omitted:
        catalogue = catalogue.drop(columns="resources")
    else:
        catalogue.at[0, "resources"] = []
    with pytest.raises(ValueError, match="No CSV resources"):
        generate(monkeypatch, config, catalogue)
    assert_release_preserved(config, before)


@pytest.mark.parametrize("identifier", ["../escape", "/tmp/escape", "a/b", "a\\b", ".."])
def test_unsafe_identifiers_are_rejected(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    identifier: str,
) -> None:
    catalogue.at[0, "id"] = identifier
    with pytest.raises(ValueError, match="identifier"):
        generate(monkeypatch, config, catalogue)
    assert not config.temp_prefix.exists()


@pytest.mark.parametrize("change", ["removed", "non-csv"])
def test_stale_resources_removed_and_static_files_preserved(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
) -> None:
    generate(monkeypatch, config, catalogue)
    (config.temp_prefix / "LICENSE.md").write_text("Keep me")
    manifest = json.loads((config.temp_prefix / "manifest.json").read_text())
    paths = {entry["resource_id"]: entry for entry in manifest["resource_files"]}
    assert set(paths) == {"one", "two"}
    if change == "removed":
        catalogue.at[0, "resources"] = catalogue.at[0, "resources"][:1]
    else:
        catalogue.at[0, "resources"][1]["format"] = "parquet"
    generate(monkeypatch, config, catalogue)
    for key in ("python", "r"):
        assert (config.temp_prefix / paths["one"][key]).is_file()
        assert not (config.temp_prefix / paths["two"][key]).exists()
    for directory, extension in (
        (config.python_output, ".ipynb"),
        (config.r_markdown_output, ".Rmd"),
    ):
        assert not (config.temp_prefix / directory / f"123@example{extension}").exists()
    assert (config.temp_prefix / "LICENSE.md").read_text() == "Keep me"


@pytest.mark.parametrize("mutation", ["duplicate", "unsafe_resource", "unsafe_url"])
def test_invalid_resources_never_replace_output(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    generate(monkeypatch, config, catalogue)
    before = release_files(config.temp_prefix)
    resources = catalogue.at[0, "resources"]
    if mutation == "duplicate":
        resources[1]["id"] = "one"
    elif mutation == "unsafe_resource":
        resources[0]["id"] = "../../outside"
    else:
        resources[0]["download_url"] = "file:///etc/passwd"
    with pytest.raises(ValueError):
        updater.run_update(config)
    assert_release_preserved(config, before)


@pytest.mark.parametrize("language", ["python", "r"])
@pytest.mark.parametrize("marker_count", [0, 2], ids=["missing", "duplicate"])
def test_invalid_template_marker_count_aborts_release(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    language: str,
    marker_count: int,
) -> None:
    generate(monkeypatch, config, catalogue)
    before = release_files(config.temp_prefix)
    templates = tmp_path / "templates"
    shutil.copytree(config.template_folder, templates)
    if language == "python":
        path = templates / config.template_python
        template = json.loads(path.read_text())
        for cell in template["cells"]:
            cell["source"] = [
                re.sub(
                    r"{{\s*DISTRIBUTION\s*}}",
                    "{{DISTRIBUTION}}\n{{ DISTRIBUTION }}" if marker_count == 2 else "pass",
                    "".join(cell["source"]),
                )
            ]
        path.write_text(json.dumps(template))
    else:
        path = templates / config.template_rmarkdown
        path.write_text(
            re.sub(
                r"{{\s*DISTRIBUTIONS\s*}}",
                "{{DISTRIBUTIONS}}\n{{ DISTRIBUTIONS }}" if marker_count == 2 else "NULL",
                path.read_text(),
            )
        )
    with pytest.raises(ValueError, match="exactly one"):
        updater.run_update(replace(config, template_folder=templates))
    assert_release_preserved(config, before)


def test_resource_filenames_stable_when_reordered(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generate(monkeypatch, config, catalogue)
    before = json.loads((config.temp_prefix / "manifest.json").read_text())["resource_files"]
    catalogue.at[0, "resources"].reverse()
    generate(monkeypatch, config, catalogue)
    after = json.loads((config.temp_prefix / "manifest.json").read_text())["resource_files"]
    assert before == after


def test_release_swap_failure_restores_previous_output(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generate(monkeypatch, config, catalogue)
    before = release_files(config.temp_prefix)
    backup = config.temp_prefix.with_name(".output.previous")
    original_rename = Path.rename

    def failing_rename(path: Path, target: Path) -> Path:
        if target == config.temp_prefix.absolute() and path != backup:
            raise OSError("simulated replacement failure")
        return original_rename(path, target)

    monkeypatch.setattr(Path, "rename", failing_rename)
    with pytest.raises(OSError, match="simulated"):
        updater.run_update(config)
    assert_release_preserved(config, before)


def test_failed_rollback_keeps_recoverable_release_and_blocks_next_run(
    generated_release: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed second rename and rollback must never delete recovery data."""
    config = generated_release
    (config.temp_prefix / "LICENSE.md").write_text("operator-owned license")
    before = release_files(config.temp_prefix)
    backup = config.temp_prefix.with_name(".output.previous")
    original_rename = Path.rename
    catalogue.at[0, "title"] = "Changed release"

    def fail_install_or_restore(path: Path, target: Path) -> Path:
        if target == config.temp_prefix.absolute():
            raise OSError("cannot install or restore")
        return original_rename(path, target)

    with monkeypatch.context() as faults:
        faults.setattr(Path, "rename", fail_install_or_restore)
        with pytest.raises(OSError, match="cannot install or restore"):
            updater.run_update(config)
    assert not config.temp_prefix.exists()
    assert release_files(backup) == before
    assert not list(config.temp_prefix.parent.glob(".starter-release-*"))

    with pytest.raises(ValueError, match="backup exists"):
        updater.run_update(config)
    assert release_files(backup) == before
    assert not config.temp_prefix.exists()
    assert not list(config.temp_prefix.parent.glob(".starter-release-*"))


def test_removed_dataset_pages_and_both_language_outputs_are_deleted(
    config: updater.Config, catalogue: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Removing a dataset removes both language outputs, including selection pages."""
    survivor = pd.DataFrame(
        [
            {
                "id": "survivor",
                "resources": [
                    {
                        "id": "one",
                        "format": " CSV ",
                        "download_url": "https://example.invalid/kept",
                    },
                    {"format": "PDF", "download_url": "https://example.invalid/ignored"},
                ],
            }
        ]
    )
    generate(monkeypatch, config, pd.concat([catalogue, survivor], ignore_index=True))
    manifest = json.loads((config.temp_prefix / "manifest.json").read_text())
    removed = [
        entry[key]
        for entry in manifest["resource_files"]
        if entry["dataset_id"] == "123@example"
        for key in ("python", "r")
    ]
    assert len(removed) == 4
    removed += [
        str(config.python_output / "123@example.ipynb"),
        str(config.r_markdown_output / "123@example.Rmd"),
    ]
    static = config.temp_prefix / "assets" / "license.txt"
    static.parent.mkdir()
    static.write_text("keep this nested static file")
    generate(monkeypatch, config, survivor)
    assert all(not (config.temp_prefix / name).exists() for name in removed)
    manifest = json.loads((config.temp_prefix / "manifest.json").read_text())
    assert manifest["datasets"] == manifest["resources"] == 1
    assert [
        (entry["dataset_id"], entry["resource_id"]) for entry in manifest["resource_files"]
    ] == [("survivor", "one")]
    for name in (
        config.python_output / "survivor.ipynb",
        config.r_markdown_output / "survivor.Rmd",
    ):
        assert not (config.temp_prefix / name).exists()
    assert static.read_text() == "keep this nested static file"
    updater.verify_output(config)


@pytest.mark.parametrize("identified", [True, False], ids=["catalogue-id", "url-fallback"])
def test_resource_identity_tracks_ids_and_unidentified_urls(
    config: updater.Config,
    catalogue: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    identified: bool,
) -> None:
    """README promises stable paths for identified resources when only URLs change."""
    resource = catalogue.at[0, "resources"][0]
    catalogue.at[0, "resources"] = [resource]
    if not identified:
        del resource["id"]
    generate(monkeypatch, config, catalogue)
    before = json.loads((config.temp_prefix / "manifest.json").read_text())["resource_files"][0]
    resource["download_url"] = "https://example.invalid/new-location"
    generate(monkeypatch, config, catalogue)
    after = json.loads((config.temp_prefix / "manifest.json").read_text())["resource_files"][0]
    if identified:
        assert after == before
    else:
        assert after["resource_id"] != before["resource_id"]
        for key in ("python", "r"):
            assert after[key] != before[key]
            assert not (config.temp_prefix / before[key]).exists()
    notebook = json.loads((config.temp_prefix / after["python"]).read_text())
    assert "https://example.invalid/new-location" in "".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )
    resource["id"] = "replacement-id"
    generate(monkeypatch, config, catalogue)
    replaced = json.loads((config.temp_prefix / "manifest.json").read_text())["resource_files"][0]
    assert replaced["resource_id"] == "replacement-id"
    for key in ("python", "r"):
        assert replaced[key] != after[key]
        assert not (config.temp_prefix / after[key]).exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"python_output": Path("../outside")},
        {"r_markdown_output": Path("/outside")},
        {"python_output": Path("nested/output")},
        {"python_output": Path("same"), "r_markdown_output": Path("same")},
        {"template_python": "../outside.ipynb"},
        {"api_limit": 0},
        {"api_sleep": -1},
    ],
    ids=[
        "parent-directory",
        "absolute-directory",
        "nested-directory",
        "shared-directory",
        "template-traversal",
        "invalid-api-limit",
        "invalid-api-sleep",
    ],
)
def test_unsafe_output_configuration_fails_before_fetch_or_write(
    generated_release: updater.Config,
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
) -> None:
    """Documented path constraints prevent invalid settings from touching a release."""
    config = generated_release
    before = release_files(config.temp_prefix)

    def unexpected_fetch(_: updater.Config) -> None:
        pytest.fail("Invalid paths must be rejected before fetching metadata")

    monkeypatch.setattr(updater, "get_full_package_list", unexpected_fetch)
    with pytest.raises(ValueError, match="Output subdirectories|Template filenames|API limit"):
        updater.run_update(replace(config, **changes))
    assert_release_preserved(config, before)


def test_readme_preview_keeps_full_catalogue_on_pages(
    config: updater.Config, catalogue: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = replace(config, readme_resource_limit=1)
    catalogue.at[0, "resources"][0]["name"] = "x" * 1000
    generate(monkeypatch, config, catalogue)
    for page, expected_rows in (("README.md", 3), ("index.md", 4)):
        overview = (config.temp_prefix / page).read_text()
        assert (
            len([line for line in overview.splitlines() if line.startswith("|")]) == expected_rows
        )
        assert "x" * config.title_max_chars + "…" in overview
        assert "x" * (config.title_max_chars + 1) not in overview
    readme = (config.temp_prefix / "README.md").read_text()
    assert "[Browse all resources](https://rnckp.github.io/starter-code_opendataswiss/)" in readme
    assert "index.md" not in readme
    assert "preview" in readme.lower()
    notebook = config.temp_prefix / config.python_output / "123@example--one.ipynb"
    assert "x" * 1000 in notebook.read_text()


def test_single_resource_dataset_omits_redundant_resource_label(
    config: updater.Config, catalogue: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalogue.at[0, "resources"] = catalogue.at[0, "resources"][:1]
    generate(monkeypatch, config, catalogue)
    for page in ("README.md", "index.md"):
        overview = (config.temp_prefix / page).read_text()
        assert "Resource:" not in overview
        assert "CSV &#124; resource" not in overview
        assert "[Colab](" in overview
