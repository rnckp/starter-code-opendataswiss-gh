"""Raw CKAN catalogue fixtures and isolated release output."""

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import updater


@pytest.fixture
def config(tmp_path: Path) -> updater.Config:
    root = Path(__file__).resolve().parents[1]
    return replace(
        updater.Config.from_yaml(root / "config.yaml"),
        template_folder=root / "_templates",
        temp_prefix=tmp_path / "output",
        api_sleep=0,
    )


@pytest.fixture
def catalogue() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": "123@example",
                "name": "catalogue-slug",
                "identifier": "external/identifier@publisher",
                "title": {"de": 'Title "quoted" | <script>bad</script>'},
                "description": {"de": 'A backslash \\ and {{ CONTACT }}\n```{r}\nstop("bad")\n```'},
                "contact_points": [{"name": 'Example "quoted"'}],
                "organization": {"display_name": {"de": "Example organization"}},
                "resources": [
                    {
                        "id": "one",
                        "name": {"de": "CSV | resource"},
                        "format": "csv",
                        "download_url": "https://example.invalid/download?name=a'b&x=1",
                    },
                    {
                        "id": "two",
                        "title": {"de": "Second CSV resource"},
                        "format": " CSV ",
                        "url": "https://example.invalid/second-download",
                    },
                ],
            }
        ]
    )


@pytest.fixture
def generated_release(
    config: updater.Config, catalogue: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> updater.Config:
    monkeypatch.setattr(updater, "get_full_package_list", lambda _: catalogue)
    updater.run_update(config)
    return config
