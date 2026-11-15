"""CKAN pagination and actual generated CSV HTTP requests, entirely offline."""

import gzip
import json
import subprocess
import sys
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from textwrap import indent
from threading import Thread
from urllib.parse import parse_qs, urlsplit

import pytest
import requests

import updater


def response(body: bytes, status: int = 200) -> requests.Response:
    result = requests.Response()
    result.status_code = status
    result.raw = BytesIO(body)
    return result


@pytest.mark.parametrize(
    ("body", "status", "error"),
    [
        (b'{"success": true, "result": []}', 503, requests.HTTPError),
        (b"not JSON", 200, requests.exceptions.JSONDecodeError),
        (b"[]", 200, ValueError),
        (b"null", 200, ValueError),
        (b"{}", 200, ValueError),
        (b'{"success": false, "error": {}}', 200, ValueError),
        (b'{"success": true, "result": {}}', 200, ValueError),
        (b'{"success": true, "result": [null]}', 200, ValueError),
    ],
)
def test_catalogue_rejects_failed_or_malformed_responses(
    config: updater.Config,
    monkeypatch: pytest.MonkeyPatch,
    body: bytes,
    status: int,
    error: type[Exception],
) -> None:
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: response(body, status))
    with pytest.raises(error):
        updater.get_full_package_list(config)


def test_pagination_advances_by_returned_count_and_requests_until_empty(
    config: updater.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    pages = iter([[{"id": "one"}], [{"id": "two"}], []])
    calls: list[dict[str, object]] = []

    def get(url: str, **kwargs: object) -> requests.Response:
        assert url == config.metadata_link
        calls.append(kwargs)
        return response(json.dumps({"success": True, "result": next(pages)}).encode())

    monkeypatch.setattr(requests, "get", get)
    result = updater.get_full_package_list(config)
    assert result["id"].tolist() == ["one", "two"]
    assert [call["params"] for call in calls] == [
        {"limit": config.api_limit, "offset": offset} for offset in [0, 1, 2]
    ]
    assert all(call["timeout"] == config.http_timeout for call in calls)
    assert all(call["headers"] == {"User-Agent": config.user_agent_generator} for call in calls)


@pytest.mark.parametrize("later_page", [[{"id": "one"}], [{"id": "ONE"}]])
def test_duplicate_pages_are_rejected(
    config: updater.Config, monkeypatch: pytest.MonkeyPatch, later_page: list[dict[str, str]]
) -> None:
    pages = iter([[{"id": "one"}], later_page])
    monkeypatch.setattr(
        requests,
        "get",
        lambda *args, **kwargs: response(
            json.dumps({"success": True, "result": next(pages)}).encode()
        ),
    )
    with pytest.raises(ValueError, match="Duplicate dataset"):
        updater.get_full_package_list(config)


def test_incomplete_catalogue_never_replaces_previous_release(
    generated_release: updater.Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = generated_release
    before = {
        p.relative_to(config.temp_prefix): p.read_bytes()
        for p in config.temp_prefix.rglob("*")
        if p.is_file()
    }
    pages = iter(
        [
            {"success": True, "result": [{"id": "one"}]},
            {"success": False, "error": {}},
        ]
    )
    # Restore the real fetch after the shared fixture generated its initial release.
    monkeypatch.undo()
    monkeypatch.setattr(
        requests, "get", lambda *args, **kwargs: response(json.dumps(next(pages)).encode())
    )
    with pytest.raises(ValueError, match="unsuccessful"):
        updater.run_update(config)
    assert before == {
        p.relative_to(config.temp_prefix): p.read_bytes()
        for p in config.temp_prefix.rglob("*")
        if p.is_file()
    }


@pytest.mark.parametrize("compressed", [False, True], ids=["plain", "gzip"])
def test_generator_and_notebook_send_configured_user_agents(
    config: updater.Config, tmp_path: Path, compressed: bool
) -> None:
    requests_seen: list[tuple[str, str | None]] = []
    catalogue: list[dict[str, object]] = []
    payload = b"a,b\n1,2\n"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlsplit(self.path)
            requests_seen.append((path.path, self.headers.get("User-Agent")))
            if path.path == "/catalogue":
                offset = parse_qs(path.query)["offset"][0]
                body = json.dumps(
                    {"success": True, "result": catalogue if offset == "0" else []}
                ).encode()
            else:
                body = {
                    "/selected": payload,
                    "/broken": b"a,b\n1,2\n3,4,5\n",
                    "/missing": b"a,b\n1,2\n",
                }[path.path]
            self.send_response(404 if path.path == "/missing" else 200)
            if compressed:
                body = gzip.compress(body)
                self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            """Suppress fixture access logs."""

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        base = f"http://127.0.0.1:{server.server_port}"
        catalogue.append(
            {
                "id": "dataset",
                "name": "catalogue-slug",
                "title": {"de": "Example"},
                "resources": [
                    {"id": "1", "format": "CSV", "download_url": base + "/selected"},
                    {"id": "2", "format": "CSV", "url": base + "/other"},
                ],
            }
        )
        config = replace(
            config,
            metadata_link=base + "/catalogue",
            user_agent_generator="Generator-Test/1",
            user_agent_python="Notebook-Test/1",
            user_agent_r="R-Test/1",
        )
        thread = Thread(target=server.serve_forever)
        thread.start()
        try:
            updater.run_update(config)
            assert requests_seen == [("/catalogue", "Generator-Test/1")] * 2
            manifest = json.loads((config.temp_prefix / "manifest.json").read_text())
            entry = manifest["resource_files"][0]
            notebook = json.loads((config.temp_prefix / entry["python"]).read_text())
            cells = {cell.get("id"): "".join(cell["source"]) for cell in notebook["cells"]}
            checks = (
                cells["distributions"]
                + "\nassert df.to_dict('list') == {'a': [1], 'b': [2]}\n"
                + "with pytest.raises(requests.HTTPError) as error:\n"
                + f"    get_dataset({base + '/missing'!r})\n"
                + "assert error.value.response.status_code == 404\n"
                + "with pytest.raises(pd.errors.ParserError):\n"
                + f"    get_dataset({base + '/broken'!r}, sep=',')\n"
            )
            script = tmp_path / "load.py"
            script.write_text(
                "import pandas as pd\nimport pytest\nimport requests\n"
                "from unittest.mock import patch\n"
                + cells["csv-reader"]
                + """
responses = []
original_get = requests.get

def tracked_get(*args: object, **kwargs: object) -> requests.Response:
    assert kwargs['stream'] is True
    assert kwargs['timeout'] == HTTP_TIMEOUT
    result = original_get(*args, **kwargs)
    responses.append(result)
    return result

with patch('requests.get', side_effect=tracked_get):
"""
                + indent(checks, "    ")
                + "assert len(responses) == 3\n"
                + "assert all(result.raw.closed for result in responses)\n"
            )
            result = subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            assert result.returncode == 0, result.stderr
            assert requests_seen == [
                ("/catalogue", "Generator-Test/1"),
                ("/catalogue", "Generator-Test/1"),
                ("/selected", "Notebook-Test/1"),
                ("/missing", "Notebook-Test/1"),
                ("/broken", "Notebook-Test/1"),
            ]
            r_code = (config.temp_prefix / entry["r"]).read_text()
            assert 'USER_AGENT <- "R-Test/1"' in r_code
            assert 'headers = c("User-Agent" = USER_AGENT)' in r_code
        finally:
            server.shutdown()
            thread.join(timeout=5)
