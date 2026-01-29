"""
Updater script for generating starter code files from opendata.swiss datasets.

Fetches dataset metadata from CKAN API, filters for CSV distributions,
and generates Python notebooks and R Markdown files with starter code.
"""

from __future__ import annotations

import json
import logging
import re
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import numpy as np
import requests
import yaml
from bs4 import BeautifulSoup as bs4
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Mapping

warnings.filterwarnings("ignore", category=FutureWarning)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# CONFIGURATION -------------------------------------------------------------- #


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"

    with config_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


# Load config at module level
CONFIG = load_config()

# Derived constants from config
PROVIDER = CONFIG["provider"]["name"]
PROVIDER_LINK = CONFIG["provider"]["link"]
BASELINK_DATAPORTAL = CONFIG["provider"]["dataportal_base"]
CKAN_API_LINK = CONFIG["provider"]["ckan_api"]
LANGUAGE = CONFIG["provider"]["language"]

GITHUB_ACCOUNT = CONFIG["github"]["account"]
REPO_NAME = CONFIG["github"]["repo_name"]
REPO_BRANCH = CONFIG["github"]["branch"]
REPO_RMARKDOWN_OUTPUT = CONFIG["github"]["rmarkdown_output"]
REPO_PYTHON_OUTPUT = CONFIG["github"]["python_output"]

TEMP_PREFIX = Path(CONFIG["paths"]["temp_prefix"])
TEMPLATE_FOLDER = Path(CONFIG["paths"]["template_folder"])
TEMPLATE_README = CONFIG["paths"]["templates"]["readme"]
TEMPLATE_HEADER = CONFIG["paths"]["templates"]["header"]
TEMPLATE_PYTHON = CONFIG["paths"]["templates"]["python"]
TEMPLATE_RMARKDOWN = CONFIG["paths"]["templates"]["rmarkdown"]

API_LIMIT = CONFIG["api"]["limit"]
API_SLEEP = CONFIG["api"]["sleep_seconds"]
API_TIMEOUT = CONFIG["api"]["timeout_seconds"]

TITLE_MAX_CHARS = CONFIG["display"]["title_max_chars"]
SORT_TABLE_BY = f"{CONFIG['display']['sort_by']}.{LANGUAGE}"

# Build keys with language suffix
KEYS_DATASET = [
    "publisher.name",
    f"organization.display_name.{LANGUAGE}",
    "organization.url",
    "maintainer",
    "maintainer_email",
    f"keywords.{LANGUAGE}",
    "issued",
    "metadata_created",
    "metadata_modified",
]
KEYS_DISTRIBUTIONS = CONFIG["keys_distributions"]
REDUCED_FEATURESET = CONFIG["reduced_featureset"]


def get_today_date() -> str:
    """Get today's date as YYYY-MM-DD string."""
    return datetime.today().strftime("%Y-%m-%d")


def get_today_datetime() -> str:
    """Get current datetime as YYYY-MM-DD HH:MM:SS string."""
    return datetime.today().strftime("%Y-%m-%d %H:%M:%S")


# FUNCTIONS ------------------------------------------------------------------ #


def get_full_package_list(
    limit: int = API_LIMIT, sleep: int = API_SLEEP
) -> pd.DataFrame:
    """Get full package list from CKAN API.

    Args:
        limit: Number of packages to fetch per request.
        sleep: Seconds to wait between requests.

    Returns:
        DataFrame containing all package data.

    Raises:
        requests.RequestException: If API requests fail.
    """
    offset = 0
    frames: list[pd.DataFrame] = []

    while True:
        logger.info(f"{offset} packages retrieved.")
        url = f"{CKAN_API_LINK}?limit={limit}&offset={offset}"

        try:
            response = requests.get(url, timeout=API_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse API response: {e}")
            raise

        if not data.get("result"):
            break

        frame = pd.DataFrame(pd.json_normalize(data["result"]))
        frames.append(frame)
        offset += limit
        time.sleep(sleep)

    if not frames:
        return pd.DataFrame()

    result = pd.concat(frames, ignore_index=True)
    return result


def has_csv_distribution(dists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter package resources to keep only CSV entries.

    Args:
        dists: List of distribution dictionaries.

    Returns:
        List of CSV distributions (empty if none found).
    """
    return [x for x in dists if x.get("format", "") == "CSV"]


def filter_csv(data: pd.DataFrame) -> pd.DataFrame:
    """Remove all datasets that have no CSV distribution.

    Args:
        data: DataFrame with resources column.

    Returns:
        Filtered DataFrame with only CSV-containing datasets.
    """
    data = data.copy()
    data["resources"] = data["resources"].apply(has_csv_distribution)
    # Filter out empty lists
    data = data[data["resources"].apply(len) > 0]
    return data.reset_index(drop=True)


def clean_features(data: pd.DataFrame) -> pd.DataFrame:
    """Clean various features in the dataset.

    Args:
        data: DataFrame with dataset metadata.

    Returns:
        Cleaned DataFrame.
    """
    data = data.copy()

    # Reduce tags to tag names.
    data["tags"] = data["tags"].apply(lambda x: [tag["name"] for tag in x])

    # Replace empty urls with NA message.
    mask = data["organization.url"] == ""
    data.loc[mask, "organization.url"] = "None provided"

    # Fill missing titles: try target language -> english -> french
    title_col = f"title.{LANGUAGE}"
    data[title_col] = (
        data[title_col]
        .replace("", np.nan)
        .fillna(data["title.en"].replace("", np.nan))
        .fillna(data["title.fr"].replace("", np.nan))
        .fillna("")
    )

    # Remove HTML tags from description.
    desc_col = f"description.{LANGUAGE}"
    data[desc_col] = data[desc_col].apply(
        lambda x: bs4(x, "html.parser").text if x else ""
    )

    # Strip whitespace from title.
    data[title_col] = data[title_col].str.strip()

    return data


def _build_metadata_string(row: pd.Series, keys: list[str]) -> str:
    """Build metadata markdown string from row data."""
    return "".join(f"- **{k.capitalize()}** `{row[k]}`\n" for k in keys)


def _build_contact_string(contact_points: list[dict[str, Any]]) -> str:
    """Extract contact information from contact_points."""
    if not contact_points:
        return "No contact information provided."

    values = [v for v in contact_points[0].values() if v and v != {}]
    if not values:
        return "No contact information provided."

    return " | ".join(str(v) for v in values)


def _process_distribution(dist: dict[str, Any], keys: list[str]) -> tuple[str, str]:
    """Process a single distribution and return metadata and download link.

    Args:
        dist: Distribution dictionary.
        keys: Keys to extract for metadata.

    Returns:
        Tuple of (metadata_string, download_url).
    """
    # Handle description - can be string or dict with language keys
    description = dist.get("description", "")
    if isinstance(description, dict):
        desc_text = description.get(LANGUAGE)
        if desc_text:
            # Remove line breaks that break comment blocks
            description = re.sub(r"[\n\r]+", " ", desc_text)
        else:
            description = ""

    # Build metadata string
    dist_copy = dist.copy()
    dist_copy["description"] = description
    md_lines = [f"# {k.capitalize():<25}: {dist_copy.get(k)}\n" for k in keys]
    metadata = "".join(md_lines)

    # Get download URL (fallback to url if download_url missing)
    download_url = dist.get("download_url") or dist.get("url", "")

    return metadata, download_url


def prepare_data_for_codebooks(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare metadata from catalogue to create code files.

    Args:
        data: DataFrame with dataset metadata.

    Returns:
        Prepared DataFrame with additional columns for code generation.
    """
    data = data.copy()

    # Add new columns for prepared data
    data["metadata"] = None
    data["contact"] = None
    data["distributions"] = None
    data["distribution_links"] = None

    for idx in tqdm(data.index, desc="Preparing codebooks"):
        # Build metadata string
        data.at[idx, "metadata"] = _build_metadata_string(data.loc[idx], KEYS_DATASET)

        # Build contact string
        data.at[idx, "contact"] = _build_contact_string(data.loc[idx, "contact_points"])

        # Process distributions
        dist_metadata = []
        dist_links = []
        for dist in data.loc[idx, "resources"]:
            metadata, url = _process_distribution(dist, KEYS_DISTRIBUTIONS)
            dist_metadata.append(metadata)
            dist_links.append(url)

        data.at[idx, "distributions"] = dist_metadata
        data.at[idx, "distribution_links"] = dist_links

    # Sort values for table
    data = data.sort_values(SORT_TABLE_BY).reset_index(drop=True)

    return data[REDUCED_FEATURESET]


def _render_template(template: str, replacements: Mapping[str, str]) -> str:
    """Render a template by replacing placeholders.

    Args:
        template: Template string with {{ PLACEHOLDER }} markers.
        replacements: Dictionary mapping placeholder names to values.

    Returns:
        Rendered template string.
    """
    result = template
    for key, value in replacements.items():
        result = result.replace(f"{{{{ {key} }}}}", str(value))
    return result


def _sanitize_text(text: str) -> str:
    """Sanitize text for use in templates (escape quotes, backslashes)."""
    text = re.sub('"', "'", text)
    text = re.sub(r"\\", "|", text)
    return text


def create_python_notebooks(data: pd.DataFrame) -> None:
    """Create Jupyter Notebooks with Python starter code.

    Args:
        data: Prepared DataFrame with dataset metadata.
    """
    # Read template once outside loop
    template_path = TEMPLATE_FOLDER / TEMPLATE_PYTHON
    template_content = template_path.read_text(encoding="utf-8")

    output_dir = TEMP_PREFIX / REPO_PYTHON_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(data.index, desc="Creating Python notebooks"):
        row = data.loc[idx]

        title = _sanitize_text(row[f"title.{LANGUAGE}"])
        description = _sanitize_text(row[f"description.{LANGUAGE}"])
        org_name = f"organization.display_name.{LANGUAGE}"

        # Build datashop link for organization
        org_link = ""
        if row["url"]:
            org_link = f"[Direct link by {row[org_name]} for dataset]({row['url']})"

        replacements = {
            "PROVIDER": PROVIDER,
            "DATASET_TITLE": title,
            "DATASET_DESCRIPTION": description,
            "DATASET_IDENTIFIER": row["identifier"],
            "DATASET_METADATA": _sanitize_text(row["metadata"]),
            "DISTRIBUTION_COUNT": str(len(row["distributions"])),
            "DATASHOP_LINK_PROVIDER": f"[Direct link by {PROVIDER} for dataset]({BASELINK_DATAPORTAL}{row['name']})",
            "DATASHOP_LINK_ORGANIZATION": org_link,
            "CONTACT": row["contact"],
        }

        py_nb_str = _render_template(template_content, replacements)

        try:
            py_nb = json.loads(py_nb_str, strict=False)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse notebook template for {row['id']}: {e}")
            continue

        # Find code cell for dataset imports (id == "0")
        dist_cell_idx = None
        for id_cell, cell in enumerate(py_nb["cells"]):
            if cell.get("id") == "0":
                dist_cell_idx = id_cell
                break

        if dist_cell_idx is None:
            logger.warning(f"No distribution cell found for {row['id']}")
            continue

        # Build code block for all distributions
        code_lines = []
        for id_dist, (dist_meta, dist_link) in enumerate(
            zip(row["distributions"], row["distribution_links"])
        ):
            code = f"# Distribution {id_dist}\n{dist_meta}\ndf = get_dataset('{dist_link}')\n"
            code_lines.append("".join(f"{line}\n" for line in code.split("\n")))

        py_nb["cells"][dist_cell_idx]["source"] = "".join(code_lines)

        # Save to disk
        output_path = output_dir / f"{row['id']}.ipynb"
        output_path.write_text(json.dumps(py_nb), encoding="utf-8")


def create_rmarkdown(data: pd.DataFrame) -> None:
    """Create R Markdown files with R starter code.

    Args:
        data: Prepared DataFrame with dataset metadata.
    """
    # Read template once outside loop
    template_path = TEMPLATE_FOLDER / TEMPLATE_RMARKDOWN
    template_content = template_path.read_text(encoding="utf-8")

    output_dir = TEMP_PREFIX / REPO_RMARKDOWN_OUTPUT
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in tqdm(data.index, desc="Creating R Markdown files"):
        row = data.loc[idx]

        title = _sanitize_text(row[f"title.{LANGUAGE}"])
        description = _sanitize_text(row[f"description.{LANGUAGE}"])
        org_name = f"organization.display_name.{LANGUAGE}"

        # Build datashop link for organization
        org_link = ""
        if row["url"]:
            org_link = f"[Direct link by **{row[org_name]}** for dataset]({row['url']})"

        # Build distribution code blocks
        code_blocks = []
        for id_dist, (dist_meta, dist_link) in enumerate(
            zip(row["distributions"], row["distribution_links"])
        ):
            code_blocks.append(
                f"# Distribution {id_dist}\n{dist_meta}\ndf <- read_delim('{dist_link}')\n\n"
            )

        replacements = {
            "DOCUMENT_TITLE": f"Open Government Data, {PROVIDER}",
            "DATASET_TITLE": title,
            "TODAY_DATE": get_today_date(),
            "DATASET_IDENTIFIER": row["identifier"],
            "DATASET_DESCRIPTION": description,
            "DATASET_METADATA": row["metadata"],
            "CONTACT": row["contact"],
            "DISTRIBUTION_COUNT": str(len(row["distributions"])),
            "DATASHOP_LINK_PROVIDER": f"[Direct link by **{PROVIDER}** for dataset]({BASELINK_DATAPORTAL}{row['name']})",
            "DATASHOP_LINK_ORGANIZATION": org_link,
            "DISTRIBUTIONS": "".join(code_blocks),
        }

        rmd_content = _render_template(template_content, replacements)

        # Save to disk
        output_path = output_dir / f"{row['id']}.Rmd"
        output_path.write_text(rmd_content, encoding="utf-8")


def get_header(dataset_count: int) -> str:
    """Retrieve header template and populate with date and count of data records.

    Args:
        dataset_count: Number of datasets.

    Returns:
        Rendered header string.
    """
    template_path = TEMPLATE_FOLDER / TEMPLATE_HEADER
    template_content = template_path.read_text(encoding="utf-8")

    gh_page = f"https://{GITHUB_ACCOUNT}.github.io/{REPO_NAME}/"
    gh_link = f"https://www.github.com/{GITHUB_ACCOUNT}/{REPO_NAME}"

    replacements = {
        "GITHUB_PAGE": gh_page,
        "GITHUB_REPO": gh_link,
        "PROVIDER": PROVIDER,
        "DATA_PORTAL": PROVIDER_LINK,
        "DATASET_COUNT": str(dataset_count),
        "TODAY_DATE": get_today_datetime(),
    }

    return _render_template(template_content, replacements)


def create_readme(dataset_count: int) -> None:
    """Retrieve README template and populate with metadata.

    Args:
        dataset_count: Number of datasets.
    """
    template_path = TEMPLATE_FOLDER / TEMPLATE_README
    template_content = template_path.read_text(encoding="utf-8")

    gh_page = f"https://{GITHUB_ACCOUNT}.github.io/{REPO_NAME}/"

    replacements = {
        "PROVIDER": PROVIDER,
        "DATASET_COUNT": str(dataset_count),
        "DATA_PORTAL": PROVIDER_LINK,
        "GITHUB_PAGE": gh_page,
        "TODAY_DATE": get_today_datetime(),
    }

    readme_content = _render_template(template_content, replacements)

    output_path = TEMP_PREFIX / "README.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(readme_content, encoding="utf-8")


def create_overview(data: pd.DataFrame, header: str) -> None:
    """Create index.md with link table.

    Args:
        data: Prepared DataFrame with dataset metadata.
        header: Header content to prepend.
    """
    baselink_r_gh = f"https://github.com/{GITHUB_ACCOUNT}/{REPO_NAME}/blob/{REPO_BRANCH}/{REPO_RMARKDOWN_OUTPUT}"
    baselink_py_gh = f"https://github.com/{GITHUB_ACCOUNT}/{REPO_NAME}/blob/{REPO_BRANCH}/{REPO_PYTHON_OUTPUT}"
    baselink_py_colab = f"https://githubtocolab.com/{GITHUB_ACCOUNT}/{REPO_NAME}/blob/{REPO_BRANCH}/{REPO_PYTHON_OUTPUT}"

    md_lines = [
        header,
        f"| Title (abbreviated to {TITLE_MAX_CHARS} chars) | Python Colab | Python GitHub | R GitHub |\n",
        "| :-- | :-- | :-- | :-- |\n",
    ]

    for idx in tqdm(data.index, desc="Creating overview"):
        # Remove square brackets from title (breaks markdown links)
        title_clean = (
            data.loc[idx, f"title.{LANGUAGE}"].replace("[", " ").replace("]", " ")
        )
        if len(title_clean) > TITLE_MAX_CHARS:
            title_clean = title_clean[:TITLE_MAX_CHARS] + "…"

        ds_link = f"{BASELINK_DATAPORTAL}{data.loc[idx, 'name']}"
        filename = data.loc[idx, "id"]

        r_gh_link = f"[R GitHub]({baselink_r_gh}{filename}.Rmd)"
        py_gh_link = f"[Python GitHub]({baselink_py_gh}{filename}.ipynb)"
        py_colab_link = f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({baselink_py_colab}{filename}.ipynb)"

        md_lines.append(
            f"| [{title_clean}]({ds_link}) | {py_colab_link} | {py_gh_link} | {r_gh_link} |\n"
        )

    output_path = TEMP_PREFIX / "index.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(md_lines), encoding="utf-8")


# MAIN ----------------------------------------------------------------------- #


def main() -> None:
    """Main entry point for the updater script."""
    logger.info("Fetching package list from CKAN API...")
    all_packages = get_full_package_list()

    if all_packages.empty:
        logger.error("No packages retrieved from API")
        return

    logger.info(f"Total packages retrieved: {len(all_packages)}")

    logger.info("Filtering for CSV distributions...")
    df = filter_csv(all_packages)
    logger.info(f"Datasets with CSV: {len(df)}")

    logger.info("Cleaning features...")
    df = clean_features(df)

    logger.info("Preparing data for codebooks...")
    df = prepare_data_for_codebooks(df)

    logger.info("Creating Python notebooks...")
    create_python_notebooks(df)

    logger.info("Creating R Markdown files...")
    create_rmarkdown(df)

    logger.info("Creating README and overview...")
    header = get_header(dataset_count=len(df))
    create_readme(dataset_count=len(df))
    create_overview(df, header)

    logger.info(f"Done! Generated code for {len(df)} datasets.")


if __name__ == "__main__":
    main()
