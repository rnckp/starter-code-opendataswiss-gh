# Starter code generator for opendata.swiss

Generate Python notebooks and R Markdown files for CSV resources on opendata.swiss. Each file loads one resource and provides metadata, summaries and plots to help users explore the data.

For ready-to-use notebooks, browse [starter-code_opendataswiss](https://github.com/rnckp/starter-code_opendataswiss) or its [overview page](https://rnckp.github.io/starter-code_opendataswiss/). This repository maintains the generator.

## Run locally

With `uv` installed, run from the repository root. Python 3.12 or newer is required; the checkout and CI use Python 3.13.

```bash
uv sync --locked
uv run python updater.py
uv run python updater.py --verify-output
```

The generator fetches CKAN catalogue metadata and selects resources listed as CSV. It writes notebooks to `_work/02_python/`, R Markdown files to `_work/01_r-markdown/`, and resource tables to `_work/README.md` and `_work/index.md`. Data is downloaded when a notebook runs. Local generation does not publish anything.

The final command verifies the generated files and their checksums without downloading metadata or executing notebooks. Python notebooks can run in Colab or a local notebook environment; R Markdown files need a local R environment. Check the inferred CSV separator and column types before using the data for analysis.

## Output and filenames

Each CSV resource gets its own notebook and R Markdown file, named `<dataset-id>--<resource-id>`. CKAN dataset and resource IDs keep filenames stable when titles or URLs change. Resources without an ID use a hash of their download URL. Catalogue links use the dataset's CKAN name.

The README previews the first 50 resources and links to the full catalogue on GitHub Pages. Both tables put notebook links first and show dataset names in the second column, with resource labels only for datasets containing multiple CSV files. `display.readme_resource_limit` controls preview size; `display.title_max_chars` bounds labels in both tables. Full labels remain in notebooks. Resource labels use CKAN `name`, then `title`, then resource ID; labels such as “Csv” are catalogue metadata, not parsed descriptions.

This replaces the previous dataset-only filenames. Old direct notebook links must be updated to the resource links in the generated overview.

**Generated notebook directories are rebuilt on every run.** Keep hand-written files outside them, and run only one generator per output directory at a time.

Generation stages and verifies a complete release before replacing the output. It preserves files outside the notebook directories except the generated `README.md`, `index.md` and `manifest.json`. Empty or invalid catalogues and failed generation leave the previous output in place.

If replacement or rollback is interrupted, a sibling `._work.previous` backup may remain. Inspect and recover it before rerunning; generation refuses to overwrite that backup. A custom output root uses `.<output-root-name>.previous`.

## Customize

Edit [config.yaml](config.yaml) to change the API, language, output paths, metadata fields or published repository links. Edit the files in [\_templates/](./_templates/) to change notebook and overview content, preserving their template markers.

To use a different configuration:

```bash
uv run python updater.py --config path/to/config.yaml
```

The default configuration is `config.yaml` in the working directory. Relative configuration, output and template paths use the working directory, including when using `--config`. Notebook output directories are relative to the output root; template filenames are relative to the template folder.

## Publish automatically

[The GitHub Actions workflow](.github/workflows/autoupdater.yml) publishes from `main` on pushes, weekly, or when triggered manually. Formatting, lint, tests and output verification must pass before publication. Pull requests run quality checks without publishing.

To set up publishing:

1. Set the target account, repository and branch in both `config.yaml` and the workflow's publishing step. Keep its `source-directory` aligned with the configured output root.
2. Add a `PAT` Actions secret with write access to the target repository. Use a repository dedicated to generated output and review `_work/LICENSE.md` before publishing.
3. Enable GitHub Pages in the target repository, deploying from the target branch's root. The generated `index.md` is the entry page.

## Development

After `uv sync --locked`, run:

```bash
uv run ruff format updater.py tests
uv run ruff check updater.py tests
uv run pytest
```

Tests exercise generated Python and check generated R content. They do not execute R Markdown; validate R template changes in an R environment.

## Related projects

- [Original version for OGD Canton of Zurich](https://github.com/rnckp/starter-code-ktzh) (archived)
- [Current version OGD Canton of Zurich](https://github.com/openZH/starter-code-openzh-generator)
- [Starter Code OGD City of Zurich](https://github.com/opendatazurich/starter-code)
- The [OGD Thurgau project](https://github.com/ogdtg/starter-code-ogdtg) provides another implementation.

Ideas and contributions are welcome through issues and pull requests.
