# Generator decisions

- Generation, templates and publication validation adapt the openZH generator at revision `3a6746e`. CKAN handling stays specific to opendata.swiss, including pagination, localized fields, resource URLs and catalogue links.
- Generate one Python/R file pair per CSV resource. CKAN dataset `id` and resource `id` determine filenames; dataset `name` determines the catalogue link. A URL hash identifies resources without an ID. This replaces legacy dataset-only paths.
- CKAN download URLs can omit the scheme on `www.` links or contain surrounding whitespace. Normalize those to HTTPS and trimmed URLs before applying HTTP URL validation; other invalid URLs still stop the release.
- Stage and verify the complete release before replacement. Managed notebook directories are rebuilt to remove stale resources; other output files are preserved except generated overview files and the manifest. A failed rollback leaves a sibling backup for recovery.
