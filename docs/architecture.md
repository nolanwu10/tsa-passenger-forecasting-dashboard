# Data Architecture

## Design Intent

The project is being rebuilt around data clarity before model iteration. Modeling should depend on stable, documented datasets rather than ad hoc root-level CSVs and scripts.

## Data Zones

`data/raw`

Original source data. Files here should be treated as immutable snapshots. The primary raw input is TSA passenger volume data.

`data/external`

Third-party enrichment data such as weather, macroeconomic, search-trend, or calendar data. These sources should keep their own schemas until joined into an interim or processed table.

`data/interim`

Scratch or partially joined datasets. These are allowed to be regenerated and should not be treated as canonical.

`data/processed`

Analysis-ready datasets. Anything in this folder should have documented columns, validation checks, and a clear lineage from raw/external inputs.

`artifacts`

Generated outputs such as trained model files, plots, and performance reports. These are not source data.

`legacy`

Previous model scripts and generated caches preserved for reference. New work should not import from `legacy`.

## Code Boundaries

`src/tsa_project/config.py`

Central path definitions. Code should import paths from here instead of hardcoding project-relative filenames.

`src/tsa_project/schemas.py`

Dataset contracts: expected columns, semantic types, and required fields.

`src/tsa_project/datasets.py`

Loaders and lightweight profiling helpers. These functions should be safe to run before modeling dependencies are installed.

`src/tsa_project/quality.py`

Validation checks for row counts, date parsing, nulls, duplicates, and required columns.

## Proposed Next Milestones

1. Define the canonical raw TSA dataset contract.
2. Replace legacy scraping with a clean ingestion module that writes timestamped raw snapshots.
3. Define a deterministic processed dataset builder.
4. Add validation tests before reintroducing model training.
5. Add modeling only after the data contract is stable.

