# Contributing to SyWeDaG

## Development setup

```bash
git clone git@github.com:IAAA-Lab/SyWeDaG.git
cd SyWeDaG
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env   # fill in AEMET_API_KEY if you plan to use the AEMET source
```

Run the app:

```bash
cd src
streamlit run main.py
```

Run the offline demo (no API key or network needed):

```bash
python examples/run_offline_demo.py
```

## Running tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

Tests live under `tests/`, mirroring the `src/` package layout. Network calls
to AEMET and Open-Meteo are mocked; no test should require internet access or
a real API key. Every pull request runs the same suite on Linux, Windows, and
macOS across the supported Python versions (see `.github/workflows/ci.yml`),
plus a smoke test that launches the packaged app and confirms it responds.

When you add a new generator, adjuster, or data-source method, add tests that
check the actual output values (ranges, invariants like `Tmin <= Tmean <=
Tmax`, totals matching predictions within tolerance), not just that the code
runs without raising.

## Code conventions

- Layered architecture: `ui/` never talks to `database/` or `data_sources/`
  directly, it goes through `application/`. See `CLAUDE.md` for the full
  module map.
- Adding a new weather data source: implement `BaseWeatherSource` in
  `src/data_sources/`, register it in `source_selector.get_data_source_instance`,
  and add an entry to `config/config.json`.
- Error handling follows the pattern already used across the codebase:
  parsing helpers (`utils/data_parsing.py`) fail soft and return `None` on
  malformed input rather than raising; `ValueError` is used for
  programmer/configuration errors (missing required arguments, invalid
  source names); I/O boundaries (HTTP requests, file reads) are wrapped in
  `try`/`except` and logged via `utils.system_utils.safe_print` rather than
  crashing the app.
- All file paths (config, assets, `.env`, the SQLite database) must be
  resolved through `utils.system_utils.get_resource_path()`.
- No hardcoded relative paths; the app must keep working both under
  `streamlit run` and inside the PyInstaller desktop build.

## Versioning

SyWeDaG follows [Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

- MAJOR: incompatible changes to the generated-data schema, the exported ZIP
  package format, or the database schema that require a migration.
- MINOR: new data sources, new correction methods, new UI features, backward
  compatible.
- PATCH: bug fixes, documentation, dependency bumps.

The current version lives in `src/_version.py`. Every user-facing change
should get an entry in `CHANGELOG.md` under `[Unreleased]`; the maintainers
move it under a numbered release when cutting one.

## Commit messages

This repository loosely follows `type(scope): short description`, for
example `fix(hourly_interpolator): fix start and end pseudo extremes` or
`feat(data_sources): add Open-Meteo source`. Check `git log` for more
examples.

## Reporting issues

Open a GitHub issue with:

- What you did and what you expected to happen.
- The actual output (error message, screenshot, or generated data that looks
  wrong).
- Your OS, Python version, and whether you're running from source or the
  packaged desktop build.

## Pull requests

- Keep PRs focused on one change.
- Add or update tests for any behavior change.
- Update `CHANGELOG.md` and, if relevant, `README.md`.
- Make sure `pytest` passes locally before opening the PR; CI will run the
  full OS/Python matrix automatically.
