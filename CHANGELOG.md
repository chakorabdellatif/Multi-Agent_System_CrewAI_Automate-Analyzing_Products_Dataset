# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

There are no tagged releases in this repository yet, so all history is recorded under [Unreleased].

## [Unreleased]

### Added
- `README.md` rewritten to describe the actual 3-agent CrewAI pipeline (data preparation, pattern analysis, visualization) implemented in `ai_analysis/src/ai_analysis/`, replacing a previous description of a 4-stage pipeline and scripts (`ingest_data.py`, `preprocess_data.py`, `analyze_data.py`, `generate_report.py`, `ai_analysis/examples/`) that do not exist in this repository.
- Root `.gitignore` to prevent further growth of the committed virtual environment and stray `.env`/`__pycache__` files (the existing `env/` directory is left untouched — see README "Known Issues").
- `docs/wiki-draft/` with Home, Getting-Started, Architecture, and FAQ pages (draft content for the GitHub Wiki).

### Security
- No committed secrets (API keys, tokens, credentials) were found in this project's own source files. The vendored `env/` directory (a fully committed Python virtual environment, ~40,000 files) was checked for common credential file patterns (`.netrc`, `pip.conf`, `.pypirc`, `.env`, AWS/GCP credential files); the only matches are generic third-party library source code, public CA certificate bundles (`certifi`, `grpc`), and `.env.example` template files bundled inside the `embedchain` package itself — none are real secrets.

### Known Issues (documented, not fixed by this change)
- `visualization_tools.py` imports `matplotlib` and `seaborn` directly, but neither is declared in `pyproject.toml` or resolved in `uv.lock` — a clean install will not provide them.
- `ai_analysis/report.md` is unrelated leftover boilerplate from the CrewAI project template (a generic "AI LLMs" report), not real output of this project.
- The root `env/` directory is a fully committed Python virtual environment (~40,000 tracked files). Removing it from history is a larger change left to the maintainer; this PR only stops it from growing further via `.gitignore`.
- `.github/workflows/python-package.yml` already existed before this change and was left as-is; it has zero recorded runs on GitHub Actions and, as configured, would lint/test the entire repository including the committed `env/` directory.

### Changed
- No other existing files were modified. `LICENSE` (MIT, already correctly attributed to Oussama ELHADJI and Abdellatif CHAKOR) was left untouched.
