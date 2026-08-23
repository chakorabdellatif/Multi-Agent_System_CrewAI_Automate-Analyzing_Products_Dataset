# Multi-Agent System — CrewAI Product Dataset Analysis

A 3-agent [CrewAI](https://www.crewai.com/) pipeline that cleans a fashion-products CSV, computes descriptive statistics, and generates matplotlib/seaborn chart images — built as a school project (ENIAD, supervised by Bentaleb Asmae).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10--3.12-blue.svg)

---

## Overview

The `ai_analysis/` package defines a sequential CrewAI crew of **3 agents** that run over `ai_analysis/data/Products.csv` (a fashion-retail dataset: brand, category, price, rating, color, size):

1. **Data Preparation Analyst** (`data_preparer`) — loads the CSV, drops missing values/duplicates, and removes price/rating outliers with the IQR method.
2. **Pattern Detection Scientist** (`pattern_analyst`) — computes top-rated brands, average price by category, most common color per category, and product count by brand; writes each result to `knowledge/patterns/`.
3. **Visualization Specialist** (`visualization_engineer`) — renders the same analyses as bar charts and a price/rating scatter plot to `knowledge/plots/`.

This is what is actually implemented and runnable in `ai_analysis/src/ai_analysis/`. An earlier version of this README described a different, 4-stage "Ingestion → Preprocessing → Analysis → Reporting" pipeline with scripts (`ingest_data.py`, `preprocess_data.py`, `analyze_data.py`, `generate_report.py`) and an `ai_analysis/examples/` folder — **none of those files exist in the repository**. This README replaces that description with the real architecture below.

## Features

Implemented as CrewAI custom tools (`ai_analysis/src/ai_analysis/tools/`):

- **Data preparation**: `LoadDataTool`, `DataCleanerTool` (NA/duplicate removal), `OutlierRemoverTool` (IQR-based).
- **Pattern analysis**: `TopRatedBrandsTool`, `AvgPriceByCategoryTool`, `TopColorPerCategoryTool`, `ProductCountByBrandTool` — each writes a `.txt` report to `knowledge/patterns/`.
- **Visualization**: `BarChartAvgPriceTool`, `ScatterPlotTool`, `BarChartTopRatedBrandsTool`, `BarChartProductCountByBrandTool` — each saves a `.png` to `knowledge/plots/` via matplotlib/seaborn.
- Real, non-empty sample outputs are committed under `ai_analysis/knowledge/patterns/*.txt` and `ai_analysis/knowledge/plots/*.png`, confirming the pipeline has actually been run end-to-end.

## Architecture

```
data/Products.csv
        │
        ▼
┌────────────────────┐   data_cleaning_task
│  data_preparer       │   (load → drop NA/dupes → remove outliers)
└─────────┬───────────┘
          ▼  data/cleaned_products.csv
┌────────────────────┐   pattern_analysis_task
│  pattern_analyst     │   (top brands, avg price/category, top color, counts)
└─────────┬───────────┘
          ▼  knowledge/patterns/*.txt
┌────────────────────┐   visualization_task
│  visualization_engineer│ (bar charts, scatter plot)
└─────────┬───────────┘
          ▼  knowledge/plots/*.png
```

Process is `Process.sequential` (`crew.py`): each task depends on the previous one's output, orchestrated by CrewAI's `Crew.kickoff()`.

## Tech Stack

| Purpose | Technology |
|---|---|
| Agent orchestration | [CrewAI](https://www.crewai.com/) (`crewai[tools]`) |
| Data handling | pandas |
| Charts | matplotlib, seaborn |
| Dependency/build | [uv](https://docs.astral.sh/uv/) + hatchling (`pyproject.toml`, `uv.lock`) |
| LLM | Requires an LLM API key configured for CrewAI (e.g. OpenAI) — CrewAI agents use an LLM to reason about tool calls |

## Getting Started

### Prerequisites

- Python >=3.10, <3.13
- [uv](https://docs.astral.sh/uv/)
- An LLM API key (e.g. `OPENAI_API_KEY`) for CrewAI's agent reasoning

### Installation

```bash
git clone https://github.com/chakorabdellatif/Multi-Agent_System_CrewAI_Automate-Analyzing_Products_Dataset.git
cd Multi-Agent_System_CrewAI_Automate-Analyzing_Products_Dataset/ai_analysis

pip install uv
crewai install     # resolves and installs from pyproject.toml / uv.lock
```

Create a `.env` file in `ai_analysis/` with your LLM key, e.g.:

```env
OPENAI_API_KEY=your_api_key_here
```

### Run

```bash
cd ai_analysis
crewai run
```

This kicks off the 3-agent crew against `data/Products.csv` and writes cleaned data, pattern reports, and charts as described above.

### Known Issues

- **Missing dependencies**: `visualization_tools.py` imports `matplotlib` and `seaborn` directly, but neither package is declared in `pyproject.toml` nor present in `uv.lock`. A clean `crewai install` will **not** install them, so the visualization step will fail with `ModuleNotFoundError` until they are installed manually (`uv add matplotlib seaborn`) or added to `pyproject.toml`.
- **`report.md`** at the repository root of `ai_analysis/` is leftover boilerplate from the CrewAI project template (a generic report about "Recent Developments in AI LLMs") — it is not output from this project's crew and should be disregarded or removed.
- **Committed virtual environment**: the top-level `env/` directory (~40,000 files) is a full Python virtual environment that was committed to git, rather than the project's actual dependencies being reproducible via `pyproject.toml`/`uv.lock` alone. It bloats the repository substantially. A root `.gitignore` has been added by this change to prevent further additions; removing the already-tracked `env/` history is a separate, larger cleanup left to the maintainer.

## Testing / CI

`.github/workflows/python-package.yml` (GitHub's default "Python package" template) already exists in this repository and was left unmodified. As configured it runs `flake8` and `pytest` over the entire repository with no path exclusions, which — given the committed `env/` virtual environment above — means it would lint/test tens of thousands of vendored files rather than just this project's code. Its GitHub Actions run history shows **zero recorded runs**, so its current behavior on this codebase is unverified. There is no project-specific automated test suite.

## Project Structure

```
Multi-Agent_System_CrewAI_Automate-Analyzing_Products_Dataset/
├── ai_analysis/
│   ├── data/
│   │   ├── Products.csv            # Raw input dataset
│   │   └── cleaned_products.csv    # Output of data_cleaning_task
│   ├── knowledge/
│   │   ├── patterns/                # Text reports from pattern_analyst
│   │   └── plots/                   # PNG charts from visualization_engineer
│   ├── src/ai_analysis/
│   │   ├── crew.py                  # Agent + task + crew definitions
│   │   ├── main.py                  # run/train/replay/test entry points
│   │   ├── config/agents.yaml       # Agent roles/goals/backstories
│   │   ├── config/tasks.yaml        # Task descriptions/dependencies
│   │   └── tools/                   # Custom CrewAI tools (see Features)
│   ├── report.md                    # Leftover template boilerplate (see Known Issues)
│   ├── pyproject.toml
│   └── uv.lock
├── env/                              # Committed virtualenv (see Known Issues)
├── LICENSE
└── CHANGELOG.md
```

## Documentation

A draft GitHub Wiki lives in [`docs/wiki-draft/`](docs/wiki-draft/) (Home, Getting Started, Architecture, FAQ) — GitHub wikis can't be reviewed via pull request, so this is provided for manual copy-in after merge.

## Changelog

See [CHANGELOG.md](CHANGELOG.md).

## Security

No committed secrets (API keys, tokens, credentials) were found in this project's own source files or in the vendored `env/` directory beyond generic library code and public CA certificate bundles. `ai_analysis/.gitignore` already excludes `.env`.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE).

## Contributors

- [Abdellatif Chakor](https://github.com/chakorabdellatif) (chakorabdellatif)
- [Oussama ELHADJI](https://github.com/Bosaj) (Bosaj)

Supervised by Bentaleb Asmae.
