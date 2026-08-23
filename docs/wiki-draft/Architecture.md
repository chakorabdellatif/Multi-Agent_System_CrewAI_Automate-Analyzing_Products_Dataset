# Architecture

## Agents (`ai_analysis/src/ai_analysis/crew.py`, `config/agents.yaml`)

| Agent | Role | Tools |
|---|---|---|
| `data_preparer` | Data Preparation Analyst | `LoadDataTool`, `DataCleanerTool`, `OutlierRemoverTool` |
| `pattern_analyst` | Pattern Detection Scientist | `TopRatedBrandsTool`, `AvgPriceByCategoryTool`, `TopColorPerCategoryTool`, `ProductCountByBrandTool` |
| `visualization_engineer` | Visualization Specialist | `BarChartAvgPriceTool`, `ScatterPlotTool`, `BarChartTopRatedBrandsTool`, `BarChartProductCountByBrandTool` |

## Tasks (`config/tasks.yaml`)

Run via `Process.sequential`, each depending on the previous task's output:

1. `data_cleaning_task` (agent: `data_preparer`) — loads `data/Products.csv`, drops NA/duplicate rows, removes price/rating outliers (IQR method), writes `data/cleaned_products.csv`.
2. `pattern_analysis_task` (agent: `pattern_analyst`, depends on step 1) — computes top-rated brands, average price by category, top color per category, and product counts by brand; writes `.txt` reports to `knowledge/patterns/`.
3. `visualization_task` (agent: `visualization_engineer`, depends on step 2) — renders the same analyses as bar charts and a price/rating scatter plot to `knowledge/plots/`.

## Data Flow

```
data/Products.csv
   → (DataManager: load, clean, remove outliers)
   → data/cleaned_products.csv
   → (pattern_analyst tools, via DataManager.get_df)
   → knowledge/patterns/*.txt
   → (visualization_engineer tools, via DataManager.get_df)
   → knowledge/plots/*.png
```

`DataManager` (`tools/data_manager.py`) is a class-level in-memory cache of the raw/processed DataFrame, shared across tools within a single crew run, and persisted to `data/cleaned_products.csv` on every update.

## Known Gaps

- `matplotlib` and `seaborn` are imported directly by `visualization_tools.py` but are not declared in `pyproject.toml`/`uv.lock`, so a fresh install does not include them.
- `ai_analysis/report.md` is leftover CrewAI project-template boilerplate (an "AI LLMs" report) unrelated to this project's actual output — it predates the real analysis code and was never replaced.
- The repository root also contains a fully committed Python virtual environment (`env/`, ~40,000 files) that is unrelated to `ai_analysis/`'s own `pyproject.toml`/`uv.lock`-based dependency management.
