# FAQ

**Why does `crewai run` fail on the visualization step?**
`matplotlib` and `seaborn` are used by the visualization tools but are not declared as dependencies of the `ai_analysis` package. Install them manually: `uv add matplotlib seaborn` (see Getting Started).

**Is `report.md` the output of this project?**
No. It's leftover boilerplate from the CrewAI project scaffold (`crewai create crew`) describing "Recent Developments in AI LLMs" — unrelated to the fashion-products analysis this repository actually performs. The real outputs are in `ai_analysis/knowledge/patterns/` and `ai_analysis/knowledge/plots/`.

**What does the crew actually produce?**
A cleaned CSV (`data/cleaned_products.csv`), four text reports (top-rated brands, average price by category, top color per category, product count by brand), and four PNG charts — all committed as real, non-empty sample outputs in the repository so you can see what a run looks like without executing it yourself.

**Is there a test suite?**
No project-specific tests exist. A generic GitHub Actions workflow (`.github/workflows/python-package.yml`) is present but has never actually run (zero recorded runs), and as configured would scan the entire repository, including the large committed `env/` directory.

**Why is there a 40,000-file `env/` folder in the repo?**
It's a fully committed Python virtual environment, most likely committed by accident (no root `.gitignore` existed to exclude it). It's unrelated to the project's actual dependency management, which uses `pyproject.toml` and `uv.lock` inside `ai_analysis/`. A `.gitignore` was added to stop it from growing further; removing the existing tracked files is a separate cleanup.

**Do I need an OpenAI key specifically?**
CrewAI needs some LLM configured for the agents to reason about tool calls; `OPENAI_API_KEY` is the default expected by the scaffold, but CrewAI supports other providers if you configure them.
