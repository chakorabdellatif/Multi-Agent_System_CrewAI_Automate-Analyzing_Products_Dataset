# Getting Started

## Prerequisites

- Python >=3.10, <3.13
- [uv](https://docs.astral.sh/uv/)
- An LLM API key (e.g. `OPENAI_API_KEY`) — CrewAI agents use an LLM to decide which tool to call

## Install

```bash
git clone https://github.com/chakorabdellatif/Multi-Agent_System_CrewAI_Automate-Analyzing_Products_Dataset.git
cd Multi-Agent_System_CrewAI_Automate-Analyzing_Products_Dataset/ai_analysis

pip install uv
crewai install
```

`crewai install` resolves dependencies from `pyproject.toml` / `uv.lock`.

**Important**: `matplotlib` and `seaborn` are imported by the visualization tools but are **not** declared as project dependencies. Install them manually before running the visualization step:

```bash
uv add matplotlib seaborn
```

## Configure

Create a `.env` file inside `ai_analysis/`:

```env
OPENAI_API_KEY=your_api_key_here
```

## Run

```bash
cd ai_analysis
crewai run
```

This runs the 3 tasks in sequence:
1. Clean `data/Products.csv` → `data/cleaned_products.csv`
2. Compute patterns → `knowledge/patterns/*.txt`
3. Generate charts → `knowledge/plots/*.png`

## Other Entry Points

`pyproject.toml` also exposes:

```bash
crewai run        # same as `run_crew`
train <n> <file>  # AiAnalysis().crew().train(...)
replay <task_id>
test <n> <eval_llm>
```

(`train`/`replay`/`test` are thin wrappers around CrewAI's own crew methods, defined in `main.py`.)

## Known Gaps

See the [FAQ](FAQ.md) and the README's "Known Issues" section — most notably the missing matplotlib/seaborn dependency declaration and the unrelated `report.md` leftover from the CrewAI project template.
