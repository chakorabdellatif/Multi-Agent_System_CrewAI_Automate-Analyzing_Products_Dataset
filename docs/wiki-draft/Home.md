# Multi-Agent System — CrewAI Product Dataset Analysis — Wiki

> **Draft note**: GitHub Wikis cannot be reviewed as part of a pull request, so this folder holds the drafted wiki content. If this PR is merged, please copy the contents of each page below into the repository's actual Wiki (Settings → Wiki, or the "Wiki" tab) manually.

A 3-agent [CrewAI](https://www.crewai.com/) pipeline that cleans a fashion-products dataset, computes descriptive statistics, and generates chart images.

## Pages

- [Getting Started](Getting-Started.md) — install and run the crew.
- [Architecture](Architecture.md) — the 3 agents, 3 tasks, and their tools.
- [FAQ](FAQ.md) — common questions, including known gaps.

## At a Glance

- **Agents**: Data Preparation Analyst, Pattern Detection Scientist, Visualization Specialist (`ai_analysis/src/ai_analysis/crew.py`).
- **Input**: `ai_analysis/data/Products.csv`.
- **Output**: `ai_analysis/knowledge/patterns/*.txt` and `ai_analysis/knowledge/plots/*.png`.
- **License**: MIT.
