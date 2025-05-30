# Multi-Agent System CrewAI: Automated Product Dataset Analysis

## Table of Contents

* [Overview](#overview)
* [Features](#features)
* [Architecture](#architecture)
* [Folder Structure](#folder-structure)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)
* [Examples](#examples)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Overview

This project implements a **multi-agent system** using CrewAI to automate the process of analyzing and extracting insights from a products dataset. Each agent specializes in a specific task—data ingestion, preprocessing, analysis, and reporting—working together to streamline large-scale product data workflows.

## Features

* **Modular agents** for ingesting, cleaning, and analyzing product data
* **Scalable architecture** leveraging CrewAI’s orchestrator
* **Automated reporting** with summary statistics and visualizations
* **Configurable pipelines** via environment settings

## Architecture

```
+----------------+     +------------------+     +-----------------+     +---------------+
| Ingestion      | --> | Preprocessing    | --> | Analysis Agent  | --> | Reporting     |
| Agent          |     | Agent            |     | (ML & Stats)    |     | Agent         |
+----------------+     +------------------+     +-----------------+     +---------------+
                  \____________________________________________^            
```

1. **Ingestion Agent**: Reads raw datasets (CSV, JSON, database).
2. **Preprocessing Agent**: Cleans, normalizes fields, handles missing values.
3. **Analysis Agent**: Applies statistical measures and machine learning models.
4. **Reporting Agent**: Generates summaries, exports charts, and writes output files.

## Folder Structure

```
├── env/                  # Environment configuration and dependencies
│   └── requirements.txt  # Python dependencies
├── ai_analysis/          # AI scripts and Jupyter notebooks
│   └── ...               # Individual analysis modules
├── README.md             # Project overview and instructions
└── .gitignore            # Files and directories to ignore
```

## Prerequisites

* Python 3.8 or higher
* Git
* pip (Python package manager)

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/chakorabdellatif/Multi-Agent_System_CrewAI_Automate-Analyzing_Products_Dataset.git
   cd Multi-Agent_System_CrewAI_Automate-Analyzing_Products_Dataset
   ```
2. **Create a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate   # Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r env/requirements.txt
   ```

## Configuration

* Copy `.env.example` to `.env` in the `env/` folder and fill in any required variables (e.g., data paths, API keys).

## Usage

1. **Run the ingestion pipeline**

   ```bash
   python ai_analysis/ingest_data.py --config env/.env
   ```
2. **Execute the preprocessing step**

   ```bash
   python ai_analysis/preprocess_data.py --config env/.env
   ```
3. **Run the analysis agent**

   ```bash
   python ai_analysis/analyze_data.py --config env/.env
   ```
4. **Generate the report**

   ```bash
   python ai_analysis/generate_report.py --output reports/summary.pdf
   ```

## Examples

Example commands and sample outputs are provided in the `ai_analysis/examples/` folder.

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m "Add YourFeature"`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

**Students**:

* Oussama ELHADJI – [oussousselhadji@gmail.com](mailto:oussousselhadji@gmail.com)
* Abdellatif CHAKOR – [abdellatifchakor09@gmail.com](mailto:abdellatifchakor09@gmail.com)

**Supervised by**: Bentaleb Asmae
