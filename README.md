# ML Projects 

[![CI](https://github.com/JeffreyZhu2323/ml-projects/actions/workflows/ci.yml/badge.svg)](https://github.com/JeffreyZhu2323/ml-projects/actions/workflows/ci.yml)

Curated ML mini-projects built in PyTorch, focused on clean training/evaluation pipelines and reproducible structure. Each project lives in `projects/` with its own README, code in `src/`, and outputs saved to `results/` and `reports/`.
**Python:** 3.10.0

## Projects
- **[Linear Regression](./projects/linear_regression/)** — regression baseline + evaluation metrics + ridge sweep + plots

## Setup

### Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
