# Streamlit Setup Guide

This file explains how to run the RiskPlus Streamlit app on another computer.

## What you need

- Python 3.12 or newer
- `pip`
- A copy of this repository
- Internet access for the first install, unless the dependencies are already cached

## Recommended setup

### 1. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Install the app dependencies

Install the runtime packages used by Streamlit and the analytics engine:

```bash
pip install -r requirements.txt
```

Install the test tools if you want to run the test suite too:

```bash
pip install -r requirements-dev.txt
```

### 3. Start the app

Launch the Streamlit entry point from the project root:

```bash
streamlit run riskplus.py
```

If that does not work in your environment, use the virtual environment Python explicitly:

```bash
.venv/bin/streamlit run riskplus.py
```

## What gets installed

The app runtime currently depends on:

- `streamlit`
- `pandas`
- `numpy`
- `plotly`
- `scipy`
- `statsmodels`
- `openpyxl`

The developer/test dependency currently used by the repo is:

- `pytest`

## How to verify the install

Run the tests:

```bash
pytest
```

If you want the same validation used during development, you can also run:

```bash
python -m compileall .
```

## Common issues

- If `streamlit` is not found, make sure the virtual environment is activated.
- If Excel uploads fail, confirm `openpyxl` is installed.
- If tests fail because of missing packages, reinstall with `pip install -r requirements.txt` and `pip install -r requirements-dev.txt`.
- If the app opens but the page is blank, check the terminal for Python errors and confirm you are launching from the repository root.

## Notes for another computer

- Copy the full repository, not just `riskplus.py`.
- Keep the `riskplus_core/`, `riskplus_ui/`, `tests/`, and `test_data/` folders together.
- If you use a different Python version, re-create the virtual environment on that machine.
- The repo already separates runtime and dev dependencies, so installing both requirement files is the safest setup.
