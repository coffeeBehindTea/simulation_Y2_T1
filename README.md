# Simulation_Y2_T1 Repository

## Overview
This repository contains the **Modelling and Simulation** project for COMP0212.  It implements a hybrid deterministic–stochastic model of student lecture attendance, fits a first‑order ODE baseline, adds categorical and weather covariates, runs Monte‑Carlo simulations with RFID sensor noise, and evaluates the model on a hold‑out set (Weeks 9‑10).

## Directory Structure
```
simulation_Y2_T1/
├─ data/                     # Cleaned CSVs used for modelling
│   ├─ combined_courses.csv   # Feature matrix (10 columns)
│   └─ …                     # Original raw CSVs (attendance & weather)
├─ docs/                     # Technical documentation and retrospectives
│   ├─ attendance_factor_retrospective.en.md
│   └─ attendance_factor_retrospective.zh-CN.md
├─ figures/                  # Publication‑quality PNGs and CSV tables
│   ├─ stochastic_simulation_monte_carlo.png
│   ├─ parameter_sensitivity_ode.png
│   ├─ simulation_performance_summary.csv
│   └─ (future figures – see *Additional Figures* section)
├─ scripts/                  # All runnable Python scripts (new folder)
│   ├─ stochastic_simulation_and_analysis.py
│   ├─ plot_baseline_ode.py   # Generates a clean ODE baseline plot
│   └─ …                     # Any helper scripts
├─ team66.tex                # Complete LaTeX report (ready for Overleaf)
├─ README.md                 # This file (updated)
└─ .gitignore                # Ignored artefacts (e.g. __pycache__)
```

## Data
- **`RAI_attendance_with_weather.csv`** – cleaned attendance + matched weather.
- **`combined_courses.csv`** – the feature matrix used by the modelling scripts (course one‑hots, weather, exam‑distance).
- All CSVs are version‑controlled; raw source files are retained in the repository for reproducibility.

## Scripts
| Script | Purpose |
|--------|---------|
| `scripts/stochastic_simulation_and_analysis.py` | End‑to‑end Monte‑Carlo simulation, sensor‑noise modelling, hold‑out statistics, sensitivity plots. |
| `scripts/plot_baseline_ode.py` | Generates a standalone plot of the fitted ODE baseline (Figure A). |
| `compare_single_factors.py` | Single‑factor fitting & evaluation (already in repo). |
| `compare_factor_pairs.py` | Two‑factor fitting & evaluation (already in repo). |
| `sweep_exam_thresholds.py` | Exam‑cutoff sensitivity sweep (already in repo). |

All scripts can be executed from the project root, e.g.:
```bash
python -m pip install -r requirements.txt   # install numpy, pandas, scipy, matplotlib
python scripts/stochastic_simulation_and_analysis.py
python scripts/plot_baseline_ode.py
```

## Results (figures already in `figures/`)
- **Monte‑Carlo attendance simulation** – `stochastic_simulation_monte_carlo.png`
- **Parameter sensitivity** – `parameter_sensitivity_ode.png`
- **Hold‑out performance summary** – `simulation_performance_summary.csv`

## Additional Figures (recommended for the report)
| Figure | Suggested filename | Description |
|--------|-------------------|-------------|
| **Figure A – ODE baseline** | `baseline_ode.png` | Plot of the fitted ODE solution $A(t)$ (Equation \ref{eq:ode_solution}) over the full 10‑week term, with training (Weeks 1‑8) and hold‑out (Weeks 9‑10) shading. |
| **Figure B – Factor weight bar chart** | `factor_weights.png` | Horizontal bar plot of the learned weights for the four course one‑hots and the two weather factors (sunny, temperature). |
| **Figure C – Exam‑cutoff heatmap** | `exam_cutoff_heatmap.png` | Heat‑map of MAE / RMSE for each exam‑cutoff $n=1\dots10$ (Table 3) to give a visual overview of the sensitivity sweep. |
| **Figure D – Residuals vs. prediction** | `residuals_scatter.png` | Scatter of observed attendance rate vs. predicted probability for the ODE + Course model (including 95 % CI). |

The LaTeX report already contains placeholders (`Insert system diagram here`, `Insert simulation result plot here`). Replace those placeholders with the PNG files listed above.

## How to Run the Project
1. **Clone the repo** (already done).
2. **Create a virtual environment** and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # or .\.venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```
3. **Execute the main analysis**:
   ```bash
   python scripts/stochastic_simulation_and_analysis.py
   ```
   This will (re)generate `figures/*.png` and `figures/simulation_performance_summary.csv`.
4. **Generate the ODE baseline figure** (optional, for the report):
   ```bash
   python scripts/plot_baseline_ode.py
   ```
   The output `baseline_ode.png` will be saved under `figures/`.
5. **Open `team66.tex` in Overleaf**, upload the PNG files from `figures/`, and compile.

## Contributing
- **Branch‑per‑feature**: create a new branch for any additional analysis or figure.
- **Commit messages** should follow the conventional‑commit style already used (e.g. `feat: add baseline ODE plot`).
- **Pull‑request**: open a PR against `main` and request a review from the team.

## License
This project is released under the MIT License – you are free to use, modify, and distribute the code for academic purposes.

---
*Last updated: 2026‑11‑22*