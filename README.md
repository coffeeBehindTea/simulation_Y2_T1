# Simulation_Y2_T1 Repository

## Overview
This repository contains the **Modelling and Simulation** project for COMP0212 (UCL Year 2 Term 1). It implements a hybrid deterministic–stochastic model of student lecture attendance, fits a first‑order ODE baseline, adds categorical and environmental covariates (course one-hots, weather, time-of-day, exam distance), runs Monte‑Carlo simulations with RFID sensor noise, and evaluates the models on a hold‑out test set (Weeks 9–10).

## Directory Structure
```text
simulation_Y2_T1/
├── data/                                # Cleaned feature datasets & source data
│   ├── raw/                             # Original raw logs & exports (read-only)
│   │   ├── COMP0206_2025_t2_attendance(in).csv
│   │   ├── COMP0207  attendance data for 2025 - 2026.csv
│   │   ├── COMP0208 attendance monitoring data 2025 - 2026.csv
│   │   ├── COMP0209 attendance monitoring data 2025-2026.csv
│   │   ├── UCL_East_hourly_weather_raw.json
│   │   └── marshgate_sunny_2026-01-12_to_2026-03-27_weekdays.json
│   ├── COMP0206.csv ~ COMP0209.csv      # Cleaned attendance logs per module
│   ├── UCL_East_class_weather.csv       # Aligned weather data per class hour
│   ├── RAI_attendance_clean_96.csv      # Unified 96-session attendance record
│   ├── RAI_attendance_with_weather.csv  # Attendance matched with hourly weather
│   └── combined_courses.csv             # Primary feature matrix used for modeling
├── docs/                                # Technical documentation and handovers
│   ├── COMP0212_Attendance_Simulation_Project_Handover.docx
│   ├── attendance_factor_retrospective.en.md
│   └── attendance_factor_retrospective.zh-CN.md
├── figures/                             # Generated plots, metrics tables, and model parameters
│   ├── factor_model_weights.json        # Fitted model coefficients
│   ├── reported_model_parameters_*.json # Verification benchmark parameter exports
│   ├── *.csv                            # Model evaluation scores (MAE, RMSE)
│   └── *.png                            # High-resolution visualization plots
├── scripts/                             # Runnable simulation & analysis scripts
│   ├── compare_single_factors.py        # Single-factor fitting & hold-out evaluation
│   ├── compare_factor_pairs.py          # Factor pair combinatorial evaluation
│   ├── compare_exam_distance.py         # Exam countdown / proximity effect modeling
│   ├── sweep_exam_thresholds.py         # Exam proximity sensitivity window sweep
│   ├── stochastic_simulation_and_analysis.py # Monte Carlo simulation & sensor noise modeling
│   ├── ode_course_adverse_time.py       # ODE + Course + Adverse Weather + Time-of-Day
│   └── ode_course_rain_time.py          # ODE + Course + Rain + Time-of-Day
├── requirements.txt                     # Python package dependencies
├── .gitignore                           # Git ignore rules
└── README.md                            # Project overview and reproduction guide
```

## Data
- **`data/raw/`**: Raw attendance logs exported from the attendance tracking system and hourly weather records from Open-Meteo.
- **`data/RAI_attendance_with_weather.csv`**: Main cleaned dataset containing per-session card-tap attendance matched with local weather and time attributes.
- **`data/combined_courses.csv`**: Multiset feature table used for cross-model validation.

## Scripts Description
| Script | Purpose |
|--------|---------|
| `scripts/compare_single_factors.py` | Fits the baseline ODE ($A(t) = A_{eq} + (A_0 - A_{eq})e^{-kt}$) and tests single covariates (Course, Sunny, Feels-like Temp, Time-of-Day). |
| `scripts/compare_factor_pairs.py` | Evaluates all 2-factor combinations against the hold-out set and exports `figures/factor_model_weights.json`. |
| `scripts/compare_exam_distance.py` | Models attendance decay as an inverse function of remaining class count before scheduled exams. |
| `scripts/sweep_exam_thresholds.py` | Performs sensitivity analysis over exam window cutoffs ($n=1\dots10$). |
| `scripts/stochastic_simulation_and_analysis.py` | End-to-end Monte Carlo simulation ($X_i \sim \text{Binomial}(N_i, P_i)$), RFID sensor noise modeling, CI estimation, and parameter sensitivity analysis. |
| `scripts/ode_course_adverse_time.py` | Independent joint evaluation for adverse weather (rain > 1mm or wind > 20km/h) and afternoon time offsets. |
| `scripts/ode_course_rain_time.py` | Independent joint evaluation for rain occurrence and afternoon time offsets. |

## Quickstart

### 1. Environment Setup
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Running Simulations & Analyses
All scripts can be executed directly from the project root:

```bash
# 1. Single-factor comparisons & baseline fitting
python scripts/compare_single_factors.py

# 2. Factor-pair evaluations & weight export
python scripts/compare_factor_pairs.py

# 3. Exam countdown proximity evaluation
python scripts/compare_exam_distance.py

# 4. Exam window sensitivity sweep
python scripts/sweep_exam_thresholds.py

# 5. Stochastic Monte Carlo simulation & sensitivity analysis
python scripts/stochastic_simulation_and_analysis.py

# 6. Adverse weather & rain interaction models
python scripts/ode_course_adverse_time.py
python scripts/ode_course_rain_time.py
```

All plots and CSV metric summaries will be generated and saved to `figures/`.
