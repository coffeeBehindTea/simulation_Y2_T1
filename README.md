# COMP0212 Course Attendance Modeling

This project uses card-tap attendance records from four courses to compare a shared ordinary differential equation (ODE) baseline with models that account for course, time of day, sunny conditions, feels-like temperature, and the number of classes until the next exam.

**Technical retrospective** · [English](docs/attendance_factor_retrospective.en.md) · [Simplified Chinese](docs/attendance_factor_retrospective.zh-CN.md)

Both versions document the data cleaning rules, model equations, implementation, full results, and reproduction commands.

## Completed Work

- **Data preparation.** Cleaned and retained 94 class sessions, matched weather at each session's start, and standardized the feature column names in English. [Combined feature table](data/combined_courses.csv)
- **Shared baseline.** Fitted the ODE using 76 sessions from teaching Weeks 1–8 and evaluated it on 18 sessions from Weeks 9–10.
- **Factor comparison.** Evaluated five single-factor models and all ten two-factor combinations on the same holdout. [Single-factor predictions](figures/single_factor_all_holdout_predictions.png) · [Two-factor predictions](figures/factor_pair_holdout_predictions.png) · [Model weights](figures/factor_model_weights.json)
- **Exam cutoff sweep.** Used the shared exam schedule to test cutoffs from 1 to 10 classes, setting distances at or above each cutoff to zero. Compared results on the fixed holdout and in week-by-week testing. [Scores](figures/exam_threshold_scores.csv) · [Comparison chart](figures/exam_threshold_errors.png)

On the fixed holdout, **ODE + course** had the lowest mean absolute error (MAE) at 10.238 percentage points, compared with 13.016 for the pure baseline. The best pair, course + feels-like temperature, reached 10.264. In the exam cutoff sweep, the pure baseline had the lowest week-by-week MAE.
