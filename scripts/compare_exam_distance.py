"""Add exam distance to the combined course table and test it as a single factor."""

from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from compare_single_factors import (
    BASELINE,
    FEATURE_COLUMNS,
    ROOT,
    fit_adjustment,
    fit_baseline,
    load_sessions,
    plot_scores,
)


# The default export keeps the last six sessions before each exam active.
DISTANCE = "classes_until_exam"
NO_EFFECT_FROM = 7
# These names distinguish distance-dependent and constant corrections within the exam window.
EXAM_MODEL = "Baseline + Exam distance"
OFFSET_ONLY = "Baseline + Constant offset"
# The supplied exam dates are keyed by course and shared across all sessions during ranking.
EXAM_DATES = {
    "COMP0209": ["2026-01-29", "2026-02-12", "2026-03-05", "2026-03-19"],
    "COMP0206": ["2026-01-30", "2026-02-13", "2026-03-06", "2026-03-27"],
}


def exam_distances(sessions, no_effect_from=NO_EFFECT_FROM):
    """Return session countdowns, using zero at or above the supplied exclusive cutoff."""
    # The exam schedule uses a noon start for both courses. Rank every
    # retained session on the shared timetable against that same schedule.
    exams = pd.DatetimeIndex(sorted(day for days in EXAM_DATES.values() for day in days))
    exams += pd.Timedelta(hours=12)
    # Use each session's start and leave the countdown missing if there is no later exam.
    start = sessions["session_time"]
    distance = pd.Series(pd.NA, index=sessions.index, dtype="Int64", name=DISTANCE)
    # Search to the right so the selected exam starts strictly after the session.
    positions = exams.searchsorted(start, side="right")
    known = positions < len(exams)
    # Group known sessions by their next exam before assigning reverse ranks.
    exam_groups = pd.Series(exams.take(positions[known]), index=start.index[known])
    # Count the current session plus later sessions before the exam. Classes
    # with the same start time at the same distance have the same rank.
    ranks = start.loc[known].groupby(exam_groups).rank(method="min", ascending=False).astype(int)
    distance.loc[known] = ranks.where(ranks.lt(no_effect_from), 0)
    return distance


def write_feature_column(sessions, distance):
    """Add or replace the exam countdown in the feature CSV, preserving the other fields."""
    # The existing export defines row order. Derived values are keyed by its shared features.
    path = ROOT / "data" / "combined_courses.csv"
    exported = pd.read_csv(path)
    values = sessions[FEATURE_COLUMNS].assign(**{DISTANCE: distance}).drop_duplicates()
    if values.duplicated(FEATURE_COLUMNS).any():
        raise ValueError("The feature export cannot distinguish classes with different exam distances.")
    # The merge indicator detects missing keys even when a matched countdown is null.
    updated = exported.drop(columns=DISTANCE, errors="ignore").merge(
        values, on=FEATURE_COLUMNS, how="left", validate="many_to_one", indicator=True)
    if not updated.pop("_merge").eq("both").all():
        raise ValueError("Some exported sessions could not be matched.")
    pd.testing.assert_frame_equal(updated[exported.columns.drop(DISTANCE, errors="ignore")],
                                  exported.drop(columns=DISTANCE, errors="ignore"))
    if updated[DISTANCE].value_counts(dropna=False).to_dict() != distance.value_counts(dropna=False).to_dict():
        raise ValueError("Exam distance counts changed during the export.")
    # The temporary CSV stays beside its destination, allowing replacement after verification.
    with NamedTemporaryFile(suffix=".csv", dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        updated.to_csv(temp, index=False, encoding="utf-8-sig", lineterminator="\n")
        pd.testing.assert_frame_equal(pd.read_csv(temp, dtype={DISTANCE: "Int64"}), updated)
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def exam_distance_design(distance, train):
    """Build the gated exam design and return its columns, active mask, and training mean."""
    # Only positive countdowns activate the exam factors, and only training rows set the mean.
    affected = distance.fillna(0).gt(0).to_numpy(dtype=bool)
    fit_rows = train.to_numpy() & affected
    center = float(distance.loc[fit_rows].mean()) if fit_rows.any() else 0.0
    # Use floating-point values for centering, with both terms zero when the gate is closed.
    numeric = distance.to_numpy(dtype=float, na_value=np.nan)
    design = np.column_stack([affected, np.where(affected, numeric - center, 0)])
    return design, affected, center


def predict_split(sessions, distance, train, test):
    """Fit exam variants on train, then return test probabilities and fitted weight details."""
    # All variants share one baseline fitted without test outcomes.
    _, baseline = fit_baseline(sessions, train)
    baseline_logit = logit(np.clip(baseline, 1e-6, 1 - 1e-6))
    design, affected, center = exam_distance_design(distance, train)
    fit_rows = train.to_numpy() & affected
    # y and n are observed and eligible counts. Only active training rows estimate corrections.
    y, n = (sessions[name].to_numpy() for name in ("Attended", "Total"))
    # Initialize both variants to the baseline if no training session activates the exam window.
    offset = np.zeros(1)
    beta = np.zeros(2)
    adjusted = baseline.copy()
    calibrated = baseline.copy()
    if fit_rows.any():
        # Estimate the constant first. A single distinct distance supports only this constant.
        offset = fit_adjustment(design[fit_rows, :1], baseline_logit[fit_rows], y[fit_rows], n[fit_rows])
        if distance.loc[fit_rows].nunique() > 1:
            beta = fit_adjustment(design[fit_rows], baseline_logit[fit_rows], y[fit_rows], n[fit_rows])
        else:
            beta[0] = offset[0]
        # Apply the estimated corrections only where the countdown activates the exam factor.
        adjusted[affected] = expit(baseline_logit[affected] + design[affected] @ beta)
        calibrated[affected] = expit(baseline_logit[affected] + offset[0])
    # Keep observed proportions and the window mask aligned with the test predictions.
    predictions = pd.DataFrame({
        "observed": y[test] / n[test],
        "within_exam_window": affected[test],
        BASELINE: baseline[test],
        EXAM_MODEL: adjusted[test],
        OFFSET_ONLY: calibrated[test],
    })
    return predictions, {"train_within_window": int(fit_rows.sum()), "distance_center": center,
                         "constant": float(beta[0]), "per_remaining_class": float(beta[1]),
                         "constant_only": float(offset[0])}


def scores(predictions, scope):
    """Return MAE and RMSE in percentage points for each model in the named test scope."""
    # Accumulate one result row per model using the same observed sessions.
    rows = []
    for model in (BASELINE, EXAM_MODEL, OFFSET_ONLY):
        errors = 100 * (predictions[model] - predictions["observed"])
        rows.append({"scope": scope, "sessions": len(predictions), "model": model,
                     "MAE_pp": errors.abs().mean(), "RMSE_pp": np.sqrt(errors.pow(2).mean())})
    return pd.DataFrame(rows)


def main():
    """Update the exam feature and export its chronological holdout and rolling comparisons."""
    # Start with validated sessions in time order and assign their common exam schedule.
    sessions = load_sessions()
    distance = exam_distances(sessions)
    weeks = sessions["Teaching Week Number"]
    # The fixed split matches the single-factor and pair comparisons.
    holdout, fit = predict_split(sessions, distance, weeks.between(1, 8), weeks.between(9, 10))
    affected = holdout.loc[holdout["within_exam_window"]]
    # Each rolling test week uses only outcomes from earlier weeks.
    rolling = pd.concat([predict_split(sessions, distance, weeks.lt(w), weeks.eq(w))[0]
                         for w in range(5, 11)], ignore_index=True)
    # Score each split on all sessions and on just the rows where exam distance can change predictions.
    rolling_affected = rolling.loc[rolling["within_exam_window"]]
    full_scores = scores(holdout, "Weeks 9–10: all sessions")
    affected_scores = scores(affected, "Weeks 9–10: 1–6 sessions before an exam")
    rolling_scores = scores(rolling, "Rolling Weeks 5–10: all sessions")
    rolling_affected_scores = scores(rolling_affected, "Rolling Weeks 5–10: 1–6 sessions before an exam")
    report = pd.concat([full_scores, affected_scores, rolling_scores, rolling_affected_scores],
                       ignore_index=True)
    if not np.isfinite(report[["MAE_pp", "RMSE_pp"]]).all().all():
        raise ValueError("Each test scope must have sessions and finite scores.")

    # Save the updated feature and the two figure views beside the full score report.
    write_feature_column(sessions, distance)
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    plot_scores(full_scores, figures / "exam_distance_holdout_errors.png",
                f"Shared exam schedule: Weeks 9–10, all sessions (n={len(holdout)})")
    plot_scores(rolling_scores, figures / "exam_distance_rolling_errors.png",
                f"Shared exam schedule: rolling Weeks 5–10, all sessions (n={len(rolling)})")
    report.to_csv(figures / "exam_distance_scores.csv", index=False, float_format="%.8f")
    print(f"Exam distance counts: {distance.value_counts(dropna=False).sort_index().to_dict()}")
    # Print coverage by course and the actual countdowns on the fixed holdout.
    in_window = distance.fillna(0).gt(0)
    print(f"Within window by course: {sessions.loc[in_window, 'Course'].value_counts().to_dict()}")
    print(f"Fit: {fit}")
    test = weeks.between(9, 10)
    print(sessions.loc[test, ["Course", "Date String", "Start Time"]]
          .assign(**{"remaining": distance.loc[test]}).to_string(index=False))
    print(report.to_string(index=False, float_format="{:.6f}".format))


if __name__ == "__main__":
    main()
