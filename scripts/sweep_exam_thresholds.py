"""Compare exclusive exam countdown cutoffs from 1 through 10 classes."""

from compare_exam_distance import EXAM_MODEL, NO_EFFECT_FROM, exam_distances, predict_split, scores
from compare_single_factors import ROOT, load_sessions

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# With cutoff n, only distances 1 through n - 1 contribute an exam adjustment.
THRESHOLDS = range(1, 11)


def summarize(predictions, scope):
    """Return exam-model scores and window coverage for one held-out set."""
    # Select the exam variant. The sweep's n=1 row already reports the pure baseline.
    metrics = scores(predictions, scope).set_index("model").loc[EXAM_MODEL]
    return {
        f"{scope}_sessions": int(metrics["sessions"]),
        f"{scope}_within_window": int(predictions["within_exam_window"].sum()),
        f"{scope}_MAE_pp": float(metrics["MAE_pp"]),
        f"{scope}_RMSE_pp": float(metrics["RMSE_pp"]),
    }


def plot_sweep(results, output):
    """Save the holdout and rolling errors as four aligned panels over the same cutoffs."""
    # The x coordinate is the first excluded countdown, so n=1 is the baseline.
    thresholds = results["no_effect_from"]
    labels = {
        "holdout": f"Weeks 9–10 holdout ({results['holdout_sessions'].iloc[0]} sessions)",
        "rolling": f"Weeks 5–10 rolling test ({results['rolling_sessions'].iloc[0]} sessions)",
    }
    # Use the same threshold axis in both splits, with one column per metric.
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.6), sharex=True, layout="constrained")
    for row, (scope, title) in enumerate(labels.items()):
        for column, metric in enumerate(("MAE", "RMSE")):
            # Each series includes the unadjusted baseline score at n=1.
            ax = axes[row, column]
            errors = results[f"{scope}_{metric}_pp"]
            best_index = errors.idxmin()
            baseline = float(errors.iloc[0])
            ax.axvline(NO_EFFECT_FROM, color="#8793A3", ls=":", lw=1.3,
                       label=f"Current cutoff n={NO_EFFECT_FROM}")
            ax.axhline(baseline, color="#556271", ls="--", lw=1.2, label="Pure baseline")
            ax.plot(thresholds, errors, "-o", color="#26598D", lw=1.7, ms=4.5,
                    label="Exam distance model")
            ax.scatter(thresholds.loc[best_index], errors.loc[best_index], s=62,
                       facecolors="white", edgecolors="#006D5B", lw=2, zorder=4, label="Lowest error")
            ax.set_title(f"{title} | {metric}", loc="left", fontsize=10)
            ax.set_ylabel("Percentage points")
            ax.set_xticks(list(THRESHOLDS))
            ax.grid(axis="y", alpha=.22)
            ax.spines[["top", "right"]].set_visible(False)
    # Label the shared x axes on the bottom panels, above the figure-wide legend.
    for ax in axes[1]:
        ax.set_xlabel("Cutoff n (distances of n or more are set to zero)")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Attendance prediction error by exam countdown cutoff", fontsize=14)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    """Refit the single exam factor at each cutoff and export the test comparison."""
    # The fixed split and expanding windows match the earlier exam-distance experiment.
    sessions = load_sessions()
    weeks = sessions["Teaching Week Number"]
    train = weeks.between(1, 8)
    test = weeks.between(9, 10)
    rolling_weeks = range(5, 11)

    # Refit each candidate independently and evaluate all held-out rows at each cutoff.
    rows = []
    for threshold in THRESHOLDS:
        distance = exam_distances(sessions, no_effect_from=threshold)
        holdout, weight = predict_split(sessions, distance, train, test)
        rolling = pd.concat(
            [predict_split(sessions, distance, weeks.lt(week), weeks.eq(week))[0]
             for week in rolling_weeks],
            ignore_index=True,
        )
        rows.append({
            "no_effect_from": threshold,
            "max_active_remaining_classes": threshold - 1,
            "training_within_window": weight["train_within_window"],
            **summarize(holdout, "holdout"),
            **summarize(rolling, "rolling"),
        })
    results = pd.DataFrame(rows)
    if not np.isfinite(results.to_numpy()).all():
        raise ValueError("All thresholds must have finite coverage and score results.")

    # Keep the sensitivity study separate from the previously published n=7 feature and scores.
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    results.to_csv(figures / "exam_threshold_scores.csv", index=False, float_format="%.8f")
    plot_sweep(results, figures / "exam_threshold_errors.png")
    print(results.to_string(index=False, float_format="{:.6f}".format))


if __name__ == "__main__":
    main()
