"""Fit all two-factor attendance models on the shared ODE baseline."""

from itertools import combinations
import json

from compare_exam_distance import DISTANCE, NO_EFFECT_FROM, exam_distance_design, exam_distances
from compare_single_factors import (
    BASELINE, ROOT, fit_adjustment, fit_baseline, load_sessions, plot_predictions, single_factor_designs,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit, logit


# Use the same exam factor name in the score table, plots, and weight export.
EXAM = "Exam distance"
# MAE sets the primary rank, with RMSE breaking ties. Both use percentage points.
METRICS = ("MAE_pp", "RMSE_pp")
# Short plot labels keep long factor names legible across two columns.
LABELS = {
    "Course": "Course",
    "Time of day": "Time of day",
    "Sunny": "Sunny",
    "Feels-like temperature": "Feels-like temp.",
    EXAM: EXAM,
}


def prediction_scores(actual, predicted):
    """Return MAE and RMSE in percentage points for aligned rate and probability arrays."""
    # Residuals are predicted minus observed. Both metrics remove the sign.
    errors = 100 * (predicted - actual)
    return np.array([np.abs(errors).mean(), np.sqrt(np.mean(errors ** 2))])


def plot_pairs(pairs, baseline, best_single, output, test_count):
    """Save pair scores with pure and best single-factor references to the output PNG.

    The pairs table is already ranked. Baseline and best_single contain MAE
    and RMSE in that order, and test_count is the shared holdout size.
    """
    # Use the same labels and row positions in both metric panels.
    labels = [f"{LABELS[a]} + {LABELS[b]}" for a, b in zip(pairs["factor_1"], pairs["factor_2"])]
    y = np.arange(len(pairs))
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), sharey=True, layout="constrained")
    # Compare each metric with its own references. Green marks an improvement over the best single factor.
    for ax, metric, base, single in zip(axes, METRICS, baseline, best_single):
        values = pairs[metric].to_numpy()
        colors = np.where(values < single, "#008270", "#3E6A9C")
        ax.barh(y, values, color=colors, height=.63)
        ax.axvline(base, color="#4D5763", lw=1.3, ls="--", label="Pure baseline")
        ax.axvline(single, color="#25404E", lw=1.4, ls=":", label="Best single factor")
        # Place one error label beside each horizontal bar.
        for pos, value in zip(y, values):
            ax.text(value + .14, pos, f"{value:.2f}", va="center", fontsize=9)
        ax.set_xlim(0, max(base, values.max()) * 1.14)
        ax.set_yticks(y, labels)
        ax.set_xlabel(f"{metric.removesuffix('_pp')} (percentage points)")
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    # The two reference lines share one legend across the panels.
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle(f"Two-factor attendance predictions in Weeks 9–10 (n={test_count})", fontsize=14)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    """Fit all ten additive pairs and export weights, scores, and prediction plots."""
    sessions = load_sessions()
    # Use the same chronological split as the earlier single-factor comparisons.
    week = sessions["Teaching Week Number"]
    train = week.between(1, 8).to_numpy()
    test = week.between(9, 10).to_numpy()
    if set(week) != set(range(1, 11)) or not (train | test).all():
        raise ValueError("Expected sessions in teaching Weeks 1–10.")

    # Freeze ODE parameters and use the session-level baseline logit in all factor fits.
    baseline_params, base = fit_baseline(sessions, train)
    base_logit = logit(np.clip(base, 1e-6, 1 - 1e-6))
    # Keep design columns and coefficient names in the same order for weight export.
    single, temp_center = single_factor_designs(sessions, train)
    designs = {name.removeprefix("Baseline + "): design for name, (design, _) in single.items()}
    terms = {name.removeprefix("Baseline + "): names for name, (_, names) in single.items()}
    # Encode exam distance as a gated offset and a centered slope within the window.
    distance = exam_distances(sessions)
    exam_terms, affected, exam_center = exam_distance_design(distance, pd.Series(train, index=sessions.index))
    designs[EXAM] = exam_terms
    terms[EXAM] = ["offset_at_training_mean", "per_remaining_class"]
    # Verify that the saved exam column agrees with the feature reconstructed from session starts.
    exported_distances = pd.read_csv(ROOT / "data" / "combined_courses.csv", dtype={DISTANCE: "Int64"})
    pd.testing.assert_frame_equal(
        exported_distances.sort_values(list(exported_distances.columns)).reset_index(drop=True),
        sessions[exported_distances.columns.drop(DISTANCE)].assign(**{DISTANCE: distance})
        .sort_values(list(exported_distances.columns)).reset_index(drop=True),
        check_dtype=False, check_exact=True,
    )
    if not np.all(exam_terms[~affected] == 0):
        raise ValueError("The exam factor must contribute zero outside the exam window.")

    # Fit on attended and eligible counts, then score against observed holdout proportions.
    y, total = (sessions[column].to_numpy() for column in ("Attended", "Total"))
    actual = y[test] / total[test]

    def predict_design(design, fit_rows):
        """Fit design on the selected rows and return holdout probabilities and coefficients."""
        # Estimate one weight per design column and transform predictions back to probabilities.
        beta = fit_adjustment(design[fit_rows], base_logit[fit_rows], y[fit_rows], total[fit_rows])
        predicted = expit(base_logit[test] + design[test] @ beta)
        if not np.isfinite(predicted).all():
            raise ValueError("Invalid holdout probabilities.")
        return predicted, beta

    # Recompute the individual references on exactly the dates used for the pairs.
    baseline = prediction_scores(actual, base[test])
    single_fits = {
        name: predict_design(design, train & affected if name == EXAM else train)
        for name, design in designs.items()
    }
    # Keep probabilities and named weights separately so plots and JSON use the same fits.
    single_predictions = {name: pred for name, (pred, _) in single_fits.items()}
    single_weights = {
        name: {name: dict(zip(terms[name], beta.tolist()))}
        for name, (_, beta) in single_fits.items()
    }
    singles = {name: prediction_scores(actual, pred) for name, pred in single_predictions.items()}
    # The best single-factor reference is selected independently for MAE and RMSE.
    best_single = np.min(list(singles.values()), axis=0)
    print(f"Train/test sessions: {train.sum()} / {test.sum()}")
    print("Reference scores (MAE pp, RMSE pp)")
    print(BASELINE, baseline)
    for name, score in singles.items():
        print(name, score)

    # Keep one results row, prediction array, and coefficient map for each pair.
    results = []
    pair_predictions = {}
    pair_weights = {}
    for a, b in combinations(designs, 2):
        # The non-exam factor blocks already span a constant. For two of
        # those blocks, keeping one common intercept avoids collinearity.
        # Both gated exam columns are identifiable and must stay together.
        right = designs[b] if b == EXAM else designs[b][:, 1:]
        # Fit both factor blocks jointly and split the solution back into named terms.
        combined = np.column_stack([designs[a], right])
        predicted, beta = predict_design(combined, train)
        pair_predictions[a, b] = predicted
        left_beta, right_beta = np.split(beta, [designs[a].shape[1]])
        if b != EXAM:
            # Explicitly store the omitted reference as zero so a consumer
            # can evaluate models with the same factor formulas.
            right_beta = np.r_[0, right_beta]
        pair_weights[f"{a} + {b}"] = {
            a: dict(zip(terms[a], left_beta.tolist())),
            b: dict(zip(terms[b], right_beta.tolist())),
        }
        # Negative deltas indicate improvement against the corresponding reference.
        score = prediction_scores(actual, predicted)
        row = {"factor_1": a, "factor_2": b}
        for j, metric in enumerate(METRICS):
            row[metric] = score[j]
            row[f"{metric}_change_vs_baseline"] = score[j] - baseline[j]
            row[f"{metric}_change_vs_better_component"] = score[j] - min(singles[a][j], singles[b][j])
            row[f"{metric}_change_vs_best_single"] = score[j] - best_single[j]
        results.append(row)

    # Use the same ordering for the table, error bars, and pair prediction panels.
    pairs = pd.DataFrame(results).sort_values(list(METRICS), ignore_index=True)
    if len(pairs) != 10 or pairs.duplicated(["factor_1", "factor_2"]).any():
        raise ValueError("Expected ten distinct pairs of the five factors.")
    # Store the results and the model formula together for later reconstruction.
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    weight_file = {
        "training_weeks": list(range(1, 9)),
        "training_sessions": int(train.sum()),
        "exam_window_training_sessions": int((train & affected).sum()),
        "holdout_weeks": [9, 10],
        "holdout_sessions": int(test.sum()),
        "baseline": {
            "equation": "A(t) = A_eq + (A0 - A_eq) * exp(-k*t), t = teaching_week - 1",
            **dict(zip(("A0", "A_eq", "k"), baseline_params.tolist())),
        },
        "prediction": "sigmoid(logit(clamp(A(t), 0.000001, 0.999999)) + sum of factor adjustments)",
        "factor_definitions": {
            "Course": {"adjustment": "Use the coefficient for the active course.",
                       "one_hot_order": terms["Course"]},
            "Time of day": {"adjustment": "Use the coefficient for morning or afternoon.",
                            "afternoon_starts_at": "12:00"},
            "Sunny": {"adjustment": "Use the coefficient for sunny or not_sunny.",
                      "sunny_weather_codes": [0, 1]},
            "Feels-like temperature": {
                "adjustment": "offset_at_training_mean + per_degree_c * (temperature_c - training_mean_c)",
                "training_mean_c": float(temp_center),
            },
            EXAM: {
                "adjustment": ("offset_at_training_mean + per_remaining_class * "
                               "(remaining_classes - training_mean_remaining_classes) inside the exam window; "
                               "zero otherwise"),
                "active_remaining_classes": [1, NO_EFFECT_FROM - 1],
                "training_mean_remaining_classes": exam_center,
            },
        },
        "pair_coding": ("When the second factor also spans a constant, its reference coefficient is set "
                        "to zero and the common intercept is represented by the first factor."),
        "single_factor_weights": single_weights,
        "two_factor_weights": pair_weights,
    }
    (figures / "factor_model_weights.json").write_text(
        json.dumps(weight_file, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    plot_pairs(pairs, baseline, best_single, figures / "factor_pair_holdout_errors.png", test.sum())
    pairs.to_csv(figures / "factor_pair_scores.csv", index=False, float_format="%.8f")
    # Keep predictions aligned with the same chronological holdout sessions in both charts.
    holdout = sessions.loc[test]
    plot_predictions(holdout, actual, {
        BASELINE: base[test],
        **{f"Baseline + {LABELS[name]}": pred for name, pred in single_predictions.items()},
    }, figures / "single_factor_all_holdout_predictions.png",
        "Baseline and single-factor models in Weeks 9–10")
    plot_predictions(holdout, actual, {
        f"{i}. {LABELS[a]} + {LABELS[b]}": pair_predictions[a, b]
        for i, (a, b) in enumerate(zip(pairs["factor_1"], pairs["factor_2"]), start=1)
    }, figures / "factor_pair_holdout_predictions.png",
        "Two-factor models in Weeks 9–10, ranked by MAE")
    print(pairs.to_string(index=False, float_format="{:.6f}".format))


if __name__ == "__main__":
    main()
