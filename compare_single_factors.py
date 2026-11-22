"""Compare single-factor attendance models against the shared ODE baseline."""

from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, minimize
from scipy.special import expit, logit


# Resolve inputs and outputs relative to the project, regardless of the working directory.
ROOT = Path(__file__).resolve().parent
# Preserve the course order in both the one-hot matrix and the exported weights.
COURSES = ["COMP0206", "COMP0207", "COMP0208", "COMP0209"]
# Common CSV fields shared by the single-factor, exam-distance, and pair models.
FEATURE_COLUMNS = ["course_date", "morning", "afternoon", "sunny", "feels_like_temperature_c", *COURSES]
# Use a common label and palette for all comparison plots.
BASELINE = "Baseline"
MODEL_COLORS = ["#607086", "#2455A4", "#008270", "#8A5A12", "#784B91", "#AA4058"]


def load_sessions():
    """Read the cleaned attendance source and return validated sessions in time order."""
    # The feature export has no unique session key. Build the features beside
    # the outcomes in the source, then reconcile them with the feature export.
    sessions = pd.read_csv(ROOT / "RAI_attendance_with_weather.csv")
    # These fields identify each session and provide its outcome and matched weather.
    required = ["Class Id", "Course", "Date String", "Start Time", "weather_datetime",
                "Teaching Week Number", "Attended", "Total", "weather_code",
                "apparent_temperature"]
    if sessions[required].isna().any().any() or not sessions["Class Id"].is_unique:
        raise ValueError("The source has missing values or duplicate session IDs.")
    if not ((sessions["Total"] > 0) & sessions["Attended"].between(0, sessions["Total"])).all():
        raise ValueError("The source has invalid attendance counts.")
    if not sessions["Course"].isin(COURSES).all():
        raise ValueError("The source has an unknown course.")
    # Use the local class start time to validate the weather join.
    start = pd.to_datetime(sessions["Date String"] + " " + sessions["Start Time"],
                           format="%Y-%m-%d %H:%M")
    if not start.equals(pd.to_datetime(sessions["weather_datetime"])):
        raise ValueError("Some weather timestamps do not match the session starts.")

    # Derive binary predictors at the class start and retain the numeric temperature in Celsius.
    sessions["session_time"] = start
    sessions["course_date"] = start.dt.strftime("%Y-%m-%d")
    sessions["morning"] = start.dt.hour.lt(12).astype(int)
    sessions["afternoon"] = 1 - sessions["morning"]
    sessions["sunny"] = sessions["weather_code"].isin([0, 1]).astype(int)
    sessions["feels_like_temperature_c"] = sessions["apparent_temperature"]
    # Each one-hot column is 1 for the corresponding course.
    for course in COURSES:
        sessions[course] = sessions["Course"].eq(course).astype(int)

    # Compare feature rows as a multiset so tied or reordered sessions remain valid.
    exported = pd.read_csv(ROOT / "data" / "combined_courses.csv", usecols=FEATURE_COLUMNS)
    pd.testing.assert_frame_equal(
        exported.sort_values(FEATURE_COLUMNS).reset_index(drop=True),
        sessions[FEATURE_COLUMNS].sort_values(FEATURE_COLUMNS).reset_index(drop=True),
        check_dtype=False,
        check_exact=True,
    )
    return sessions.sort_values(["session_time", "Course", "Class Id"], ignore_index=True)


def ode(t, initial, equilibrium, decay):
    """Evaluate the ODE at week offsets t using the initial rate, equilibrium, and decay per week."""
    return equilibrium + (initial - equilibrium) * np.exp(-decay * t)


def fit_baseline(sessions, train):
    """Fit the ODE on rows selected by train and return its parameters and all session probabilities."""
    # Pool attendance counts across courses within each training week.
    weekly = sessions.loc[train].groupby("Teaching Week Number")[["Attended", "Total"]].sum()
    # Fit the weekly ratios by bounded least squares. The covariance is unused.
    params, _ = curve_fit(ode, weekly.index.to_numpy() - 1,
                          (weekly["Attended"] / weekly["Total"]).to_numpy(),
                          p0=[.8, .3, .2], bounds=([0, 0, 0], [1, 1, 2]), maxfev=10000)
    # Map the shared weekly curve back to the original session rows.
    base = ode(sessions["Teaching Week Number"].to_numpy() - 1, *params)
    return params, base


def fit_adjustment(design, baseline_logit, attended, total):
    """Return log-odds weights for the design columns using a fixed ODE offset.

    Each design row describes a training session. The baseline_logit offset,
    observed attended count, and eligible total count align with those rows.
    """
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("The factor has an unidentifiable coefficient in training.")

    def loss_and_gradient(beta):
        """Evaluate the normalized binomial loss and its gradient at the candidate weights beta."""
        # Eta is the adjusted log odds. Logaddexp keeps log(1 + exp(eta)) stable.
        eta = baseline_logit + design @ beta
        loss = np.sum(total * np.logaddexp(0, eta) - attended * eta)
        # The score residual is expected minus observed attendance for each row.
        grad = design.T @ (total * expit(eta) - attended)
        return loss / total.sum(), grad / total.sum()

    # Start BFGS at zero adjustment and use the analytic gradient.
    result = minimize(loss_and_gradient, np.zeros(design.shape[1]), jac=True,
                      method="BFGS", options={"gtol": 1e-9})
    if not result.success and np.linalg.norm(result.jac, ord=np.inf) > 1e-7:
        raise RuntimeError(f"Factor fit failed: {result.message}")
    return result.x


def plot_scores(scores, output, title="Attendance prediction error on 18 sessions in Weeks 9–10"):
    """Save MAE and RMSE from scores to the output PNG under the supplied title."""
    # Keep both metric panels in MAE order and reuse the source model colors.
    ordered = scores.sort_values("MAE_pp").reset_index(drop=True)
    y = np.arange(len(ordered))
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), sharey=True, layout="constrained")
    colors_by_model = dict(zip(scores["model"], MODEL_COLORS))
    colors = ordered["model"].map(colors_by_model)
    # Each axis receives one error series and its dashed baseline reference.
    for ax, metric in zip(axes, ["MAE", "RMSE"]):
        values = ordered[f"{metric}_pp"]
        reference = float(scores.loc[scores["model"].eq(BASELINE), f"{metric}_pp"].iloc[0])
        ax.barh(y, values, color=colors, height=.59)
        ax.axvline(reference, color="#364152", ls="--", lw=1.2, label="Baseline")
        # Label each bar at the corresponding y position.
        for yi, value in zip(y, values):
            ax.text(value + .16, yi, f"{value:.2f}", ha="left", va="center", fontsize=9)
        ax.set_xlim(0, float(values.max()) * 1.16)
        ax.set_yticks(y, ordered["model"])
        ax.set_xlabel(f"{metric} (percentage points)")
        ax.grid(axis="x", alpha=.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=14)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_predictions(test, actual, predictions, output,
                     title="Observed and predicted attendance in Weeks 9–10"):
    """Save one observed-versus-predicted panel per model to the output PNG.

    The test frame supplies session order and dates. The actual probabilities
    and each named array in predictions must follow those same rows.
    """
    # Space points by session so multiple classes on the same date stay visible.
    x = np.arange(len(test))
    dates = test["session_time"].dt.normalize().reset_index(drop=True)
    tick_positions = dates.index[~dates.duplicated()]
    tick_labels = dates.iloc[tick_positions].dt.strftime("%d %b")
    week_start = int(np.flatnonzero(test["Teaching Week Number"].eq(10))[0])

    # Use two columns and enough panel rows for every supplied model.
    panel_rows = (len(predictions) + 1) // 2
    fig, axes = plt.subplots(panel_rows, 2, figsize=(12.4, 2.55 * panel_rows + .75),
                             sharex=True, sharey=True, squeeze=False,
                             layout="constrained")
    # Each panel shows errors in percentage points for the same observed sessions.
    for ax, (model, predicted), color in zip(axes.flat, predictions.items(), cycle(MODEL_COLORS)):
        score = 100 * np.abs(predicted - actual).mean()
        ax.axvline(week_start - .5, color="#A0AAB8", ls="--", lw=1)
        ax.vlines(x, actual * 100, predicted * 100, color="#CDD5DF", lw=1)
        ax.plot(x, predicted * 100, color=color, lw=1.15, alpha=.65, zorder=2)
        ax.scatter(x, actual * 100, s=29, color="#292F3B", zorder=3)
        ax.scatter(x, predicted * 100, s=36, color=color, marker="D",
                   zorder=4)
        ax.set_title(f"{model}  |  MAE {score:.2f} pp", loc="left", fontsize=10)
        ax.set_ylim(0, 100)
        ax.set_xlim(-.6, len(test) - .4)
        ax.set_xticks(tick_positions, tick_labels, rotation=32, ha="right", fontsize=8)
        ax.tick_params(axis="x", labelbottom=True)
        ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(predictions):]:
        ax.set_visible(False)
    fig.legend(handles=[
        Line2D([], [], marker="o", ls="none", color="#292F3B", label="Observed"),
        Line2D([], [], marker="D", lw=1.15, color="#2455A4", label="Predicted by session"),
        Line2D([], [], marker="|", markersize=12, ls="none", color="#CDD5DF",
               label="Observed–predicted gap"),
        Line2D([], [], ls="--", color="#A0AAB8", label="Start of Week 10"),
    ], loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    fig.supylabel("Recorded attendance rate (%)")
    fig.suptitle(title, fontsize=14)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def single_factor_designs(sessions, train):
    """Return per-factor design matrices and column names, plus the training temperature mean."""
    # Center temperatures using training sessions only. The sunny flag defines two categories.
    temp_center = sessions.loc[train, "feels_like_temperature_c"].mean()
    sunny = sessions["sunny"].to_numpy()
    # Matrix columns and their weight names share the same ordering.
    designs = {
        "Baseline + Course": (sessions[COURSES].to_numpy(), COURSES),
        "Baseline + Time of day": (sessions[["morning", "afternoon"]].to_numpy(), ["morning", "afternoon"]),
        "Baseline + Sunny": (np.column_stack([1 - sunny, sunny]), ["not_sunny", "sunny"]),
        "Baseline + Feels-like temperature": (
            np.column_stack([np.ones(len(sessions)),
                             sessions["feels_like_temperature_c"].to_numpy() - temp_center]),
            ["offset_at_training_mean", "per_degree_c"]),
    }
    return designs, temp_center


def main():
    """Run the fixed chronological comparison and write single-factor charts and scores."""
    sessions = load_sessions()
    # Boolean masks keep Weeks 9–10 separate from parameter estimation.
    weeks = sessions["Teaching Week Number"]
    train = weeks.between(1, 8)
    test = weeks.between(9, 10)
    if set(weeks) != set(range(1, 11)) or not (train | test).all():
        raise ValueError("Expected sessions in teaching Weeks 1–10.")

    # Every factor model starts from the same fitted ODE and a numerically safe logit.
    params, base = fit_baseline(sessions, train)
    base_logit = logit(np.clip(base, 1e-6, 1 - 1e-6))
    designs, temp_center = single_factor_designs(sessions, train)
    # The binomial fit uses counts y and n. Actual contains the holdout attendance proportions.
    y = sessions["Attended"].to_numpy()
    n = sessions["Total"].to_numpy()
    actual = y[test] / n[test]
    predictions = {BASELINE: base[test]}
    print(f"Train/test sessions: {train.sum()} / {test.sum()}")
    print(f"ODE parameters (initial, equilibrium, decay): {params}")
    print(f"Training mean feels-like temperature: {temp_center:.4f} C")
    # Fit each design's weights on the training rows and store its holdout probabilities.
    for name, (design, terms) in designs.items():
        beta = fit_adjustment(design[train], base_logit[train], y[train], n[train])
        predictions[name] = expit(base_logit[test] + design[test] @ beta)
        print(f"{name}: {dict(zip(terms, np.round(beta, 6)))}")

    # Convert the per-session residuals to percentage points before summarizing them.
    rows = []
    for name, predicted in predictions.items():
        if not np.isfinite(predicted).all() or not ((0 <= predicted) & (predicted <= 1)).all():
            raise ValueError(f"Invalid predictions for {name}")
        errors_pp = 100 * (predicted - actual)
        rows.append({"model": name, "MAE_pp": np.abs(errors_pp).mean(),
                     "RMSE_pp": np.sqrt(np.mean(errors_pp ** 2))})
    # The baseline is inserted first, so its score is the reference for both deltas.
    scores = pd.DataFrame(rows)
    scores["MAE_change_vs_baseline_pp"] = scores["MAE_pp"] - scores.loc[0, "MAE_pp"]
    scores["RMSE_change_vs_baseline_pp"] = scores["RMSE_pp"] - scores.loc[0, "RMSE_pp"]
    print(scores.sort_values("MAE_pp").to_string(index=False, float_format="{:.6f}".format))

    # Keep the plots and machine-readable error table together.
    figures = ROOT / "figures"
    figures.mkdir(exist_ok=True)
    plot_scores(scores, figures / "single_factor_holdout_errors.png")
    plot_predictions(sessions.loc[test], actual, predictions,
                     figures / "single_factor_holdout_predictions.png")
    scores.sort_values("MAE_pp").to_csv(figures / "single_factor_scores.csv",
                                       index=False, float_format="%.8f")


if __name__ == "__main__":
    main()
