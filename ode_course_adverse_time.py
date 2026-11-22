"""
COMP0212 Person 2: independent ODE + Course + Adverse Weather + Class Time model.

Inputs (place beside this .py file):
    RAI_attendance_with_weather.csv

Output folder:
    ODE_Adverse_Results/

All charts show EACH individual class session, never weekly-aggregated attendance.
Training: Weeks 1-8; holdout testing: Weeks 9-10.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.optimize import curve_fit, minimize
from scipy.special import expit, logit

# Resolve the input file and this variant's results folder relative to the script.
HERE = Path(__file__).resolve().parent
INPUT_CSV = HERE / "RAI_attendance_with_weather.csv"
OUTPUT_DIR = HERE / "ODE_Adverse_Results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# The mode selects the threshold rule. The display name is used in reports.
WEATHER_MODE = "adverse"
WEATHER_NAME = "Adverse Weather"
RAIN_THRESHOLD = 1.0       # mm; adverse if rain > 1.0 mm in the matched hour
WIND_THRESHOLD = 20.0     # km/h; adverse if wind > 20 km/h
CUTOFF_HOUR = 12          # 12:00 and later => afternoon


def ode_model(t, A0, A_eq, k):
    """Return attendance at week offsets t with initial rate A0, equilibrium A_eq, and decay k."""
    return A_eq + (A0 - A_eq) * np.exp(-k * t)


def errors(actual, predicted):
    """Return unweighted MAE and RMSE for aligned observed and predicted proportions."""
    # Convert both series to numeric arrays for elementwise comparison.
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    return (float(np.mean(np.abs(actual - predicted))),
            float(np.sqrt(np.mean((actual - predicted) ** 2))))


def fit_model(train_df, factors, A0, A_eq, k, course_names):
    """Jointly fit the course log-odds offsets and listed binary factors.

    Use session counts from train_df, ODE parameters A0, A_eq, and k, and the
    stable ordering in course_names. Return two dictionaries keyed by course
    and factor name. Each variant starts its optimizer at zero.
    """
    # Give each session its teaching week's fitted baseline before applying any factors.
    base = ode_model(train_df["Teaching Week Number"].to_numpy() - 1,
                     A0, A_eq, k)
    base_z = logit(np.clip(base, 1e-6, 1 - 1e-6))
    # Course codes index the first block of coefficients in the optimizer vector.
    codes = pd.Categorical(train_df["Module"], categories=course_names).codes
    if (codes < 0).any():
        raise ValueError("Unknown course name in training data")
    # X contains the requested factors. y and n are attended and eligible counts.
    X = (np.column_stack([train_df[f].to_numpy(dtype=float) for f in factors])
         if factors else np.empty((len(train_df), 0)))
    y = train_df["Attended"].to_numpy(dtype=float)
    n = train_df["Total"].to_numpy(dtype=float)
    n_courses = len(course_names)

    def negative_log_likelihood(theta):
        """Evaluate binomial loss at theta, ordered as course weights then factor weights."""
        # z contains adjusted log odds and p is clipped to keep logarithms finite.
        z = base_z + theta[:n_courses][codes]
        if factors:
            z += X @ theta[n_courses:]
        p = np.clip(expit(z), 1e-9, 1 - 1e-9)
        return -float(np.sum(y * np.log(p) + (n - y) * np.log(1 - p)))

    # Fit all requested adjustments together while the ODE baseline stays fixed.
    result = minimize(negative_log_likelihood,
                      x0=np.zeros(n_courses + len(factors)),
                      method="L-BFGS-B")
    if not result.success:
        raise RuntimeError(f"Fit failed for {factors}: {result.message}")
    return dict(zip(course_names, result.x[:n_courses])), dict(
        zip(factors, result.x[n_courses:]))


def predict(df, A0, A_eq, k, course_offsets=None, factor_offsets=None):
    """Predict probabilities for df using the ODE parameters and optional offset mappings."""
    # p0 is the shared weekly baseline for each session's teaching week.
    p0 = ode_model(df["Teaching Week Number"].to_numpy() - 1, A0, A_eq, k)
    if course_offsets is None:
        return p0
    # Look up the per-course shift before adding the named feature weights beta.
    course_values = df["Module"].map(course_offsets)
    if course_values.isna().any():
        raise ValueError("Unknown course in predictions")
    z = logit(np.clip(p0, 1e-6, 1 - 1e-6)) + course_values.to_numpy()
    for factor, beta in (factor_offsets or {}).items():
        z += beta * df[factor].to_numpy(dtype=float)
    return expit(z)


def plot_per_session(df, predicted_col, title, filename, course_names):
    """Save predicted_col against observed rates in df, with one panel per named course.

    Use title for the figure heading and filename for its PNG destination.
    """
    # Use common y limits across courses and sort each panel by actual class start.
    fig, axes = plt.subplots(2, 2, figsize=(15, 9), sharey=True)
    for ax, name in zip(axes.flat, course_names):
        part = df.loc[df["Module"] == name].sort_values(
            ["session_time", "Class Id"] if "Class Id" in df else ["session_time"])
        # Use the timestamp of each retained class as the horizontal coordinate.
        dates = part["session_time"]
        ax.scatter(dates, part["actual_rate"], s=28, color="black",
                   alpha=.75, marker="o", label="Observed (each session)", zorder=3)
        # Line is a sequence of session predictions, not a weekly average.
        ax.plot(dates, part[predicted_col], "-o", linewidth=1.45,
                markersize=3.6, color="#2779B3", label="Predicted (each session)")
        # Mark the chronological start of the holdout on every course panel.
        first_test = df.loc[df["Teaching Week Number"] > 8, "session_time"].min()
        if pd.notna(first_test):
            ax.axvline(first_test, ls="--", color="#BD6F35", lw=1,
                       label="Week 9 begins")
        ax.set_title(name, fontsize=10)
        ax.set_ylim(0, 1.03)
        ax.set_xlabel("Session date")
        ax.set_ylabel("Recorded attendance rate")
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.tick_params(axis="x", rotation=30)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7, loc="best")
    fig.suptitle(title + " | every scheduled session", fontsize=14, y=.995)
    fig.tight_layout(rect=[0, 0, 1, .97])
    fig.savefig(OUTPUT_DIR / filename, dpi=250, bbox_inches="tight")
    plt.close(fig)


def main():
    """Fit the configured weather variants and export per-session predictions and holdout errors."""
    if not INPUT_CSV.exists():
        raise FileNotFoundError(
            f"Missing {INPUT_CSV.name}. Put this script next to that CSV."
        )
    # Each source row represents a session with recorded counts and matched weather.
    df = pd.read_csv(INPUT_CSV)
    mandatory = {"Date String", "Start Time", "Module", "Teaching Week Number",
                 "Attended", "Total", "rain", "wind_kmh"}
    # Stop before fitting if the input omits any required source columns.
    absent = mandatory - set(df.columns)
    if absent:
        raise ValueError(f"Required CSV columns missing: {sorted(absent)}")

    # Normalize local start times and coerce the numeric fit inputs before validation.
    df["session_time"] = pd.to_datetime(
        df["Date String"].astype(str) + " " + df["Start Time"].astype(str),
        errors="raise"
    )
    for col in ["Attended", "Total", "rain", "wind_kmh", "Teaching Week Number"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[["Attended", "Total", "Teaching Week Number"]].isna().any().any():
        raise ValueError("Attendance counts or teaching week missing")
    if (df["Total"] <= 0).any() or (df["Attended"] > df["Total"]).any():
        raise ValueError("Invalid attendance counts")
    if df[["rain", "wind_kmh"]].isna().any().any():
        raise ValueError("Missing rain or wind: do not silently call unknown weather normal")
    # Define the target proportion and the binary time-of-day predictor.
    df["actual_rate"] = df["Attended"] / df["Total"]
    df["Afternoon"] = (df["session_time"].dt.hour >= CUTOFF_HOUR).astype(int)

    # Weather is a binary predictor derived from this variant's selected rule.
    if WEATHER_MODE == "rain":
        df["Weather"] = (df["rain"] > 0).astype(int)
        weather_definition = "Rain = 1 when rain > 0 mm in the matched start-hour"
    else:
        df["Weather"] = ((df["rain"] > RAIN_THRESHOLD) |
                         (df["wind_kmh"] > WIND_THRESHOLD)).astype(int)
        weather_definition = (
            "Adverse = 1 when rain > 1 mm OR wind > 20 km/h; "
            "cloud/overcast alone is NOT adverse")

    # Sort for plotting and reserve later weeks for holdout evaluation.
    df = df.sort_values(["session_time", "Module"]).reset_index(drop=True)
    train = df[df["Teaching Week Number"] <= 8].copy()
    test = df[df["Teaching Week Number"] > 8].copy()
    if train.empty or test.empty:
        raise ValueError("Both Weeks 1-8 and Weeks 9-10 must be present")
    # Use the same stable course ordering in plots and coefficient lookups.
    course_names = sorted(df["Module"].unique().tolist())
    print(f"\n=== {WEATHER_NAME} model ===")
    print(f"Rows: {len(df)} (train={len(train)}, test={len(test)})")
    print(weather_definition)
    print("Weather 0/1:", df["Weather"].value_counts().to_dict())
    print("Afternoon 0/1:", df["Afternoon"].value_counts().to_dict())
    print("Weather by teaching week:\n", pd.crosstab(
        df["Teaching Week Number"], df["Weather"]))
    print("Afternoon by course:\n", pd.crosstab(df["Module"], df["Afternoon"]))

    # As in the original uploaded script: fit ODE to WEEKLY pooled data on Weeks 1-8.
    # Visualizations and final error comparisons use EACH lesson independently.
    weekly_train = train.groupby("Teaching Week Number").agg(
        attended=("Attended", "sum"), total=("Total", "sum"))
    weekly_train["rate"] = weekly_train["attended"] / weekly_train["total"]
    # Keep the fitted initial rate, equilibrium, and decay. Discard the covariance.
    ode_params, _ = curve_fit(
        ode_model, weekly_train.index.to_numpy() - 1,
        weekly_train["rate"].to_numpy(),
        p0=[.8, .3, .2], bounds=([0, 0, 0], [1, 1, 2]), maxfev=10000)
    A0, A_eq, k = ode_params
    print(f"\nODE: A0={A0:.6f}, A_eq={A_eq:.6f}, k={k:.6f}")

    # Independently fit 4 nested/alternative models, with no shared optimizer state.
    course_betas, _ = fit_model(train, [], A0, A_eq, k, course_names)
    course_weather_betas, weather_only = fit_model(
        train, ["Weather"], A0, A_eq, k, course_names)
    course_time_betas, time_only = fit_model(
        train, ["Afternoon"], A0, A_eq, k, course_names)
    full_course_betas, full_extra = fit_model(
        train, ["Weather", "Afternoon"], A0, A_eq, k, course_names)

    # Predictions and model scores from the SAME per-session rows.
    df["Pred_Base_ODE"] = predict(df, A0, A_eq, k)
    df["Pred_Course"] = predict(df, A0, A_eq, k, course_betas)
    df["Pred_Course_Weather"] = predict(
        df, A0, A_eq, k, course_weather_betas, weather_only)
    df["Pred_Course_Time"] = predict(
        df, A0, A_eq, k, course_time_betas, time_only)
    df["Pred_Full"] = predict(
        df, A0, A_eq, k, full_course_betas, full_extra)

    # Each tuple names the model, its prediction column, and its plot file.
    variants = [
        ("Base ODE", "Pred_Base_ODE", "01_ODE_4_Courses.png"),
        ("ODE + Course", "Pred_Course", "02_ODE_Plus_Course_4_Courses.png"),
        ("ODE + Course + " + WEATHER_NAME,
         "Pred_Course_Weather", "03_ODE_Plus_Weather_4_Courses.png"),
        ("ODE + Course + Afternoon", "Pred_Course_Time",
         "04_ODE_Plus_Time_4_Courses.png"),
        ("ODE + Course + " + WEATHER_NAME + " + Afternoon",
         "Pred_Full", "05_Full_Model_4_Courses.png"),
    ]
    # report collects one MAE and RMSE row per variant on the same holdout classes.
    report = []
    held_out = df["Teaching Week Number"] > 8
    for title, col, filename in variants:
        plot_per_session(df, col, title, filename, course_names)
        mae, rmse = errors(df.loc[held_out, "actual_rate"],
                           df.loc[held_out, col])
        report.append({"Model": title, "MAE": mae, "RMSE": rmse})

    # Export the aggregate holdout scores and the full chronological session table.
    scores = pd.DataFrame(report)
    scores.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    df.to_csv(OUTPUT_DIR / "per_session_predictions.csv", index=False)

    # Compact comparison chart: errors calculated on individual Week 9-10 lessons.
    fig, ax = plt.subplots(figsize=(11.5, 5))
    # Center a pair of adjacent metric bars around each model's y coordinate.
    y = np.arange(len(scores))
    bar_width = .34
    ax.barh(y - bar_width/2, scores["MAE"], bar_width, label="MAE")
    ax.barh(y + bar_width/2, scores["RMSE"], bar_width, label="RMSE")
    ax.set_yticks(y, scores["Model"])
    ax.invert_yaxis()
    ax.set_xlabel("Error on individual sessions in Weeks 9-10 (lower is better)")
    ax.set_title("Holdout model comparison — " + WEATHER_NAME)
    ax.grid(axis="x", alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_Model_Comparison.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    print("\n=== Joint fitted factors (training Weeks 1-8) ===")
    print("Weather only factor:", weather_only)
    print("Afternoon only factor:", time_only)
    print("Full model course offsets:", full_course_betas)
    print("Full model additional factors:", full_extra)
    print("\n=== Holdout errors: EACH individual session in Weeks 9-10 ===")
    print(scores.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved 6 PNGs and 2 CSVs to:", OUTPUT_DIR)
    print("Note: session-to-session lines connect class predictions; not a continuous ODE solution")


if __name__ == "__main__":
    main()
