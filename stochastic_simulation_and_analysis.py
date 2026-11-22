"""
Stochastic Simulation and Statistical Analysis for Course Attendance
COMP0212 Modelling and Simulation

This script:
1. Loads cleaned session attendance and environmental data using load_sessions().
2. Computes the Baseline ODE, ODE + Course, and ODE + Course + Weather predicted probabilities.
3. Simulates individual student attendance via Monte Carlo Binomial draws:
   X_i ~ Binomial(N_i, P_i), where P_i is the model-derived attendance probability.
4. Simulates sensor/card-reader noise on the observed counts (Gaussian / additive counting noise).
5. Computes statistical metrics: Mean Error, SD, 95% Confidence Intervals, RMSE, MAE.
6. Evaluates parameter sensitivity (varying decay rate k and initial attendance A0).
7. Exports publication-quality figures for the LaTeX report into figures/.
8. Exports a summary metrics CSV.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit, logit

from compare_single_factors import load_sessions, fit_baseline

# Set random seed for exact reproducibility
SEED = 42
RNG = np.random.default_rng(SEED)

ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT / "figures" / "factor_model_weights.json"
FIGURES_DIR = ROOT / "figures"

def run_simulation_pipeline():
    sessions = load_sessions()
    with open(WEIGHTS_PATH, "r") as f:
        weights = json.load(f)

    # 1. Baseline ODE
    A0 = weights["baseline"]["A0"]
    A_eq = weights["baseline"]["A_eq"]
    k = weights["baseline"]["k"]

    t = sessions["Teaching Week Number"].to_numpy() - 1.0
    A_t = A_eq + (A0 - A_eq) * np.exp(-k * t)
    base_logit = logit(np.clip(A_t, 1e-6, 1.0 - 1e-6))

    # Course one-hot
    courses = ["COMP0206", "COMP0207", "COMP0208", "COMP0209"]
    course_dummies = pd.get_dummies(sessions["Course"])[courses].to_numpy()

    # 2. ODE + Course
    course_w_dict = weights["single_factor_weights"]["Course"]["Course"]
    course_weights = [course_w_dict[c] for c in courses]
    eta_course = base_logit + course_dummies @ np.array(course_weights)
    p_course = expit(eta_course)

    # 3. ODE + Course + Temp
    pair_w = weights["two_factor_weights"]["Course + Feels-like temperature"]
    pair_course_w = [pair_w["Course"][c] for c in courses]
    c_mean = weights["factor_definitions"]["Feels-like temperature"]["training_mean_c"]
    temp_centered = sessions["apparent_temperature"].to_numpy() - c_mean
    beta_temp = pair_w["Feels-like temperature"]["per_degree_c"]
    eta_pair = base_logit + course_dummies @ np.array(pair_course_w) + temp_centered * beta_temp
    p_pair = expit(eta_pair)

    sessions["p_baseline"] = A_t
    sessions["p_course"] = p_course
    sessions["p_course_temp"] = p_pair

    # Cohort size
    cohort_size = sessions["Total"].iloc[0] # 58
    n_sessions = len(sessions)
    num_trials = 2000

    sim_results = {}
    for model_name, p in [
        ("Baseline ODE", A_t),
        ("ODE + Course", p_course),
        ("ODE + Course + Temp", p_pair)
    ]:
        # Binomial Monte Carlo draws: shape (num_trials, n_sessions)
        sim_counts = RNG.binomial(n=cohort_size, p=p, size=(num_trials, n_sessions))
        sim_rates = sim_counts / cohort_size

        # Sensor RFID noise: N(0, 0.015^2)
        sensor_noise = RNG.normal(loc=0.0, scale=0.015, size=(num_trials, n_sessions))
        measured_rates = np.clip(sim_rates + sensor_noise, 0.0, 1.0)

        mean_rate = np.mean(measured_rates, axis=0)
        sd_rate = np.std(measured_rates, axis=0)
        ci_lower = np.percentile(measured_rates, 2.5, axis=0)
        ci_upper = np.percentile(measured_rates, 97.5, axis=0)

        sim_results[model_name] = {
            "mean_rate": mean_rate,
            "sd_rate": sd_rate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper
        }

    # Statistical Evaluation on Weeks 9-10 holdout
    sessions["attendance_rate"] = sessions["Attended"] / sessions["Total"]
    holdout_mask = (sessions["Teaching Week Number"] >= 9).to_numpy()
    actual_rates = sessions.loc[holdout_mask, "attendance_rate"].to_numpy()
    n_holdout = len(actual_rates) # 18

    stats_summary = []
    for model_name, res in sim_results.items():
        sim_means = res["mean_rate"][holdout_mask]
        errors = (sim_means - actual_rates) * 100.0 # pp

        mean_err = np.mean(errors)
        sd_err = np.std(errors, ddof=1)
        t_crit = 2.110 # t_0.975 for df=17
        ci_half = t_crit * (sd_err / np.sqrt(n_holdout))

        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))

        stats_summary.append({
            "Model": model_name,
            "Mean Error (pp)": round(mean_err, 3),
            "SD of Error (pp)": round(sd_err, 3),
            "95% CI Lower (pp)": round(mean_err - ci_half, 3),
            "95% CI Upper (pp)": round(mean_err + ci_half, 3),
            "MAE (pp)": round(mae, 3),
            "RMSE (pp)": round(rmse, 3)
        })

    summary_df = pd.DataFrame(stats_summary)
    summary_csv = FIGURES_DIR / "simulation_performance_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    print("Saved summary table to:", summary_csv)
    print(summary_df.to_string(index=False))

    # 1. Monte Carlo Distribution Plot
    plt.figure(figsize=(11, 5.2))
    session_idx = np.arange(n_sessions)
    obs = sessions["attendance_rate"].to_numpy() * 100.0

    res = sim_results["ODE + Course"]
    mean_sim = res["mean_rate"] * 100.0
    ci_low = res["ci_lower"] * 100.0
    ci_up = res["ci_upper"] * 100.0

    split_idx = np.where(sessions["Teaching Week Number"].to_numpy() >= 9)[0][0]

    plt.axvspan(-0.5, split_idx - 0.5, color="#f1f5f9", alpha=0.9, label="Training Set (Weeks 1–8, 76 sessions)")
    plt.axvspan(split_idx - 0.5, n_sessions - 0.5, color="#fef3c7", alpha=0.6, label="Holdout Set (Weeks 9–10, 18 sessions)")
    plt.axvline(split_idx - 0.5, color="#e11d48", linestyle="--", linewidth=1.5, label="Train/Holdout Split")

    plt.fill_between(session_idx, ci_low, ci_up, color="#0284c7", alpha=0.25, label="95% Monte Carlo CI (Binomial + Sensor Noise)")
    plt.plot(session_idx, mean_sim, color="#0369a1", linewidth=2.0, label="Stochastic Simulation Mean (ODE + Course)")
    plt.scatter(session_idx, obs, color="#0f172a", s=22, zorder=5, label="Recorded Card-Tap Attendance")

    plt.title("Stochastic Monte Carlo Attendance Simulation vs Ground Truth (94 Sessions)", fontsize=12, fontweight="bold")
    plt.xlabel("Teaching Session Sequence (Chronological)", fontsize=10.5)
    plt.ylabel("Attendance Rate (%)", fontsize=10.5)
    plt.ylim(0, 105)
    plt.xlim(-1, n_sessions)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower left", frameon=True, fontsize=9)
    plt.tight_layout()

    mc_plot = FIGURES_DIR / "stochastic_simulation_monte_carlo.png"
    plt.savefig(mc_plot, dpi=300)
    plt.close()
    print("Saved Monte Carlo plot to:", mc_plot)

    # 2. Sensitivity Analysis Plot
    t_smooth = np.linspace(0, 9, 200)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    k_vals = [0.10, 0.15, k, 0.25, 0.35]
    colors_k = plt.cm.Blues(np.linspace(0.4, 0.9, len(k_vals)))
    for kval, c in zip(k_vals, colors_k):
        curve = A_eq + (A0 - A_eq) * np.exp(-kval * t_smooth)
        lbl = f"k = {kval:.3f}" + (" (Fitted Base)" if abs(kval - k) < 1e-4 else "")
        lw = 2.4 if abs(kval - k) < 1e-4 else 1.4
        ax1.plot(t_smooth + 1, curve * 100.0, label=lbl, color=c, linewidth=lw)

    ax1.set_title("Sensitivity to Decay Rate $k$", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Teaching Week", fontsize=10)
    ax1.set_ylabel("Expected Attendance Rate (%)", fontsize=10)
    ax1.set_ylim(20, 90)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(frameon=True, fontsize=9)

    A0_vals = [0.70, 0.75, A0, 0.85, 0.90]
    colors_A0 = plt.cm.Oranges(np.linspace(0.4, 0.9, len(A0_vals)))
    for A0val, c in zip(A0_vals, colors_A0):
        curve = A_eq + (A0val - A_eq) * np.exp(-k * t_smooth)
        lbl = f"$A_0$ = {A0val:.3f}" + (" (Fitted Base)" if abs(A0val - A0) < 1e-4 else "")
        lw = 2.4 if abs(A0val - A0) < 1e-4 else 1.4
        ax2.plot(t_smooth + 1, curve * 100.0, label=lbl, color=c, linewidth=lw)

    ax2.set_title("Sensitivity to Initial Attendance $A_0$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Teaching Week", fontsize=10)
    ax2.set_ylabel("Expected Attendance Rate (%)", fontsize=10)
    ax2.set_ylim(20, 95)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(frameon=True, fontsize=9)

    plt.tight_layout()
    sens_plot = FIGURES_DIR / "parameter_sensitivity_ode.png"
    plt.savefig(sens_plot, dpi=300)
    plt.close()
    print("Saved Sensitivity plot to:", sens_plot)

if __name__ == "__main__":
    run_simulation_pipeline()
