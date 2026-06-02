import os
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test, logrank_test
from sklearn.preprocessing import StandardScaler
 
COVARIATES_CSV = "./shhs1-dataset-0.21.0-subsampled.csv"
OUTPUT_DIR     = "./task1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
print("=" * 70)
print("  TASK 1 — Part 2: Analysis & Survival Prediction")
print("=" * 70)

# 1. Load the previously generated HRV features
hrv_csv_path = os.path.join(OUTPUT_DIR, "hrv_features.csv")
print(f"Loading previously generated features from {hrv_csv_path}...")
hrv_df = pd.read_csv(hrv_csv_path)

# 2. Load covariates and merge
print("Loading covariates...")
cov = pd.read_csv(COVARIATES_CSV, usecols=[
    "nsrrid", "age_s1", "gender", "bmi_s1", "ahi_a0h3",
    "avgsat", "minsat", "slpeffp", "slpprdp", "timeremp",
    "times34p", "systbp", "htnderv_s1", "ess_s1", "smokstat_s1",
])
cov = cov.rename(columns={
    "age_s1": "age", "gender": "sex", "bmi_s1": "bmi",
    "ahi_a0h3": "ahi", "avgsat": "avg_spo2", "minsat": "min_spo2",
    "slpeffp": "sleep_efficiency", "slpprdp": "sleep_duration_min",
    "timeremp": "rem_min", "times34p": "sws_min", "systbp": "systolic_bp",
    "htnderv_s1": "hypertension", "ess_s1": "ess", "smokstat_s1": "smoking",
})
cov["female"] = (cov["sex"] == 2).astype(int)
 
df = hrv_df.merge(cov, on="nsrrid", how="left")
df.to_csv(os.path.join(OUTPUT_DIR, "hrv_plus_covariates.csv"), index=False)
print(f"Merged with clinical covariates -> {OUTPUT_DIR}/hrv_plus_covariates.csv\n")
 
print("=" * 70)
print("  EXPLORATORY ANALYSIS")
print("=" * 70)
 
hrv_metrics = ["SDNN", "RMSSD", "pNN50", "LF_power", "HF_power",
               "LF_HF", "SD1", "SD2", "Total_power"]
 
print(f"\n  {'Metric':15s}  {'Alive':30s}  {'Dead':30s}  p-value")
print("  " + "-" * 80)
for m in hrv_metrics:
    alive = df.loc[df.event_observed == 0, m].dropna()
    dead  = df.loc[df.event_observed == 1, m].dropna()
    if len(alive) > 0 and len(dead) > 0:
        _, p = mannwhitneyu(alive, dead, alternative="two-sided")
        print(f"  {m:15s}  {alive.mean():7.3f} +/- {alive.std():6.3f}          "
              f"{dead.mean():7.3f} +/- {dead.std():6.3f}     {p:.3f}")
 
plot_df = df[["event_observed"] + hrv_metrics].copy()
plot_df["Status"] = plot_df["event_observed"].map({0: "Alive", 1: "Dead"})
 
fig, axes = plt.subplots(3, 3, figsize=(14, 11))
fig.suptitle("HRV Metrics by Vital Status  (SHHS-1)",
             fontsize=15, fontweight="bold", y=1.01)
axes = axes.flatten()
 
for ax, m in zip(axes, hrv_metrics):
    tmp = plot_df[["Status", m]].dropna()
    sns.violinplot(data=tmp, x="Status", y=m, ax=ax,
                   palette={"Alive": "#4C9BE8", "Dead": "#E84C4C"},
                   inner="box", cut=0)
    g1 = tmp.loc[tmp.Status == "Alive", m].values
    g2 = tmp.loc[tmp.Status == "Dead",  m].values
    if len(g1) > 0 and len(g2) > 0:
        _, p = mannwhitneyu(g1, g2, alternative="two-sided")
        ax.set_title(f"{m}  (p={p:.3f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_hrv_by_vital_status.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Figure 1 saved -> fig1_hrv_by_vital_status.png")
 
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Kaplan-Meier Survival Curves by HRV Median Split  (SHHS-1)",
             fontsize=13, fontweight="bold")
 
for ax, metric in zip(axes, ["SDNN", "RMSSD", "HF_power"]):
    tmp = df[["duration", "event_observed", metric]].dropna().copy()
    med = tmp[metric].median()
    tmp["group"] = (tmp[metric] >= med).map(
        {True: f"High {metric}", False: f"Low {metric}"}
    )
    kmf    = KaplanMeierFitter()
    colors = ["#2196F3", "#F44336"]
    for (grp, grp_df), col in zip(tmp.groupby("group"), colors):
        kmf.fit(grp_df["duration"], grp_df["event_observed"], label=grp)
        kmf.plot_survival_function(ax=ax, ci_show=True, color=col)
    g_high = tmp[tmp.group.str.startswith("High")]
    g_low  = tmp[tmp.group.str.startswith("Low")]
    lr     = logrank_test(g_high["duration"], g_low["duration"],
                          g_high["event_observed"], g_low["event_observed"])
    ax.set_title(f"Median split: {metric}\n(log-rank p={lr.p_value:.3f})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Follow-up (days)", fontsize=10)
    ax.set_ylabel("Survival probability", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_kaplan_meier.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 2 saved -> fig2_kaplan_meier.png")
 
corr_cols = ["SDNN", "RMSSD", "pNN50", "CV_RR",
             "VLF_power", "LF_power", "HF_power", "Total_power",
             "LF_HF", "SD1", "SD2", "SD1_SD2"]
fig, ax = plt.subplots(figsize=(10, 8))
corr = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            center=0, ax=ax, annot_kws={"size": 8},
            linewidths=0.3, vmin=-1, vmax=1)
ax.set_title("HRV Metrics Correlation Matrix", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_hrv_correlation.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 3 saved -> fig3_hrv_correlation.png")
 
apnea_cols = ["n_apnea_events", "n_hypopnea_events",
              "total_apnea_duration_sec", "n_arousal_events",
              "n_plm_events", "apnea_index_from_xml"]
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("XML-Derived Event Counts by Vital Status  (SHHS-1)",
             fontsize=13, fontweight="bold")
axes = axes.flatten()
plot_df2 = df[["event_observed"] + apnea_cols].copy()
plot_df2["Status"] = plot_df2["event_observed"].map({0: "Alive", 1: "Dead"})
 
for ax, col in zip(axes, apnea_cols):
    tmp = plot_df2[["Status", col]].dropna()
    sns.boxplot(data=tmp, x="Status", y=col, ax=ax,
                palette={"Alive": "#4C9BE8", "Dead": "#E84C4C"})
    g1 = tmp.loc[tmp.Status == "Alive", col].values
    g2 = tmp.loc[tmp.Status == "Dead",  col].values
    if len(g1) > 1 and len(g2) > 1:
        _, p = mannwhitneyu(g1, g2, alternative="two-sided")
        ax.set_title(f"{col}\n(p={p:.3f})", fontsize=9, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_xml_events_by_vital_status.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 4 saved -> fig4_xml_events_by_vital_status.png")
 
print("\n" + "=" * 70)
print("  SURVIVAL ANALYSIS — Cox Proportional Hazards")
print("=" * 70)
 
# Avoiding collinearity
hrv_features = [
    "SDNN", 
    "VLF_power", 
    "LF_power", 
    "HF_power", 
    "RMSSD",   # Kept for clinical relevance
    "pNN50"    # Kept for clinical relevance
]
# Simplified to prevent singular matrix errors in Models B and C
xml_features = [
    "n_arousal_events", 
    "apnea_index_from_xml" 
]

cov_features = [
    "age", 
    "female", 
    "bmi", 
    "avg_spo2", 
    "sleep_efficiency", 
    "hypertension"
    
] 
print("\n  Model A: HRV metrics only")
df_A  = df[["duration", "event_observed"] + hrv_features].dropna()
dfs_A = df_A.copy()
dfs_A[hrv_features] = StandardScaler().fit_transform(df_A[hrv_features])
cox_A = CoxPHFitter(penalizer=0.1)
cox_A.fit(dfs_A, duration_col="duration", event_col="event_observed")
cox_A.print_summary(decimals=4)
ci_A  = cox_A.concordance_index_
print(f"\n  C-index (Model A — HRV only) : {ci_A:.4f}")
 
print("\n  Model B: HRV + XML annotation features")
feat_B = hrv_features + xml_features
df_B   = df[["duration", "event_observed"] + feat_B].dropna()
dfs_B  = df_B.copy()
dfs_B[feat_B] = StandardScaler().fit_transform(df_B[feat_B])
cox_B  = CoxPHFitter(penalizer=0.1)
cox_B.fit(dfs_B, duration_col="duration", event_col="event_observed")
cox_B.print_summary(decimals=4)
ci_B   = cox_B.concordance_index_
print(f"\n  C-index (Model B — HRV + XML features) : {ci_B:.4f}")
 
print("\n  Model C: HRV + XML + clinical covariates")
feat_C = hrv_features + xml_features + cov_features
df_C   = df[["duration", "event_observed"] + feat_C].dropna()
dfs_C  = df_C.copy()
dfs_C[feat_C] = StandardScaler().fit_transform(df_C[feat_C])
cox_C  = CoxPHFitter(penalizer=0.1)
cox_C.fit(dfs_C, duration_col="duration", event_col="event_observed")
cox_C.print_summary(decimals=4)
ci_C   = cox_C.concordance_index_
print(f"\n  C-index (Model C — HRV + XML + clinical) : {ci_C:.4f}")
 
print("\n  Proportional Hazards Assumption Test (Model C):")
try:
    ph = proportional_hazard_test(cox_C, dfs_C, time_transform="rank")
    print(ph.summary.to_string())
except Exception as e:
    print(f"  Skipped: {e}")
 
fig, ax = plt.subplots(figsize=(9, 8))
summary = cox_C.summary.sort_values("exp(coef)")
y_pos   = list(range(len(summary)))
colors  = ["#E84C4C" if v < 1 else "#4C9BE8" for v in summary["exp(coef)"]]
 
ax.barh(y_pos, summary["exp(coef)"] - 1, left=1,
        color=colors, alpha=0.75, height=0.6)
ax.errorbar(
    summary["exp(coef)"], y_pos,
    xerr=[
        summary["exp(coef)"] - summary["exp(coef) lower 95%"],
        summary["exp(coef) upper 95%"] - summary["exp(coef)"],
    ],
    fmt="none", color="black", capsize=4, linewidth=1.5,
)
ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
ax.set_yticks(y_pos)
ax.set_yticklabels(summary.index, fontsize=10)
ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=11)
ax.set_title(
    f"Cox Model C — Hazard Ratios  |  C-index = {ci_C:.3f}",
    fontsize=12, fontweight="bold",
)
ax.grid(axis="x", alpha=0.3)
for i, (_, row_data) in enumerate(summary.iterrows()):
    p    = row_data["p"]
    star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    ax.text(summary["exp(coef) upper 95%"].max() * 1.02, i,
            f"p={p:.3f} {star}", va="center", fontsize=8)
 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_hazard_ratios.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Figure 5 saved -> fig5_hazard_ratios.png")
 
pd.DataFrame({
    "Model":      ["A: HRV only", "B: HRV + XML", "C: HRV + XML + Clinical"],
    "N_subjects": [len(df_A), len(df_B), len(df_C)],
    "N_events":   [int(df_A.event_observed.sum()),
                   int(df_B.event_observed.sum()),
                   int(df_C.event_observed.sum())],
    "C_index":    [ci_A, ci_B, ci_C],
"AIC":        [cox_A.AIC_partial_, cox_B.AIC_partial_, cox_C.AIC_partial_],}).to_csv(os.path.join(OUTPUT_DIR, "model_performance.csv"), index=False)
print(f"  Performance table saved -> model_performance.csv")
