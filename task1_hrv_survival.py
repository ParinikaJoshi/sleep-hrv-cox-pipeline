import os
import warnings
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import mne
import neurokit2 as nk
import xmltodict
from scipy.signal import welch
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test, logrank_test
from sklearn.preprocessing import StandardScaler
 
DATA_DIR       = "./dataset/dataset"
OUTCOMES_CSV   = "./outcomes.csv"
COVARIATES_CSV = "./shhs1-dataset-0.21.0-subsampled.csv"
OUTPUT_DIR     = "./task1_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
 
CHUNK_SEC = 5 * 60
 
 
def parse_xml_events(xml_path):
    """
    Parse the SHHS annotation XML and return a dict containing:
      - sleep_stages : list of (start_sec, duration_sec, stage_code)
      - apnea_events : list of (start_sec, duration_sec, event_type)
      - arousal_events: list of (start_sec, duration_sec)
      - plm_events   : list of (start_sec, duration_sec)
 
    Event types in SHHS XML:
      Stages|Stages          -> sleep stages (codes 0-5, 9)
      Obstructive apnea|...  -> obstructive apnea
      Central Apnea|...      -> central apnea
      Mixed Apnea|...        -> mixed apnea
      Hypopnea|...           -> hypopnea
      SpO2 artifact|...      -> artifact (ignored)
      Arousal|...            -> arousal
      RERA|...               -> respiratory effort related arousal
      Limb Movement|...      -> limb movement
    """
    try:
        with open(xml_path, "r", encoding="utf-8", errors="replace") as f:
            doc = xmltodict.parse(f.read())
    except Exception as e:
        print(f"\n    [ERR] Cannot parse XML: {e}")
        return None
 
    try:
        events = doc["PSGAnnotation"]["ScoredEvents"]["ScoredEvent"]
    except KeyError:
        return None
 
    if isinstance(events, dict):
        events = [events]
 
    sleep_stages   = []
    apnea_events   = []
    arousal_events = []
    plm_events     = []
 
    for ev in events:
        etype = ev.get("EventType", "")
        econcept = ev.get("EventConcept", "")
        try:
            start    = float(ev.get("Start", 0))
            duration = float(ev.get("Duration", 0))
        except (ValueError, TypeError):
            continue
 
        if etype == "Stages|Stages":
            try:
                code = int(econcept.split("|")[1])
                sleep_stages.append((start, duration, code))
            except Exception:
                pass
 
        elif any(k in econcept.lower() for k in
                 ["obstructive apnea", "central apnea", "mixed apnea", "hypopnea"]):
            apnea_events.append((start, duration, econcept.split("|")[0].strip()))
 
        elif "arousal" in econcept.lower() or "rera" in econcept.lower():
            arousal_events.append((start, duration))
 
        elif "limb movement" in econcept.lower():
            plm_events.append((start, duration))
 
    return {
        "sleep_stages":    sleep_stages,
        "apnea_events":    apnea_events,
        "arousal_events":  arousal_events,
        "plm_events":      plm_events,
    }
 
 
def band_power(freqs, psd, lo, hi):
    mask = (freqs >= lo) & (freqs < hi)
    if mask.sum() < 2:
        return np.nan
    return float(np.trapezoid(psd[mask], freqs[mask]))
 
 
def hrv_from_rr(rr_ms, sfreq=None):
    """Compute all HRV metrics from an RR interval series (in ms)."""
    if len(rr_ms) < 20:
        return None
 
    sdnn    = float(np.std(rr_ms, ddof=1))
    rmssd   = float(np.sqrt(np.mean(np.diff(rr_ms) ** 2)))
    pnn50   = float(np.mean(np.abs(np.diff(rr_ms)) > 50) * 100)
    mean_rr = float(np.mean(rr_ms))
    mean_hr = 60000.0 / mean_rr
    cv_rr   = sdnn / mean_rr * 100.0
 
    fs_r    = 4.0
    rr_sec  = rr_ms / 1000.0
    cumtime = np.insert(np.cumsum(rr_sec), 0, 0)[:-1]
    t_uni   = np.arange(0, cumtime[-1], 1.0 / fs_r)
    rr_uni  = np.interp(t_uni, cumtime, rr_sec)
 
    nperseg    = min(256, len(rr_uni))
    freqs, psd = welch(rr_uni, fs=fs_r, nperseg=nperseg, noverlap=nperseg // 2)
 
    vlf   = band_power(freqs, psd, 0.003, 0.04)
    lf    = band_power(freqs, psd, 0.04,  0.15)
    hf    = band_power(freqs, psd, 0.15,  0.40)
    tp    = band_power(freqs, psd, 0.003, 0.40)
    lf_hf = lf / hf if hf and hf > 0 else np.nan
    denom  = (tp - vlf) if (tp and vlf and tp - vlf > 0) else np.nan
    lf_nu  = lf / denom * 100 if not np.isnan(denom) else np.nan
    hf_nu  = hf / denom * 100 if not np.isnan(denom) else np.nan
 
    rr1     = rr_ms[:-1]
    rr2     = rr_ms[1:]
    sd1     = float(np.std((rr2 - rr1) / np.sqrt(2), ddof=1))
    sd2     = float(np.std((rr2 + rr1) / np.sqrt(2), ddof=1))
    sd1_sd2 = sd1 / sd2 if sd2 > 0 else np.nan
 
    return {
        "SDNN": sdnn, "RMSSD": rmssd, "pNN50": pnn50,
        "mean_HR": mean_hr, "CV_RR": cv_rr, "mean_RR": mean_rr,
        "VLF_power": vlf, "LF_power": lf, "HF_power": hf,
        "Total_power": tp, "LF_HF": lf_hf, "LF_nu": lf_nu, "HF_nu": hf_nu,
        "SD1": sd1, "SD2": sd2, "SD1_SD2": sd1_sd2,
    }
 
 
def compute_hrv(edf_path, xml_path):
    """
    Full-night HRV computation using both EDF and XML.
 
    Steps:
      1. Load ECG channel from EDF
      2. Parse XML to get sleep stages and apnea event timestamps
      3. Detect R-peaks across the full night in 5-min chunks (neurokit2)
      4. Compute overall full-night HRV metrics
      5. Compute HRV separately during sleep vs wake (from XML stages)
      6. Compute HRV during apnea-free sleep epochs vs apnea epochs
      7. Extract apnea burden metrics from XML annotations
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    except Exception as e:
        print(f"\n    [ERR] EDF read error: {e}")
        return None
 
    sfreq     = raw.info["sfreq"]
    total_sec = raw.n_times / sfreq
 
    ecg_ch = next(
        (ch for ch in raw.ch_names if "ecg" in ch.lower() or "ekg" in ch.lower()),
        None,
    )
    if ecg_ch is None:
        print(f"\n    [WARN] No ECG channel. Available: {raw.ch_names}")
        return None
 
    raw.load_data(verbose=False)
    ecg_full = raw.get_data(picks=[ecg_ch])[0]
    del raw
 
    chunk_samples = int(CHUNK_SEC * sfreq)
    all_rpeaks    = []
 
    for start_samp in range(0, len(ecg_full), chunk_samples):
        chunk = ecg_full[start_samp : start_samp + chunk_samples]
        if len(chunk) < int(10 * sfreq):
            continue
        try:
            _, info     = nk.ecg_process(chunk, sampling_rate=int(sfreq))
            local_peaks = info["ECG_R_Peaks"]
            all_rpeaks.extend(local_peaks + start_samp)
        except Exception:
            continue
 
    if len(all_rpeaks) < 50:
        print(f"\n    [WARN] Only {len(all_rpeaks)} R-peaks detected")
        return None
 
    rpeaks    = np.array(sorted(set(all_rpeaks)))
    rpeak_sec = rpeaks / sfreq
 
    rr_ms_full = np.diff(rpeaks) / sfreq * 1000.0
    rr_ms_full = rr_ms_full[(rr_ms_full > 300) & (rr_ms_full < 2000)]
 
    overall = hrv_from_rr(rr_ms_full)
    if overall is None:
        return None
    overall["n_beats"] = len(rr_ms_full)
 
    annotations = parse_xml_events(xml_path)
 
    apnea_burden = {
        "n_apnea_events": 0,
        "n_hypopnea_events": 0,
        "total_apnea_duration_sec": 0.0,
        "n_arousal_events": 0,
        "n_plm_events": 0,
        "apnea_index_from_xml": np.nan,
    }
 
    sleep_rr   = None
    wake_rr    = None
    apnea_rr   = None
    clean_rr   = None
 
    if annotations:
        sleep_stages  = annotations["sleep_stages"]
        apnea_events  = annotations["apnea_events"]
        arousal_events= annotations["arousal_events"]
        plm_events    = annotations["plm_events"]
 
        apnea_burden["n_arousal_events"] = len(arousal_events)
        apnea_burden["n_plm_events"]     = len(plm_events)
 
        n_apnea   = sum(1 for _, _, t in apnea_events if "apnea" in t.lower())
        n_hypopnea= sum(1 for _, _, t in apnea_events if "hypopnea" in t.lower())
        total_dur = sum(d for _, d, _ in apnea_events)
        apnea_burden["n_apnea_events"]          = n_apnea
        apnea_burden["n_hypopnea_events"]        = n_hypopnea
        apnea_burden["total_apnea_duration_sec"] = total_dur
 
        sleep_sec = sum(
            d for _, d, c in sleep_stages if c in [1, 2, 3, 4, 5]
        )
        if sleep_sec > 0:
            apnea_burden["apnea_index_from_xml"] = (
                (n_apnea + n_hypopnea) / (sleep_sec / 3600.0)
            )
 
        sleep_mask = np.zeros(len(rpeak_sec), dtype=bool)
        for start, dur, code in sleep_stages:
            if code in [1, 2, 3, 4, 5]:
                mask = (rpeak_sec >= start) & (rpeak_sec < start + dur)
                sleep_mask |= mask
 
        wake_mask = ~sleep_mask
 
        def rr_from_mask(mask):
            idx = np.where(mask)[0]
            if len(idx) < 2:
                return None
            rr = np.diff(rpeaks[idx]) / sfreq * 1000.0
            rr = rr[(rr > 300) & (rr < 2000)]
            return rr if len(rr) >= 10 else None
 
        sleep_rr = rr_from_mask(sleep_mask)
        wake_rr  = rr_from_mask(wake_mask)
 
        apnea_mask = np.zeros(len(rpeak_sec), dtype=bool)
        for start, dur, _ in apnea_events:
            mask = (rpeak_sec >= start) & (rpeak_sec < start + dur)
            apnea_mask |= mask
 
        clean_sleep_mask = sleep_mask & ~apnea_mask
        apnea_sleep_mask = sleep_mask & apnea_mask
 
        apnea_rr = rr_from_mask(apnea_sleep_mask)
        clean_rr = rr_from_mask(clean_sleep_mask)
 
    result = {**overall, **apnea_burden}
 
    sleep_hrv = hrv_from_rr(sleep_rr) if sleep_rr is not None else {}
    for k, v in sleep_hrv.items():
        result[f"sleep_{k}"] = v
 
    wake_hrv = hrv_from_rr(wake_rr) if wake_rr is not None else {}
    for k, v in wake_hrv.items():
        result[f"wake_{k}"] = v
 
    apnea_hrv = hrv_from_rr(apnea_rr) if apnea_rr is not None else {}
    for k, v in apnea_hrv.items():
        result[f"apnea_{k}"] = v
 
    clean_hrv = hrv_from_rr(clean_rr) if clean_rr is not None else {}
    for k, v in clean_hrv.items():
        result[f"clean_{k}"] = v
 
    if apnea_hrv and clean_hrv:
        for k in ["SDNN", "RMSSD", "HF_power", "LF_HF"]:
            a = apnea_hrv.get(k)
            c = clean_hrv.get(k)
            if a is not None and c is not None and c > 0:
                result[f"apnea_vs_clean_{k}_ratio"] = a / c
 
    return result
 
 
print("=" * 70)
print("  TASK 1 — HRV Analysis & Survival Prediction  |  SHHS-1")
print("=" * 70)
 
outcomes = pd.read_csv(OUTCOMES_CSV)
outcomes["event_observed"] = (outcomes["vital"] == 0).astype(int)
n_events = outcomes["event_observed"].sum()
print(f"\n  Subjects : {len(outcomes)}")
print(f"  Deaths   : {n_events}  ({n_events/len(outcomes)*100:.1f}%)")
print(f"  Censored : {len(outcomes) - n_events}")
print(f"  Output   : {OUTPUT_DIR}/\n")
 
records = []
for i, row in outcomes.iterrows():
    sid      = int(row["nsrrid"])
    edf_path = os.path.join(DATA_DIR, f"shhs1-{sid}.edf")
    xml_path = os.path.join(DATA_DIR, f"shhs1-{sid}-nsrr.xml")
    n        = i + 1
 
    if not os.path.exists(edf_path):
        print(f"  [{n:02d}/50] Subject {sid} — EDF not found, skipping")
        continue
 
    if not os.path.exists(xml_path):
        print(f"  [{n:02d}/50] Subject {sid} — XML not found, skipping")
        continue
 
    print(f"  [{n:02d}/50] Subject {sid} ...", end=" ", flush=True)
    hrv = compute_hrv(edf_path, xml_path)
 
    if hrv is None:
        print("FAILED")
        continue
 
    hrv["nsrrid"]         = sid
    hrv["duration"]       = row["censdate"]
    hrv["event_observed"] = row["event_observed"]
    records.append(hrv)
    print(
        f"SDNN={hrv['SDNN']:.1f}ms  RMSSD={hrv['RMSSD']:.1f}ms  "
        f"pNN50={hrv['pNN50']:.1f}%  LF={hrv['LF_power']:.4f}  "
        f"HF={hrv['HF_power']:.4f}  apneas={hrv['n_apnea_events']}  "
        f"arousals={hrv['n_arousal_events']}"
    )
 
hrv_df = pd.DataFrame(records)
hrv_df.to_csv(os.path.join(OUTPUT_DIR, "hrv_features.csv"), index=False)
print(f"\n  Processed {len(hrv_df)}/50 subjects")
print(f"  Saved -> {OUTPUT_DIR}/hrv_features.csv\n")
 
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
print(f"  Merged with clinical covariates -> {OUTPUT_DIR}/hrv_plus_covariates.csv\n")
 
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
 
hrv_features = ["SDNN", "RMSSD", "pNN50", "LF_power", "HF_power",
                "LF_HF", "Total_power", "SD1", "SD2"]
xml_features = ["n_apnea_events", "n_arousal_events", "n_plm_events",
                "apnea_index_from_xml"]
cov_features = ["age", "female", "bmi", "ahi", "avg_spo2",
                "sleep_efficiency", "hypertension"]
 
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
    "AIC":        [cox_A.AIC_, cox_B.AIC_, cox_C.AIC_],
}).to_csv(os.path.join(OUTPUT_DIR, "model_performance.csv"), index=False)
print(f"  Performance table saved -> model_performance.csv")
 
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"""
  Subjects processed  : {len(hrv_df)}/50
  Data sources used   :
    EDF  -> ECG channel -> full-night R-peak detection -> 16 HRV metrics
    XML  -> sleep stages, apnea events, arousals, limb movements
            -> stage-specific HRV (sleep vs wake)
            -> apnea-specific HRV (during vs clean epochs)
            -> apnea burden metrics (counts, duration, index)
    CSV  -> age, sex, BMI, AHI, SpO2, sleep efficiency, hypertension
 
  HRV metrics (16)    : SDNN, RMSSD, pNN50, CV_RR, mean_HR
                        VLF, LF, HF, Total Power, LF/HF, LF_nu, HF_nu
                        SD1, SD2, SD1/SD2
 
  Survival models     :
    Model A (HRV only)            C-index = {ci_A:.4f}
    Model B (HRV + XML)           C-index = {ci_B:.4f}
    Model C (HRV + XML + clinical)C-index = {ci_C:.4f}
 
  Outputs in          : {OUTPUT_DIR}/
""")