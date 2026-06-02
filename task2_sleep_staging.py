import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mne
import yasa
import xmltodict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    cohen_kappa_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

DATA_DIR       = "./dataset/dataset"
OUTCOMES_CSV   = "./outcomes.csv"
COVARIATES_CSV = "./shhs1-dataset-0.21.0-subsampled.csv"
OUTPUT_DIR     = "./task2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

EPOCH_SEC   = 30
STAGE_ORDER = ["W", "N1", "N2", "N3", "R"]

# Stages 3 and 4 are both deep NREM (N3) per AASM 2007 rules
STAGE_MAP = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "N3", 5: "R", 9: None}

YASA_LABEL_MAP = {
    "WAKE": "W",
    "N1":   "N1",
    "N2":   "N2",
    "N3":   "N3",
    "REM":  "R",
    "W":    "W",
    "R":    "R",
}


def parse_expert_stages(xml_path):
    """
    Parse SHHS XML annotation file and return a list of expert sleep stage
    labels, one per 30-second epoch. Epochs with code 9 (movement/unknown)
    are returned as None and excluded from evaluation.
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
        print(f"\n    [WARN] Unexpected XML structure in {xml_path}")
        return None

    if isinstance(events, dict):
        events = [events]

    stage_events = []
    for ev in events:
        if ev.get("EventType", "") != "Stages|Stages":
            continue
        concept = ev.get("EventConcept", "")
        parts   = concept.split("|")
        if len(parts) != 2:
            continue
        try:
            code     = int(parts[1])
            start    = float(ev.get("Start", 0))
            duration = float(ev.get("Duration", EPOCH_SEC))
            n_epochs = max(1, int(round(duration / EPOCH_SEC)))
            label    = STAGE_MAP.get(code)
            stage_events.append((start, n_epochs, label))
        except (ValueError, TypeError):
            continue

    stage_events.sort(key=lambda x: x[0])
    labels = []
    for _, n_epochs, label in stage_events:
        labels.extend([label] * n_epochs)

    return labels if labels else None


def run_yasa_staging(edf_path, age=None, male=None):
    """
    Load EDF and run YASA's pre-trained LightGBM sleep staging model.

    YASA uses:
      - EEG (required): central electrode, preferably C4-A1 or EEG
      - EOG (optional): improves REM detection significantly
      - EMG (optional): marginal improvement per YASA publication
      - age + sex metadata (optional): improves accuracy

    SHHS channel names:
      EEG  -> 'EEG' (C4-A1 reference) or 'EEG(sec)' (C3-A2 reference)
      EOG  -> 'EOG(L)' or 'EOG(R)'
      EMG  -> 'EMG'

    Important: MNE must be told the correct channel types before passing
    to YASA, otherwise EMG gets misidentified as EEG.

    Returns list of predicted stage strings per 30-second epoch.
    """
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    except Exception as e:
        print(f"\n    [ERR] Cannot load EDF: {e}")
        return None, None

    ch_lower = {ch.lower(): ch for ch in raw.ch_names}

    # EEG channel — prefer C4-A1 (primary) over C3-A2 (secondary)
    eeg_ch = None
    for cand in ["eeg", "eeg(sec)", "c4-a1", "c3-a2", "c4", "c3"]:
        if cand in ch_lower:
            eeg_ch = ch_lower[cand]
            break
    if eeg_ch is None:
        matches = [ch for ch in raw.ch_names if "eeg" in ch.lower()]
        eeg_ch  = matches[0] if matches else None
    if eeg_ch is None:
        print(f"\n    [WARN] No EEG channel found. Available: {raw.ch_names}")
        return None, None

    # EOG channel — prefer left eye (LOC)
    eog_ch = None
    for cand in ["eog(l)", "loc-a2", "eog(r)", "roc-a1", "loc", "roc"]:
        if cand in ch_lower:
            eog_ch = ch_lower[cand]
            break
    if eog_ch is None:
        matches = [ch for ch in raw.ch_names
                   if "eog" in ch.lower() or "loc" in ch.lower()]
        eog_ch  = matches[0] if matches else None

    # EMG channel
    emg_ch = None
    for cand in ["emg", "chin1-chin2", "chin"]:
        if cand in ch_lower:
            emg_ch = ch_lower[cand]
            break
    if emg_ch is None:
        matches = [ch for ch in raw.ch_names
                   if "emg" in ch.lower() or "chin" in ch.lower()]
        emg_ch  = matches[0] if matches else None

    type_map = {}
    if eog_ch:
        type_map[eog_ch] = "eog"
    if emg_ch:
        type_map[emg_ch] = "emg"
    if type_map:
        raw.set_channel_types(type_map, verbose=False)

    print(f"\n    EEG={eeg_ch}  EOG={eog_ch}  EMG={emg_ch}", end="  ")

    metadata = {}
    if age is not None:
        metadata["age"] = int(age)
    if male is not None:
        metadata["male"] = bool(male)

    try:
        sls  = yasa.SleepStaging(
            raw,
            eeg_name=eeg_ch,
            eog_name=eog_ch,
            emg_name=emg_ch,
            metadata=metadata if metadata else None,
        )
        hyp  = sls.predict()
        # In YASA >= 0.6.4, predict() returns a Hypnogram object
        # hyp.hypno contains the string array of predicted stages
        if hasattr(hyp, "hypno"):
            pred_raw = list(hyp.hypno)
        else:
            pred_raw = list(hyp)
        pred = [YASA_LABEL_MAP.get(str(p), str(p)) for p in pred_raw]
        return pred, sls
    except Exception as e:
        print(f"\n    [ERR] YASA failed: {e}")
        return None, None


print("=" * 70)
print("  TASK 2 — Automatic Sleep Staging (YASA Pre-trained Model)")
print("  Sleep Heart Health Study (SHHS-1)")
print("=" * 70)

outcomes = pd.read_csv(OUTCOMES_CSV)

# Load age and sex from covariates to pass as metadata to YASA
cov = pd.read_csv(COVARIATES_CSV, usecols=["nsrrid", "age_s1", "gender"])
cov["male"] = (cov["gender"] == 1).astype(bool)
meta = dict(zip(cov["nsrrid"], zip(cov["age_s1"], cov["male"])))

subject_ids = outcomes["nsrrid"].astype(int).tolist()

all_true, all_pred     = [], []
per_subject_records    = []
n_processed            = 0

for n, sid in enumerate(subject_ids, 1):
    edf_path = os.path.join(DATA_DIR, f"shhs1-{sid}.edf")
    xml_path = os.path.join(DATA_DIR, f"shhs1-{sid}-nsrr.xml")

    if not os.path.exists(edf_path) or not os.path.exists(xml_path):
        print(f"\n  [{n:02d}/50] Subject {sid} — missing files, skipping")
        continue

    print(f"\n  [{n:02d}/50] Subject {sid}", end="")

    age, male = meta.get(sid, (None, None))

    true_labels = parse_expert_stages(xml_path)
    if true_labels is None:
        print(" — XML parsing failed, skipping")
        continue

    # Run YASA
    result = run_yasa_staging(edf_path, age=age, male=male)
    if result[0] is None:
        print(" — YASA failed, skipping")
        continue
    pred_labels, sls = result

    min_len     = min(len(true_labels), len(pred_labels))
    true_labels = true_labels[:min_len]
    pred_labels = pred_labels[:min_len]

    # Remove epochs where true label is None (movement/artifact code 9)
    pairs = [(t, p) for t, p in zip(true_labels, pred_labels)
             if t is not None]
    if not pairs:
        print(" — no valid epochs after filtering")
        continue

    t_valid, p_valid = zip(*pairs)
    t_valid = list(t_valid)
    p_valid = list(p_valid)

    all_true.extend(t_valid)
    all_pred.extend(p_valid)

    kappa = cohen_kappa_score(t_valid, p_valid)
    acc   = accuracy_score(t_valid, p_valid)
    f1    = f1_score(t_valid, p_valid, labels=STAGE_ORDER,
                     average="macro", zero_division=0)

    per_subject_records.append({
        "nsrrid":   sid,
        "n_epochs": len(t_valid),
        "kappa":    kappa,
        "accuracy": acc,
        "f1_macro": f1,
    })
    n_processed += 1
    print(f"  epochs={len(t_valid)}  κ={kappa:.3f}  acc={acc:.3f}  F1={f1:.3f}")

print(f"\n\n  Processed {n_processed}/50 subjects")
print(f"  Total epochs pooled: {len(all_true)}\n")

if not all_true:
    print("[ERROR] No epochs to evaluate. Check file paths.")
    exit(1)

labels_present = [s for s in STAGE_ORDER
                  if s in set(all_true) or s in set(all_pred)]

kappa_all   = cohen_kappa_score(all_true, all_pred)
acc_all     = accuracy_score(all_true, all_pred)
f1_macro    = f1_score(all_true, all_pred, labels=labels_present,
                       average="macro", zero_division=0)
f1_weighted = f1_score(all_true, all_pred, labels=labels_present,
                       average="weighted", zero_division=0)
prec_macro  = precision_score(all_true, all_pred, labels=labels_present,
                              average="macro", zero_division=0)
rec_macro   = recall_score(all_true, all_pred, labels=labels_present,
                           average="macro", zero_division=0)

print("=" * 70)
print("  AGGREGATE PERFORMANCE (pooled across all subjects)")
print("=" * 70)
print(f"\n  {'Metric':<25}  {'Value':>8}")
print("  " + "-" * 36)
print(f"  {'Accuracy':<25}  {acc_all:>8.4f}  ({acc_all*100:.1f}%)")
print(f"  {'Cohens Kappa':<25}  {kappa_all:>8.4f}  ({kappa_all:.4f})")
print(f"  {'F1 (macro)':<25}  {f1_macro:>8.4f}")
print(f"  {'F1 (weighted)':<25}  {f1_weighted:>8.4f}")
print(f"  {'Precision (macro)':<25}  {prec_macro:>8.4f}")
print(f"  {'Recall (macro)':<25}  {rec_macro:>8.4f}")

print(f"\n  Per-class Report:")
print(classification_report(all_true, all_pred,
                             labels=labels_present,
                             target_names=labels_present,
                             zero_division=0))

ps_df = pd.DataFrame(per_subject_records)
ps_df.to_csv(os.path.join(OUTPUT_DIR, "per_subject_metrics.csv"), index=False)

# Aggregate metrics
pd.DataFrame([
    {"metric": "Accuracy",          "value": acc_all},
    {"metric": "Cohens Kappa",      "value": kappa_all},
    {"metric": "F1 (macro)",        "value": f1_macro},
    {"metric": "F1 (weighted)",     "value": f1_weighted},
    {"metric": "Precision (macro)", "value": prec_macro},
    {"metric": "Recall (macro)",    "value": rec_macro},
]).to_csv(os.path.join(OUTPUT_DIR, "aggregate_metrics.csv"), index=False)
print(f"  Metrics saved -> {OUTPUT_DIR}/aggregate_metrics.csv")

# Figure 1: Confusion matrices (raw + normalised side by side)
cm     = confusion_matrix(all_true, all_pred, labels=labels_present)
cm_df  = pd.DataFrame(cm, index=labels_present, columns=labels_present)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm_df = pd.DataFrame(cm_norm, index=labels_present, columns=labels_present)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(
    f"Sleep Staging Confusion Matrix — YASA vs Expert  (SHHS-1)\n"
    f"κ={kappa_all:.3f}   Accuracy={acc_all*100:.1f}%   "
    f"F1(macro)={f1_macro:.3f}",
    fontsize=13, fontweight="bold"
)
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
            linewidths=0.5, ax=axes[0],
            cbar_kws={"label": "Epoch count"})
axes[0].set_xlabel("Predicted Stage", fontsize=11)
axes[0].set_ylabel("True Stage", fontsize=11)
axes[0].set_title("Raw Counts", fontsize=11, fontweight="bold")

sns.heatmap(cm_norm_df, annot=True, fmt=".2f", cmap="Blues",
            linewidths=0.5, ax=axes[1],
            cbar_kws={"label": "Recall (row-normalised)"})
axes[1].set_xlabel("Predicted Stage", fontsize=11)
axes[1].set_ylabel("True Stage", fontsize=11)
axes[1].set_title("Row-Normalised (Recall per Class)",
                  fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig1_confusion_matrix.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 1 saved -> fig1_confusion_matrix.png")

# Figure 2: Per-class Precision / F1 / Recall bar chart
per_f1   = f1_score(all_true, all_pred, labels=labels_present,
                    average=None, zero_division=0)
per_prec = precision_score(all_true, all_pred, labels=labels_present,
                           average=None, zero_division=0)
per_rec  = recall_score(all_true, all_pred, labels=labels_present,
                        average=None, zero_division=0)

x     = np.arange(len(labels_present))
width = 0.26
fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width, per_prec, width, label="Precision",
            color="#4C9BE8", alpha=0.85)
b2 = ax.bar(x,          per_f1,  width, label="F1-score",
            color="#4CAF50", alpha=0.85)
b3 = ax.bar(x + width,  per_rec, width, label="Recall",
            color="#FF9800", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels_present, fontsize=12)
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Per-class Precision / F1 / Recall\n(YASA vs Expert Sleep Staging)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.2f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig2_per_class_metrics.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 2 saved -> fig2_per_class_metrics.png")

# Figure 3: Per-subject kappa and accuracy histograms
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Per-subject Performance Distribution", fontsize=13,
             fontweight="bold")

axes[0].hist(ps_df["kappa"], bins=12, color="#4C9BE8",
             edgecolor="white", linewidth=0.8)
axes[0].axvline(ps_df["kappa"].mean(), color="red", linestyle="--",
                label=f"Mean κ={ps_df['kappa'].mean():.3f}")
axes[0].axvline(ps_df["kappa"].median(), color="orange", linestyle="--",
                label=f"Median κ={ps_df['kappa'].median():.3f}")
axes[0].set_xlabel("Cohen's Kappa", fontsize=11)
axes[0].set_ylabel("Number of subjects", fontsize=11)
axes[0].set_title("Cohen's Kappa Distribution", fontsize=11, fontweight="bold")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(ps_df["accuracy"], bins=12, color="#4CAF50",
             edgecolor="white", linewidth=0.8)
axes[1].axvline(ps_df["accuracy"].mean(), color="red", linestyle="--",
                label=f"Mean={ps_df['accuracy'].mean():.3f}")
axes[1].set_xlabel("Accuracy", fontsize=11)
axes[1].set_ylabel("Number of subjects", fontsize=11)
axes[1].set_title("Accuracy Distribution", fontsize=11, fontweight="bold")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig3_per_subject_distribution.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 3 saved -> fig3_per_subject_distribution.png")

# Figure 4: True vs Predicted stage proportion comparison
true_counts = pd.Series(all_true).value_counts().reindex(
    labels_present, fill_value=0)
pred_counts = pd.Series(all_pred).value_counts().reindex(
    labels_present, fill_value=0)
total = len(all_true)

fig, ax = plt.subplots(figsize=(9, 5))
x     = np.arange(len(labels_present))
width = 0.35
ax.bar(x - width/2, true_counts / total * 100, width,
       label="Expert (True)", color="#4C9BE8", alpha=0.85)
ax.bar(x + width/2, pred_counts / total * 100, width,
       label="YASA (Predicted)", color="#FF9800", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels_present, fontsize=12)
ax.set_ylabel("% of total epochs", fontsize=11)
ax.set_title("Sleep Stage Distribution: Expert vs YASA",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig4_stage_distribution.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 4 saved -> fig4_stage_distribution.png")

# Figure 5: Per-subject kappa scatter (ordered)
fig, ax = plt.subplots(figsize=(12, 4))
ps_sorted = ps_df.sort_values("kappa").reset_index(drop=True)
colors    = ["#E84C4C" if k < 0.4 else "#FF9800" if k < 0.6
             else "#4CAF50" for k in ps_sorted["kappa"]]
ax.bar(range(len(ps_sorted)), ps_sorted["kappa"],
       color=colors, alpha=0.85, edgecolor="white")
ax.axhline(0.4, color="red",    linestyle="--", linewidth=1,
           label="κ=0.4 (fair)")
ax.axhline(0.6, color="orange", linestyle="--", linewidth=1,
           label="κ=0.6 (good)")
ax.axhline(0.8, color="green",  linestyle="--", linewidth=1,
           label="κ=0.8 (excellent)")
ax.axhline(ps_df["kappa"].mean(), color="blue", linestyle="-",
           linewidth=1.5, label=f"Mean κ={ps_df['kappa'].mean():.3f}")
ax.set_xlabel("Subject (sorted by kappa)", fontsize=11)
ax.set_ylabel("Cohen's Kappa", fontsize=11)
ax.set_title("Per-subject Cohen's Kappa — YASA vs Expert Scoring",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.set_ylim(0, 1.0)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fig5_per_subject_kappa_ranked.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print(f"  Figure 5 saved -> fig5_per_subject_kappa_ranked.png")

print(f"""
{'='*70}
  TASK 2 SUMMARY
{'='*70}

  Subjects evaluated   : {n_processed}
  Total epochs         : {len(all_true)}
  Model used           : YASA pre-trained LightGBM (Vallat & Walker, eLife 2021)
  Inputs to YASA       : EEG (C4-A1) + EOG (LOC) + EMG + age + sex

  Aggregate Performance:
    Accuracy           : {acc_all:.4f}  ({acc_all*100:.1f}%)
    Cohens Kappa       : {kappa_all:.4f}
    F1 (macro)         : {f1_macro:.4f}
    F1 (weighted)      : {f1_weighted:.4f}
    Precision (macro)  : {prec_macro:.4f}
    Recall (macro)     : {rec_macro:.4f}

  Per-subject kappa    : mean={ps_df['kappa'].mean():.3f}
                         median={ps_df['kappa'].median():.3f}
                         min={ps_df['kappa'].min():.3f}
                         max={ps_df['kappa'].max():.3f}

  Outputs in           : {OUTPUT_DIR}/
    aggregate_metrics.csv
    per_subject_metrics.csv
    fig1_confusion_matrix.png
    fig2_per_class_metrics.png
    fig3_per_subject_distribution.png
    fig4_stage_distribution.png
    fig5_per_subject_kappa_ranked.png
""")
