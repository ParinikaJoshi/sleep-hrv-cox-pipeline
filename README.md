# Full-Night ECG HRV Pipeline for Sleep Apnea Mortality Prediction

**Domain:** Sleep Physiology · Biomedical Signal Processing  
**Tools:** Python · MNE · NeuroKit2 · Cox PH · LightGBM · YASA

---

## Overview

Built an end-to-end ECG processing pipeline for overnight polysomnography (PSG) recordings from the Sleep Heart Health Study (SHHS-1). The pipeline extracts time-domain, frequency-domain, and nonlinear HRV features across full-night recordings and links them to long-term mortality outcomes.

---

## What I Built

- Processed 50 overnight PSG recordings, extracting **16 HRV metrics** per recording using MNE and NeuroKit2
- Automated sleep stage classification using **YASA's LightGBM classifier** (Cohen's κ = 0.814 vs. expert annotation)
- Fit a **Cox Proportional Hazards model** on HRV features, achieving a **C-index = 0.835** for mortality prediction
- Modularized pipeline supports batch processing of additional SHHS recordings with minimal configuration

---

## Key Results

| Metric | Value |
|---|---|
| C-index (Cox PH, mortality) | **0.835** |
| Sleep staging agreement (Cohen's κ) | **0.814** |
| HRV features extracted | 16 (time, frequency, nonlinear domains) |
| Recordings processed | 50 overnight PSGs |

---

## Pipeline Architecture

```
Raw EDF/ECG → R-peak detection (MNE/NeuroKit2)
           → HRV feature extraction (16 metrics)
           → Sleep staging (YASA LightGBM)
           → Stage-stratified HRV summary
           → Cox PH survival model
           → C-index evaluation
```

---

## Technical Stack

| Component | Library/Method |
|---|---|
| Signal processing | MNE, NeuroKit2 |
| Sleep staging | YASA (LightGBM backend) |
| Survival modeling | lifelines (Cox PH) |
| Feature engineering | Custom HRV extraction module |
| Data format | EDF polysomnography files |

---

## HRV Features Extracted

**Time domain:** RMSSD, SDNN, pNN50, mean RR, CV-RR  
**Frequency domain:** LF power, HF power, LF/HF ratio, total power, VLF  
**Nonlinear:** SD1, SD2 (Poincaré), SampEn, DFA α1, ApEn, REC%

---

## Clinical Relevance

HRV suppression during sleep is an established marker of autonomic dysfunction and elevated cardiovascular mortality risk. This pipeline enables scalable, objective HRV profiling without manual annotation — directly applicable to real-world evidence generation from wearable and PSG data.

---

## Data Source

Sleep Heart Health Study (SHHS-1) — NHLBI BioLINCC  
