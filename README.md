# Chicago Crime Analysis & Prediction (2020–2022)

**Prepared by:** Tan Kai Boon  |  **GitHub:** [tKaiB/Capstone\_Chicago](https://github.com/tKaiB/Capstone_Chicago)  |  **Stack:** Python · XGBoost · Power BI · Scikit-Learn

## Overview

A comprehensive data science capstone studying **627,163 cleaned crime records** from the Chicago Police Department across 2020–2022. The project covers the full arc of the COVID-19 pandemic - from pre-lockdown baseline through acute lockdown, reopening, and post-lockdown recovery - and delivers both a Power BI analytics dashboard and XGBoost machine learning models for arrest prediction and domestic incident classification.

Insights derived from the exploratory analysis directly inform the predictive models, aimed at forecasting crime risk and supporting evidence-based public safety planning at the community-area level.

## Key Findings at a Glance

| Metric | Value |
| --- | --- |
| Total cleaned records | 627,163 (2020–2022) |
| Overall arrest rate | 13.25% (83K arrests / 544K non-arrests) |
| Domestic incident rate | 19.72% (124K flagged incidents) |
| Property crime share | 49.69% (dominant category) |
| Arrest rate 2020 → 2022 | 16.3% → 11.3% (declining despite rising volume) |
| Arrest prediction ROC-AUC | 0.9537 (XGBoost, test set) |
| Domestic prediction ROC-AUC | 0.9768 (XGBoost, test set) |
| Top crime hotspot | Area 25 — Austin (\~35,000 incidents) |
| Peak crime window | Friday–Sunday, Afternoon/Evening (12:00–23:59) |
| Lowest arrest rate crime | Deceptive Practice at 1.76% |

**Standout finding:** Domestic violence risk did not spike during the strict March–June 2020 lockdown (feature importance rank: 21st). Instead, the post-lockdown era of 2021–2022 drove sustained elevated risk — a 7.6× higher importance score - pointing to long-term economic strain and delayed reporting rather than a brief pandemic anomaly.

## Dataset

Source: **Chicago Police Department Incident Logs**, publicly available via the Chicago Data Portal.

| Attribute | Detail |
| --- | --- |
| Raw records | 639,406 incidents |
| Cleaned records | 627,163 rows × 12 columns |
| Dropped rows | \~12,243 (missing coordinates or ward IDs) |
| Date range | 2020–2022 (6 CSV partitions concatenated) |
| Pandemic window | Pre-lockdown → Lockdown → Reopening → Second Wave → Post-Lockdown |

The data includes crime type, arrest status, geographic coordinates, beat/district assignments, location descriptors, and domestic flags.

## Project Phases

### Phase 1 — Exploratory Data Analysis (`Cap_Chicago_Crime.ipynb`)

- Temporal heatmaps: day-of-week × time-of-day incident volumes
- Seasonal cyclicality analysis across 2020–2022
- Arrest and clearance rate breakdowns by crime type, district, and time window
- Crime category composition (violent vs. property vs. other) year-over-year
- Domestic incident distribution by location descriptor and community area
- KMeans spatial clustering (Elbow method) on 627K lat/long coordinate pairs
- Power BI dashboard integration for interactive spatial and temporal drill-down

### Phase 2 — Predictive Modelling (`Cap_Chicago_Prediction.ipynb`)

- **Target 1 — Arrest prediction:** Logistic Regression, Random Forest (depth=15), and XGBoost (n=100, depth=6) benchmarked on ROC-AUC; XGBoost selected at 0.9537 test AUC
- **Target 2 — Domestic incident classification:** XGBoost achieved 0.9768 test ROC-AUC with 92% balanced accuracy
- Class imbalance handled via SMOTE (13.2% positive → balanced 544K/544K split)
- Feature engineering: Time\_Of\_Day bins, 5-phase Pandemic\_Phase variable, 4-category Holiday\_Group
- Final feature set: 23 predictive inputs after dropping collinear District/Ward columns

## Repository Structure

```
Capstone_Chicago/
├── notebooks
  ├── Cap_Chicago_Crime.ipynb         # Phase 1: EDA, visualisations, KMeans clustering
  ├── Cap_Chicago_Prediction.ipynb    # Phase 2: ML pipeline, SMOTE, model benchmarking
├── chicago_crime_analysis_presentation.pdf 
└── README.md                       # This file
```

> **Start here:** Open `Cap_Chicago_Crime.ipynb` for the full EDA walkthrough, then `Cap_Chicago_Prediction.ipynb` for the modelling pipeline. The PDF presentation summarises all key findings.

## Tools & Tech Stack

| Tool / Library | Purpose |
| --- | --- |
| Python 3.12 | Core language |
| Pandas | Data cleaning, feature engineering |
| Scikit-Learn | Logistic Regression, Random Forest, preprocessing, SMOTE pipeline |
| Imbalanced-Learn | SMOTE oversampling for class balancing |
| XGBoost | Primary classifier (arrest & domestic prediction) |
| Matplotlib / Seaborn | EDA visualisations and ROC curve plots |
| Power BI | Interactive dashboards (spatial heatmaps, temporal drill-down) |
| Google Colab | Cloud notebook execution |

## Research Questions

1. **Spatial Distribution** — Which community areas exhibit the highest crime concentrations, and do hotspots remain spatially stable across 2020–2022?
2. **Temporal Patterns** — How do crime frequencies vary across day of week, time of day, and season, and are there identifiable cyclical patterns year-over-year?
3. **Arrest & Clearance Rates** — What crime types and locations are associated with lower arrest rates, and does arrest likelihood vary significantly by district or time period?
4. **Crime Type Dynamics** — How has the composition of crime types shifted between 2020 and 2022, and which categories show the most volatility?
5. **Domestic & Location Risk** — How do domestic-flagged incidents vary across community areas, and did the pandemic meaningfully alter these distributions?

## Limitations

- **COVID-19 Window Anomaly:** Models trained on 2020–2022 pandemic dynamics may need re-calibration against post-2023 baseline data.
- **SMOTE Synthetic Noise:** Oversampling minority classes creates synthetic feature combinations that may slightly inflate tree-based model performance metrics.
- **Binary Lockdown Simplification:** A binary `Lockdown_Period` flag fails to capture the nuance of the 5-phase `Pandemic_Phase` breakdown used in final models.

## Acknowledgements

Dataset sourced from the [Chicago Data Portal](https://data.cityofchicago.org/) — Chicago Police Department crime incident reports.
