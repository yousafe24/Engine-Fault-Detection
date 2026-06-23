# Engine Fault Detection — Unsupervised Anomaly Detection
### Autoencoder vs PCA vs Isolation Forest on EngineFaultDB

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)](https://scikit-learn.org)
[![Dataset](https://img.shields.io/badge/Dataset-IEEE%20Access%202023-red.svg)](https://doi.org/10.1109/ACCESS.2023.3331316)

---

## Overview

This project addresses automotive engine fault detection as an **unsupervised anomaly detection problem**. All models are trained exclusively on healthy engine data — no fault samples are used during training — simulating real-world deployment conditions where labeled fault data is rarely available.

Three approaches are compared:

| Model | Type | Anomaly Signal |
|---|---|---|
| PCA | Linear representation (baseline) | Reconstruction error |
| Isolation Forest | Ensemble / decision trees (baseline) | Average path length in trees |
| **Autoencoder** | Non-linear representation (method) | Reconstruction error |

**Research question:** *Can engine faults be detected without using fault data during training, and does a non-linear learned representation outperform classical linear and tree-based approaches for this task?*

---

## Dataset

**EngineFaultDB** — Vergara et al., IEEE Access 2023  
DOI: [10.1109/ACCESS.2023.3331316](https://doi.org/10.1109/ACCESS.2023.3331316)

Data collected from a **C14NE spark ignition engine** under controlled laboratory conditions using an NGA 6000 gas analyzer and a USB 6008 data acquisition card.

| Label | Fault Type | Description | Samples |
|---|---|---|---|
| 0 | No fault | Normal engine operation | 16,000 |
| 1 | Fault type 1 | Rich mixture (high fuel pressure, clogged air filter, defective injector) | 10,998 |
| 2 | Fault type 2 | Lean mixture (low fuel pressure, faulty pressure regulator) | 15,000 |
| 3 | Fault type 3 | Low voltage (worn spark plugs, faulty ignition cables, defective coil) | 14,001 |

**14 sensor features:** MAP · TPS · Force · Power · RPM · Consumption L/H · Consumption L/100KM · Speed · CO · HC · CO2 · O2 · Lambda · AFR

---

## Project Structure
engine-fault-detection/

│

├── data/

│   └── EngineFaultDB_Final.csv

│

├── notebooks/

│   ├── 1-eda.ipynb

│   ├── 2-baseline-pca.ipynb

│   ├── 3-baseline-isolation-forest.ipynb

│   └── 4-autoencoder.ipynb

│

├── figures/

│   ├── class_distribution.png

│   ├── correlation_heatmap.png

│   ├── feature_distributions.png

│   ├── bottleneck_tuning.png

│   ├── training_loss.png

│   ├── reconstruction_error_dist.png

│   ├── threshold_optimization.png

│   ├── ae_confusion_matrix.png

│   ├── if_confusion_matrix.png

│   ├── pca_confusion_matrix.png

│   └── roc_curve_comparison.png

│

├── models/

│   ├── autoencoder.keras

│   ├── pca_baseline.pkl

│   └── isolation_forest.pkl

│

├── requirements.txt

└── README.md

---

## Methodology

### CRISP-DM Pipeline
Business Understanding → Data Understanding → Data Preparation

↓

Modeling (PCA · Isolation Forest · Autoencoder)

↓

Evaluation (ROC-AUC · F1 · Recall · Confusion Matrix)

### Core Design Decisions

**1 — Train on healthy data only**  
All three models are fitted exclusively on 12,800 healthy training samples. The normalization pipeline is also fitted on healthy data only to prevent data leakage.

**2 — Realistic evaluation set**  
The original dataset has roughly equal class sizes, which would create a 93% fault test set — unrealistic for deployment. Fault samples are subsampled to 200 per type (~17% fault rate), reflecting real-world anomaly rarity.

**3 — Threshold optimization**  
Rather than hardcoding a threshold (e.g. mean + 1σ), all percentile thresholds are swept and the optimal F1 threshold is selected from the data. Identical strategy applied to all three models for fair comparison.

**4 — Bottleneck tuning**  
The autoencoder bottleneck size is tuned by training five models (N = 2, 3, 4, 5, 6 neurons) and selecting the one with the lowest validation reconstruction loss.

**5 — Early stopping**  
`patience=30`, `min_delta=1e-4`. The model exhibited late convergence beyond epoch 500, making a fixed epoch budget insufficient. Early stopping found the right stopping point automatically.

---

## Results

| Model | ROC-AUC | F1 | Recall | Precision |
|---|---|---|---|---|
| Isolation Forest (baseline) | ~0.73 | ~0.55 | ~0.60 | ~0.51 |
| PCA (baseline) | ~0.85 | ~0.72 | ~0.68 | ~0.77 |
| **Autoencoder (method)** | **~0.93** | **~0.80** | **~0.83** | **~0.78** |
| Random Forest — paper † | — | 0.748 | 0.748 | 0.748 |

† Original paper result uses supervised classification with full access to labeled fault data during training.

**Primary metric is ROC-AUC** — threshold-independent, robust to class imbalance, and comparable across models with different anomaly score scales.

**Key result:** The autoencoder trained with zero fault data achieves F1 ≈ 0.80, competitive with the supervised Random Forest from the original paper (F1 = 0.748) which had full access to labeled fault data.

---

## Setup

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/engine-fault-detection
cd engine-fault-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download EngineFaultDB_Final.csv from:
# https://github.com/Leo-Thomas/EngineFaultDB
# Place it in the data/ folder

# 4. Run notebooks in order
# 1-eda.ipynb
# 2-baseline-pca.ipynb
# 3-baseline-isolation-forest.ipynb
# 4-autoencoder.ipynb  ← loads saved baselines for final comparison
```

---

## Requirements
tensorflow>=2.12

scikit-learn>=1.3

numpy>=1.24

pandas>=2.0

matplotlib>=3.7

seaborn>=0.12

joblib>=1.3

jupyter>=1.0

---

## Learnings

**Evaluation set design is critical.**  
Using all fault samples creates a test set that is 93% faulty — not representative of real engine monitoring where faults are rare. The threshold learned on such a set would perform poorly in deployment.

**Threshold selection matters more than expected.**  
The standard approach of hardcoding mean + one standard deviation is arbitrary. Sweeping all percentile thresholds and selecting by F1 produces a data-driven, justifiable decision boundary.

**Late convergence is real.**  
The autoencoder continued improving beyond 500 epochs before converging. A fixed epoch budget would have stopped training too early. Early stopping with a patience parameter resolved this automatically.

**PCA as the direct baseline.**  
PCA and the autoencoder follow identical pipelines — compress, reconstruct, measure error. The only difference is linearity. This makes the comparison clean and the research question answerable: the AUC gap between PCA and the autoencoder is attributable entirely to the non-linearity of the learned representation.

---

## Limitations

- The dataset was designed for supervised classification — fault classes are well-populated and balanced. True anomaly detection benchmarks typically have fault rates under 5%.
- The autoencoder performs binary detection only — it cannot identify which fault type is present.
- The healthy test set is used both for early stopping validation and final evaluation. A three-way split (train / validation / test) would provide cleaner separation.

---

## Future Work

- Apply the pipeline to a genuinely rare-anomaly dataset such as NASA CMAPSS turbofan engine degradation data
- Extend to a Variational Autoencoder (VAE) for probabilistic anomaly scoring
- Add per-fault-type detection rate analysis
- Integrate real-time streaming inference for live engine monitoring

---

## Citation

```bibtex
@ARTICLE{10311597,
  author={Vergara, Mary and Ramos, Leo and Rivera-Campoverde, Néstor Diego and Rivas-Echeverría, Francklin},
  journal={IEEE Access},
  title={EngineFaultDB: A Novel Dataset for Automotive Engine Fault Classification and Baseline Results},
  year={2023},
  volume={11},
  pages={126155-126171},
  doi={10.1109/ACCESS.2023.3331316}
}
```



https://github.com/user-attachments/assets/093555b0-c8fa-4a35-abd6-a9863367de78






