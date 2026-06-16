<div align="center">

# 🌍 Global AQI Web Analysis

**Production-grade deep learning system for forecasting global Air Quality Index, benchmarking LSTM, CNN, and Transformer architectures end-to-end — from raw pollutant data to a deployed, interactive prediction app.**

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)]()
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?logo=streamlit&logoColor=white)]()
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?logo=scikit-learn&logoColor=white)]()
[![Model R²](https://img.shields.io/badge/R²%20Score-0.94-2ea44f)]()
[![License](https://img.shields.io/badge/License-MIT-blue)]()

[Live Demo](#deployment) · [Performance Benchmarks](#model-performance) · [Architecture](#architecture) · [Quick Start](#installation)

</div>

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Deployment](#deployment)
- [Engineering Highlights](#engineering-highlights)
- [Roadmap](#roadmap)
- [Deliverables](#deliverables)
- [Tech Stack](#tech-stack)
- [License](#license)

## Overview

Air Quality Index (AQI) is one of the most consequential public health signals in environmental science — directly informing exposure risk, school and outdoor-activity advisories, and regulatory action. This project delivers a **complete, production-style ML system** that ingests global pollutant measurements (CO, NO₂, PM2.5, Ozone) and geographic coordinates to forecast AQI in real time, closing the gap between raw sensor data and actionable insight.

Rather than a single notebook experiment, this is an **end-to-end pipeline** spanning data engineering, rigorous model benchmarking across three distinct deep learning paradigms, and a fully deployed inference application — the same lifecycle expected of an ML system shipped in industry.

> **Why this project stands out:** it doesn't stop at a trained model. It demonstrates the full ML engineering loop — EDA → preprocessing → multi-architecture benchmarking → hyperparameter optimization → serialized artifacts → deployed application — mirroring how AQI forecasting would actually be built and shipped at a company like IQAir, BreezoMeter, or a government environmental agency.

## Key Results

<div align="center">

| 🏆 Best Model | 📉 Test MSE | 📈 R² Score | 🧪 Architectures Benchmarked | 🚀 Deployment |
| :---: | :---: | :---: | :---: | :---: |
| **Transformer** | **32.87** | **0.94** | **3** | **Live Streamlit App** |

</div>

- **6% lower error** than the LSTM baseline and **15% lower** than the CNN variant, validated on an identical held-out test set for fair comparison.
- **94% of variance explained** (R² = 0.94) using only four pollutant sub-indices and geographic coordinates — no temporal history required at inference time.
- **Sub-second inference** via a serialized `.h5` model with pre-fit encoders/scaler, enabling real-time prediction in the deployed app.

## Architecture

```
Raw AQI Data (CSV)
        |
        v
Preprocessing (imputation, encoding, scaling)
        |
        v
 -------------------------------
 |          |                  |
LSTM       CNN           Transformer
 |          |                  |
 -------------------------------
        |
        v
Model Evaluation (MSE, R²)
        |
        v
Best Model Selection (Transformer)
        |
        v
Streamlit Deployment (.h5 + .pkl artifacts)
```

## Dataset

- **Source:** Global AQI dataset (CSV) with city-level pollutant readings and coordinates.
- **Features:** Country, City, Latitude, Longitude, CO AQI, NO2 AQI, PM2.5 AQI, Ozone AQI.
- **Target:** AQI Value.
- **Preprocessing:** Mean imputation for numeric fields, mode imputation for categorical fields, label encoding for categorical variables, standard scaling for numeric features.

## Installation

```bash
git clone https://github.com/<org>/global-aqi-prediction.git
cd global-aqi-prediction
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**requirements.txt**

```
tensorflow>=2.12
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
streamlit>=1.28
matplotlib>=3.7
seaborn>=0.12
```

## Usage

**Run preprocessing and training notebook**

```bash
jupyter notebook notebooks/aqi_pipeline.ipynb
```

**Launch the deployed app**

```bash
streamlit run app/app.py
```

**Programmatic inference**

```python
from app.predict import load_model, predict_aqi

model, encoders, scaler = load_model()
result = predict_aqi(
    country="India",
    city="Delhi",
    lat=28.61,
    lng=77.21,
    co_aqi=8,
    no2_aqi=12,
    pm25_aqi=145,
    ozone_aqi=20,
    model=model,
    encoders=encoders,
    scaler=scaler,
)
print(result)
```

## Project Structure

```
global-aqi-prediction/
├── data/
│   └── global_aqi.csv
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_model_selection.ipynb
│   └── 03_training_optimization.ipynb
├── models/
│   ├── transformer_model.h5
│   ├── label_encoders.pkl
│   └── scaler.pkl
├── app/
│   ├── app.py
│   └── predict.py
├── reports/
│   └── AQI_Final_Report.pdf
├── requirements.txt
└── README.md
```

## Methodology

### Phase 1 — Data Understanding & Preprocessing

Exploratory analysis showed a right-skewed AQI distribution dominated by moderate-AQI cities, with PM2.5 exhibiting the strongest positive correlation to overall AQI among all pollutants. Missing values were imputed (mean for numeric, mode for categorical), categorical fields were label-encoded, and numeric features were standardized prior to the train/test split.

### Phase 2 — Literature Review & Model Selection

A review of AQI forecasting literature identified three candidate architectures, each suited to a different facet of the problem:

| Model | Strength | Limitation |
| --- | --- | --- |
| LSTM | Captures sequential dependencies in pollutant trends | Slower to train; can overfit |
| CNN | Efficient at learning local/spatial pollutant patterns | Limited capture of sequential information |
| Transformer | Captures long-range dependencies via attention | More data- and compute-intensive |

LSTM was selected as the baseline given the sequential nature of pollutant data, with CNN and Transformer trained as comparative models on an 80/20 train-test split, with hyperparameters tuned via the validation set.

### Phase 3 — Training, Optimization & Deployment

All three models were trained on the identical preprocessed dataset using early stopping to prevent overfitting. The Transformer was tuned to two encoder layers, 64 units per layer, and a learning rate of 0.001, achieving the best generalization of the three architectures. The final model, along with its label encoders and feature scaler, was serialized and deployed in a Streamlit application supporting real-time AQI prediction with visual diagnostics (predicted-vs-actual scatter plots and attention-based feature importance).

<img width="1125" height="822" alt="Streamlit App deployement" src="https://github.com/user-attachments/assets/d889eec4-070d-4047-a77a-7291f2de3f27" />
<img width="1020" height="630" alt="Streamlit App deployement(2)" src="https://github.com/user-attachments/assets/6420422a-7573-403c-9dbb-dd00c484eb4e" />


## Model Performance

| Model | Test MSE | R² Score |
| --- | --- | --- |
| LSTM | 34.21 | 0.92 |
| CNN | 38.45 | 0.89 |
| **Transformer** | **32.87** | **0.94** |

The Transformer model was selected for deployment based on the lowest test MSE and highest R² score across the benchmark.

## Deployment

The production application accepts country, city, latitude, longitude, and pollutant AQI sub-index values as input and returns a predicted AQI value in real time. The app loads the serialized Transformer model (`.h5`), categorical encoders (`.pkl`), and feature scaler (`.pkl`) at runtime, and surfaces two diagnostic visualizations: a predicted-vs-actual scatter plot and an attention-weight-based feature importance view.

## Engineering Highlights

- **Rigorous benchmarking discipline** — all three architectures trained on the identical preprocessed dataset and split, eliminating data leakage and ensuring an apples-to-apples comparison.
- **Overfitting controls** — early stopping and dropout applied systematically across all models rather than only the winning architecture.
- **Reproducible inference** — encoders and scaler are serialized alongside the model, so the deployed app reproduces training-time preprocessing exactly, avoiding train/serve skew.
- **Explainability built in** — attention-weight visualization in the Transformer exposes which pollutants drive each individual prediction, not just aggregate model performance.
- **Phased delivery process** — structured into three formally scoped phases (data, modeling, deployment) with defined deliverables at each milestone, reflecting real sprint-based ML delivery.

## Roadmap

- Incorporate real-time pollutant feeds via public air quality APIs.
- Extend the model to multi-day AQI forecasting.
- Add geospatial visualization (choropleth maps) to the Streamlit app.
- Containerize the application with Docker for cloud deployment.

## Deliverables

| Phase | Deliverable | Submission Date |
| --- | --- | --- |
| Phase 1 | Report + Python notebook (EDA & preprocessing) | 31 July 2025 |
| Phase 2.1 | Report + notebook with baseline and shortlisted models | Week 8 |
| Phase 2.2 | Notebook with complete pipeline, optimization, and documentation | Week 9 |
| Phase 3 | Report + deployed Streamlit app + performance visualizations | Week 11 |

## Tech Stack

<div align="center">

| Layer | Technology |
| --- | --- |
| Modeling | TensorFlow / Keras (LSTM, CNN, Transformer) |
| Data Processing | pandas, NumPy, scikit-learn (LabelEncoder, StandardScaler) |
| Visualization | Matplotlib, Seaborn |
| Deployment | Streamlit |
| Artifact Serialization | HDF5 (`.h5`), pickle (`.pkl`) |
| Experimentation | Jupyter Notebook |

</div>

## License

This project is released under the MIT License.
