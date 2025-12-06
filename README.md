# 🌾 🇮🇩 InSight Padi: National Rice Productivity Prediction Platform based on Data Integration

[![GitHub Repo Size](https://img.shields.io/github/repo-size/adhimurizkyr/insight-prediksi_panen_nasional?style=flat-square)](https://github.com/adhimurizkyr/insight-prediksi_panen_nasional)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?style=flat-square)](https://streamlit.io/)
[![Machine Learning](https://img.shields.io/badge/Model-Random%20Forest-2E8B57?style=flat-square)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellowgreen?style=flat-square)](https://opensource.org/licenses/MIT)

## ✍️ Project Overview

The **InSight Padi** project is an **end-to-end Data Pipeline** implementation aimed at predicting rice harvest yields to support national food security.

We built a system architecture that integrates historical agricultural data from **BPS** (Statistics Indonesia) with multi-variable weather data (temperature, rainfall, radiation) from **BMKG/Open-Meteo**. This enriched data, processed through *Feature Engineering*, is used to train a **Random Forest Regressor** model, which achieves an aggregate national accuracy of **~96.5%**.

The Streamlit dashboard (`app.py`) presents intuitive visualizations of trends, geographical distribution, and Actual vs. Prediction comparisons.

***

## ⚙️ System Architecture & Methodology

The primary focus of this project is on designing a **stable data pipeline** and strategic **feature engineering**.

### Tahapan Utama (Key Stages)

| Stage | Description | Key Components |
| :--- | :--- | :--- |
| **ETL (Transformation)** | Data alignment of historical BPS and Weather data. | `pandas`, Temporal Alignment |
| **Feature Engineering** | Creation of **Weather Lagging Features** (1-3 months prior) as key predictors. | `numpy`, Lagging Features |
| **Model Training** | Training a Regression model to predict the `produksi` (production in tons) value. | Random Forest Regressor, `scikit-learn` |
| **Deployment** | Serving prediction results and historical visualizations. | Streamlit Cloud, `app.py` |



[Image of machine learning model improvement steps]


### 2. Prediction Model
* **Model:** Random Forest Regressor (`scikit-learn`).
* **Target:** `produksi` (production in tons).
* **Aggregate National Accuracy:** **~96.5%** (based on the total Actual vs. Prediction difference in the test month).

***

## ✨ Key Dashboard Features (app.py)

The **InSight Padi** dashboard is designed for quick and detailed insights:

| Category | Feature | Description |
| :---: | :--- | :--- |
| **Prediction** | **National Prediction vs. Actual** | Comparison of predicted vs. actual production in a Grouped Bar Chart per province. |
| **Geospatial** | **Interactive Distribution Map** | Geographical visualization (Plotly Scatter Geo) showing rice production hotspots across Indonesia. |
| **Time Series** | **Time Trend Analysis** | Line graphs to visualize monthly rice production fluctuations at the provincial level. |
| **Detailed Analysis** | **Weather & Harvest Detail** | Separate tabs for analyzing specific weather metrics and harvest data at the selected location/month. |
| **Ranking** | **Top Producers Ranking** | Displays the top 10 rice-producing provinces (Actual Data). |

***

## 💻 Technology Stack

| Category | Technology |
| :--- | :--- |
| **Core Programming** | Python 3.9+ |
| **Data Science** | `pandas`, `numpy`, `scikit-learn` |
| **Model Persistence** | `joblib` |
| **Dashboard & UI** | `streamlit` |
| **Visualization** | `altair`, `plotly` |

### File `requirements.txt`:
```txt
streamlit
pandas
altair
joblib
numpy
plotly 
```

Step
1. Clone Repo,git clone https://github.com/adhimurizkyr/insight-prediksi_panen_nasional.git,Downloads all files from GitHub.
2. Enter Folder,cd insight-prediksi_panen_nasional,Navigates to the project directory.
3. Install Dependencies,pip install -r requirements.txt,Installs all required Python libraries.
4. Run App,streamlit run app.py,The application will open in your browser at http://localhost:8501.
