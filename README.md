# ⚡ Hospital Energy Analytics & Fractal Engine (PoC)

An interactive, production-ready Streamlit dashboard designed for Hospital Energy Management and Fractal Analysis. This Proof-of-Concept (PoC) combines domain-specific biomedical engineering insights with advanced mathematical tools (Higuchi Fractal Dimension & Hurst Exponent) to assess energy complexity, baseload behavior, and potential operational anomalies in healthcare facilities.

---

### Live Application
Access the deployed Streamlit application here:
[Hospital Energy Analytics Dashboard](https://hospital-energy-analytics-f2vzlkckoewcde6rzzq3pu.streamlit.app/)

---

## Key Features

* **Robust Data Ingestion Pipeline:**
  * Automatic CSV delimiter detection (comma `,` or semicolon `;`).
  * Flexible Column Mapping Configuration via sidebar expander.
  * Robust date-time parsing and numeric data sanitization.

* **Executive Dashboard & KPIs (Tab 1):**
  * Baseload identification using the 5th percentile ($P_{05}$) to reflect continuous 24/7 medical loads.
  * Peak Demand, Load Factor calculation, and financial/carbon emission projections.
  * Responsive metric cards with large-scale number formatting (k, M, B).

* **Interactive Time Series Analytics (Tab 2):**
  * Dynamic preset buttons relative to the dataset's date range (*Full Dataset, Last Day, Last 7 Days, Last 30 Days*).
  * Interactive Plotly visualizations for granular time-window filtering.

* **Fractal Analysis Engine (Tab 3):**
  * **Higuchi Fractal Dimension (HFD):** Signal integration via `np.cumsum` to convert Fractional Gaussian Noise (fGn) to Fractional Brownian Motion (fBm) for mathematically sound fractal analysis.
  * **Hurst Exponent ($H = 2 - FD$):** Assesses signal persistence ($H > 0.5$) vs anti-persistence ($H < 0.5$).
  * **Sensitivity Analysis Plot:** Evaluates scale invariance ($FD$ and $H$) across minimal scale windows $k \in [5, 30]$.

* **Advanced Energy Diagnostics (Tab 4):**
  * **24-Hour Consumption Heatmap:** Identifies night-time baseload anomalies and seasonal peaks.
  * **Load Duration Curve (LDC):** Essential for battery storage (BESS) sizing and solar PV curtailment analysis.

---

## Tech Stack

* **Framework:** Streamlit
* **Data Processing:** Pandas, NumPy
* **Data Visualization:** Plotly Express & Plotly Graph Objects
* **Language:** Python 3.10+

---

## Local Setup & Installation

To run this repository locally:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/G-Dimitrakakis/hospital-energy-analytics.git](https://github.com/G-Dimitrakakis/hospital-energy-analytics.git)
   cd hospital-energy-analytics
