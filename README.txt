# HOTEL REVENUE DECISION SUPPORT SYSTEM
## Application Prototype and Data Pipeline

**Course:** AA-5960 — Master's Research Project  
**Project:** Dynamic Hotel Room Pricing and Overbooking Optimization Decision-Support System  
**Group 11:** Chao-Ting Chueh, Bilal Syed

---

## 1. PROJECT OVERVIEW

This project develops a Decision Support System (DSS) designed to help hotel revenue managers analyze booking cancellations, evaluate overbooking risks, and make data-informed room inventory decisions.

The application combines historical synthetic hotel booking data, machine learning, and Monte Carlo simulation to evaluate alternative overbooking strategies.

The current prototype focuses on cancellation prediction and risk-constrained overbooking cost optimization. Dynamic room pricing recommendations are planned for future development.

## 2. APPLICATION FEATURES

The Streamlit application contains two main screens.

### Screen 1: Hotel Overview

Displays descriptive performance indicators for the selected hotel:

- Hotel room capacity
- Total reservations
- Historical cancellation rate
- Average booking ADR (Average Daily Rate)
- Historical monthly occupancy trends

Users can switch between the Resort Hotel and City Hotel.

### Screen 2: Overbooking Decision Support

Provides interactive analytical and simulation features:

**Booking Cancellation Prediction**
- Uses a trained Logistic Regression model.
- Estimates individual booking cancellation probabilities.
- Displays average predicted cancellation risk and expected non-cancelled bookings for the selected stay date.

**Dynamic Overbooking Simulation**
- Uses Monte Carlo simulation to estimate room occupancy outcomes.
- Evaluates additional booking quantities under different cancellation scenarios.
- Estimates walk probability and expected empty rooms.
- Recommends additional bookings based on expected cost minimization and a user-defined walk-risk constraint.

**Financial Analysis**
- Allows users to adjust assumed ADR.
- Allows adjustment of contribution margin.
- Allows adjustment of walk cost as a multiple of ADR.
- Calculates expected spoilage cost, walk cost, and total cost.
- Visualizes the relationship between additional bookings, costs, and walk risk.

**Baseline Comparison**
- Displays the original cancellation-rate-based overbooking recommendation.
- Provides a reference for evaluating the simulation-based approach.

## 3. DATA SOURCE AND GENERATION

The project uses a fully synthetic relational database calibrated against the real Hotel Booking Demand dataset published by Antonio et al. (2019).

The original dataset includes 119,390 booking records from two Portuguese hotels covering July 2015 through August 2017.

The real datasets are stored in:

- data/raw/H1.csv — Resort Hotel
- data/raw/H2.csv — City Hotel

The original records are used for calibration and validation. No original booking records are copied directly into the final synthetic database.

The synthetic data generator uses calibrated booking distributions, cancellation relationships, seasonality, and explicitly stated assumptions.

Key dataset characteristics:

- 60,000 synthetic booking records
- 11 relational CSV tables
- One SQLite database
- Reservation snapshots and daily occupancy records
- Simulated hotel capacity, walking events, and cost parameters

The two simulated hotels have room capacities of 156 and 102 rooms.

## 4. DATA VALIDATION

The original data pipeline includes 22 automated validation checks covering:

- Schema and identifier integrity
- Primary and foreign key relationships
- Missing and invalid values
- Booking chronology
- Cancellation and no-show rates
- Statistical consistency with calibration targets
- Reproducibility of generated data

The previous validated dataset achieved:

**22 PASS / 0 FAIL**

Selected calibration results:

- Resort cancellation rate: 28.0% (real target: 27.8%)
- City cancellation rate: 42.1% (real target: 41.7%)
- No-show rates: approximately 0.7%–1.1%
- Simulated walk-night rate: 1.75%

Validation outputs are available in the reports/ directory.

These checks establish the integrity and selected statistical properties of the synthetic dataset. They do not independently establish the real-world accuracy of the application’s overbooking recommendations.

## 5. MACHINE LEARNING MODEL

The application uses Logistic Regression to predict the probability that an individual hotel booking will be canceled.

Predictor variables include:

- Booking lead time
- Average daily rate (ADR)
- Deposit type
- Market segment
- Distribution channel
- Previous cancellations
- Repeat guest status
- Number of guests and special requests
- Length of stay
- Hotel identifier

The model excludes booking outcome fields from its predictors.

The initial model evaluation produced:

- ROC-AUC: 0.879
- Brier Score: 0.1239

The model is saved as:

cancellation_model.pkl

These results were obtained from the synthetic dataset. Additional calibration and decision-time validation are needed before operational use.

## 6. OVERBOOKING OPTIMIZATION

The application uses Monte Carlo simulation to evaluate different quantities of additional reservations.

For each candidate quantity, the simulation estimates:

- Probability of exceeding hotel room capacity
- Expected number of empty rooms
- Expected number of walked rooms
- Expected spoilage cost
- Expected walk cost
- Expected total cost

The decision objective is to minimize expected total cost subject to the maximum acceptable walk-risk constraint.

Expected Total Cost = Expected Spoilage Cost + Expected Walk Cost

The simulation uses the following adjustable assumptions:

- Contribution margin: default 70% of ADR
- Walk cost multiplier: default 2 × ADR
- Maximum acceptable walk risk: default 5%
- Monte Carlo simulation runs: 10,000
- Random seed: 42

The simulation's additional booking quantity represents the number of new reservations accepted beyond the existing scenario population. This differs from the standard overbooking buffer, defined as authorized reservations above physical room capacity.

## 7. PROJECT FILE STRUCTURE

    Hotel_DSS_Project/
    |
    |-- app.py
    |-- train_model.py
    |-- overbooking.py
    |-- cancellation_model.pkl
    |-- requirements.txt
    |-- README.txt
    |
    |-- code/
    |   |-- generate.py
    |   |-- validate.py
    |   |-- schema.sql
    |   |-- default.json
    |
    |-- data/
    |   |-- raw/
    |   |   |-- H1.csv
    |   |   |-- H2.csv
    |   |
    |   |-- final/
    |       |-- hotel_dss.db
    |       |-- 11 relational CSV tables
    |
    |-- evidence/
    |   |-- source_register.csv
    |   |-- calibration_register.csv
    |   |-- assumptions_register.csv
    |
    |-- oracle/
    |   |-- oracle_booking_truth.csv
    |   |-- truth_parameters.json
    |
    |-- reports/
    |   |-- quality_report.md
    |   |-- validation_metrics.csv
    |   |-- run_manifest.json
    |
    |-- run_manifest.json

The original Weeks 7–8 submission report is available separately as report.docx if included in the submission package.

## 8. SOFTWARE REQUIREMENTS

Recommended environment:

- Python 3.11 or 3.12
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- Joblib

SQLite3 is included with the standard Python library.

## 9. INSTALLATION AND APPLICATION EXECUTION

Open a terminal in the project root directory.

**Step 1: Create a Python virtual environment (recommended).**

    python3 -m venv .venv

On macOS:

    source .venv/bin/activate

**Step 2: Install dependencies.**

    python -m pip install -r requirements.txt

**Step 3: Confirm the dataset exists.**

Verify that the following file is present:

    data/final/hotel_dss.db

**Step 4: Train the cancellation model if the saved model is unavailable or needs rebuilding.**

    python train_model.py

This generates:

    cancellation_model.pkl

**Step 5: Start the application.**

    python -m streamlit run app.py

The application should open in a web browser.

Default local address:

    http://localhost:8501

**Step 6: Use the application.**

Select the hotel, switch between the two screens, and explore the historical data and simulated overbooking recommendations.

## 10. REPRODUCING THE DATA PIPELINE

To rebuild the synthetic dataset from the stored configuration:

    python code/generate.py

To validate the generated dataset:

    python code/validate.py

The original full-size dataset validation reported 22 PASS / 0 FAIL.

A smaller demonstration dataset can be generated using:

    python code/generate.py small

Note that generating a smaller dataset changes the database used by the application. Retrain the cancellation model after rebuilding the data to maintain consistency.

The generator uses calibration parameters stored in code/default.json. It does not independently recompute the calibration targets from the raw H1.csv and H2.csv files during each run.

The master seed and independent random streams support deterministic synthetic data generation. Reproducibility records and checksums are stored in the run manifests.

## 11. APPLICATION LIMITATIONS

The prototype has several limitations:

1. All final booking records are synthetic and may not generalize to current hotel operations.

2. The application does not connect to a live Property Management System (PMS).

3. The simulation assumes each reservation occupies one room.

4. Cancellation events are simulated independently.

5. Additional reservations are assumed to have cancellation risks similar to existing reservations.

6. The simulation does not separately model no-show behavior.

7. Financial cost parameters are based on adjustable assumptions rather than actual hotel accounting data.

8. Historical stay-date booking populations are not reconstructed using a specific decision-date booking ledger. Some bookings may not have been active at the corresponding historical decision time.

9. Multi-night room inventory is not jointly optimized across future stay dates.

10. The application does not yet implement a validated dynamic pricing optimization model.

Accordingly, recommendations should be interpreted as simulation-based decision-support demonstrations, not real operational hotel inventory instructions.

## 12. PROTOTYPE DEMONSTRATION

The application prototype is demonstrated in a short video covering:

- Selection of a hotel
- Historical occupancy and cancellation metrics
- Booking-level cancellation predictions
- Adjustment of the maximum acceptable walk risk
- Adjustment of walk cost assumptions
- Dynamic overbooking and expected cost results
- Comparison with the original fixed-cancellation-rate baseline

The demonstration illustrates how validated synthetic data can support an interactive decision-making workflow.

## 13. FUTURE DEVELOPMENT

Future work will focus on:

- Decision-date-aware active reservation filtering
- Improved cancellation probability calibration
- More realistic no-show modeling
- Multi-night inventory optimization
- Validation of simulated financial assumptions
- Direct evaluation against alternative overbooking policies
- Dynamic pricing recommendations
- Testing with real hotel operational data

## 14. REFERENCES

Antonio, N., de Almeida, A., & Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41–49. https://doi.org/10.1016/j.dib.2018.11.126

Antonio, N., de Almeida, A., & Nunes, L. (2019). An automated machine learning based decision support system to predict hotel booking cancellations. Data Science Journal, 18(1), 32. https://doi.org/10.5334/dsj-2019-032

Talluri, K. T., & van Ryzin, G. J. (2004). The theory and practice of revenue management. Springer.

---

**End of README**
---

## WEEK 10 UPDATE — Predictive Modeling & What-If Analysis

New screen: **Predictive Models & What-If**, covering two KPIs:
  KPI 1 — Booking cancellation probability (Gradient Boosting Classifier, ROC-AUC ~0.88)
  KPI 2 — Expected nightly occupancy rate (Gradient Boosting Regressor, R^2 ~0.18)

For each KPI the app shows:
  - overall predictor importance (mean |SHAP|, whole model),
  - an interactive what-if with a fixed baseline for comparison,
  - per-prediction SHAP contributions (this case only),
  - interpretation (direction / magnitude / meaning; association, not causation),
  - an auto-generated scenario summary (inputs, updated prediction, change, limitations).

New files: train_models_wk10.py, wk10_screens.py, occupancy_model.pkl,
requirements.txt, SETUP.md, Week10_Progress_Report.docx.
Run: see SETUP.md  (pip install -r requirements.txt ; streamlit run app.py)
