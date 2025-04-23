## Slide 1: Title Slide

**Quantitative Signal Discovery: Finding Potential Stock Predictors in FRED Data**

* Leveraging Python for Financial Data Analysis
* [Your Name/Date]

---

## Slide 2: Introduction & Goal

**Objective:**

* To systematically identify and evaluate potential leading economic indicators (signals) from the FRED database that might predict the behavior of a target stock.

**Why?**

* Inform quantitative trading strategies.
* Enhance macroeconomic analysis related to specific equities.
* Provide a data-driven approach to feature selection for predictive models.

**Tool:**

* A Python script automating data fetching, statistical analysis, and signal ranking.

---

## Slide 3: Core Workflow Overview

1.  **Setup:** Configure API keys, define target stock & dates.
2.  **Fetch Target Data:** Get stock price/return data (yfinance).
3.  **Fetch Potential Predictors:** Get Top N popular FRED series IDs.
4.  **Fetch Predictor Data:** Get observations for each FRED series.
5.  **Align Data:** Create a unified daily time series dataset.
6.  **Analyze Relationships:**
    * Test Stationarity (ADF) & Difference if needed.
    * Test Cointegration (Engle-Granger).
    * Find Optimal Lag (VAR + AIC).
    * Test Granger Causality (at optimal lag).
    * Calculate Mutual Information & Correlations.
7.  **Rank Signals:** Score predictors based on statistical significance and predictive potential.
8.  **Model (Optional):** Run baseline predictive models (Prophet, XGBoost) on the top signal.
9.  **Output:** Display and save ranked results.

---

## Slide 4: Step 1: Setup & Configuration

**Goal:** Prepare the script environment and parameters.

**How:**

* **FRED API Key:** Securely loaded from environment variables (`os.getenv("FRED_API_KEY")`). *Crucial for accessing FRED data.*
* **Command-Line Arguments (`argparse`):**
    * `ticker`: Target stock symbol (e.g., 'AAPL').
    * `start_date`, `end_date`: Analysis period.
    * `--fetch_top_n`: How many popular FRED series to retrieve (default 1000, max 1000).
    * `--use_log_returns`: Option to use log returns (often better for modeling) instead of prices.
    * `--max_granger_lag`: Max lag to check for VAR/Granger tests.
    * `--output`: Optional file to save results.
    * `--run_models`, `--forecast_horizon`, `--xgboost_lags`: Optional modeling parameters.
* **Logging:** Configured for informative status updates (`logging` module).
* **Constants:** Defined for API limits, p-value thresholds, etc.

---

## Slide 5: Step 2: Data Fetching (Stock Data)

**Goal:** Retrieve historical price data for the target stock.

**How:**

* **Library:** `yfinance`
* **Function:** `fetch_stock_data()`
* **Process:**
    * Uses `yf.Ticker(ticker).history()` to get daily data.
    * Selects 'Adj Close' (accounts for dividends/splits).
    * Optionally calculates log returns (`np.log(data / data.shift(1))`) for better stationarity properties.
    * Sets a daily frequency (`asfreq('D')`), filling non-trading days with NaNs initially.
    * Ensures timezone-naive index.

---

## Slide 6: Step 3: Data Fetching (FRED Popular Series IDs)

**Goal:** Identify a broad set of potential predictor series from FRED without requiring specific user searches.

**How:**

* **Function:** `search_fred_popular_series()`
* **API Endpoint:** `fred/series/search`
* **Workaround:**
    * Uses a generic `search_text` (e.g., "data").
    * Sorts results by `popularity` (descending).
    * Uses the `limit` parameter (capped at the API max of 1000) specified by `--fetch_top_n`.
    * This retrieves the IDs of the (up to) 1000 most popular series matching the generic term.

* **Result:** A list of FRED Series IDs (e.g., 'GDP', 'CPIAUCSL', 'UNRATE') to investigate further.

---

## Slide 7: Step 4: Data Fetching (FRED Observations)

**Goal:** Retrieve the actual time series data (observations) for each identified FRED series ID.

**How:**

* **Function:** `fetch_fred_series_data()`
* **API Endpoint:** `fred/series/observations`
* **Process:**
    * Iterates through the list of FRED Series IDs.
    * For each ID, calls the API endpoint specifying the `series_id`, `api_key`, `observation_start`, and `observation_end`.
    * Parses the JSON response.
    * Converts values to numeric (handling FRED's '.' for missing data).
    * Sets the date as the index.
    * Respects API rate limits using `time.sleep()`.

* **Result:** A dictionary where keys are Series IDs and values are Pandas Series containing the time series data.

---

## Slide 8: Step 5: Data Alignment & Preparation

**Goal:** Combine the target stock data and all fetched FRED series into a single DataFrame with a consistent daily index.

**How:**

* **Function:** `align_and_prepare_data()`
* **Process:**
    * Creates a master daily date range based on the target stock's data extent.
    * Reindexes the stock data to this daily index.
    * Iterates through fetched FRED series:
        * Resamples each FRED series to daily frequency using **forward fill (`ffill()`)**. *This carries the last known value forward, a common way to handle differing frequencies, but introduces assumptions.*
        * Reindexes the daily FRED series to the master index.
    * Drops rows where the *target* stock data is missing (typically weekends/holidays).
    * Drops any FRED columns that are entirely empty after alignment.

* **Result:** A single Pandas DataFrame (`combined_df`) ready for analysis.

---

## Slide 9: Step 6: Statistical Analysis - Stationarity

**Goal:** Determine if the time series have constant statistical properties (mean, variance) over time. *Crucial for reliable correlation, Granger causality, and many models.*

**How:**

* **Test:** Augmented Dickey-Fuller (ADF) Test (`statsmodels.tsa.stattools.adfuller`)
* **Function:** `test_stationarity()`
* **Process:**
    * Applied to both target and predictor series (levels first).
    * **Null Hypothesis (H0):** The series has a unit root (is non-stationary).
    * **Interpretation:** If p-value < threshold (e.g., 0.05), reject H0 -> series is likely stationary.
    * **Action:** If a series (target or predictor) is found non-stationary, its **first difference** (`series.diff().dropna()`) is used for subsequent Granger/correlation tests.
    * An ADF test is also run *after* differencing as a confirmation check.

---

## Slide 10: Step 7: Statistical Analysis - Cointegration

**Goal:** Check for a stable, long-run equilibrium relationship between two *non-stationary* series. (If they wander, do they wander together?)

**How:**

* **Test:** Engle-Granger Cointegration Test (`statsmodels.tsa.stattools.coint`)
* **Function:** Applied within `calculate_relationship_metrics()`
* **Process:**
    * Performed *only* if both the target and predictor series were found to be non-stationary in their levels via ADF.
    * **Null Hypothesis (H0):** The series are NOT cointegrated.
    * **Interpretation:** If p-value < threshold (e.g., 0.05), reject H0 -> series are likely cointegrated.
* **Significance:** A cointegrating relationship suggests the *levels* of the predictor might be useful for predicting the *levels* of the target, even if their short-term changes (returns/differences) aren't strongly related.

---

## Slide 11: Step 8: Statistical Analysis - Optimal Lag Selection

**Goal:** Determine the most appropriate time lag for testing Granger causality, avoiding arbitrary choices.

**How:**

* **Method:** Fit Vector Autoregression (VAR) models for different lags and select the lag minimizing an information criterion.
* **Criterion:** Akaike Information Criterion (AIC) - balances model fit and complexity, often preferred for prediction. (`statsmodels.tsa.api.VAR`)
* **Function:** `find_optimal_lag_var()`
* **Process:**
    * Uses the (potentially differenced) stationary data for the target and predictor.
    * Fits bivariate VAR(p) models for lags p = 1 up to `max_granger_lag`.
    * Calculates AIC for each lag.
    * Selects the lag `k*` with the lowest AIC value.

* **Result:** `optimal_lag_aic` (the lag `k*`) used in the next step.

---

## Slide 12: Step 9: Statistical Analysis - Granger Causality

**Goal:** Test if past values of the predictor series statistically help predict future values of the target series, *given past values of the target itself*. (Does predictor X "Granger-cause" target Y?)

**How:**

* **Test:** Granger Causality F-test (`statsmodels.tsa.stattools.grangercausalitytests`)
* **Function:** Applied within `calculate_relationship_metrics()`
* **Process:**
    * Uses the (potentially differenced) stationary data.
    * Performs the test specifically at the `optimal_lag_aic` determined previously.
    * **Null Hypothesis (H0):** The predictor series does NOT Granger-cause the target series.
    * **Interpretation:** If p-value < threshold (e.g., 0.05), reject H0 -> evidence suggests the predictor *does* Granger-cause the target at that lag.

* **Note:** Granger causality is about statistical predictability, not necessarily true economic causation.

---

## Slide 13: Step 10: Statistical Analysis - Correlations & Mutual Info

**Goal:** Measure other forms of association between the target and predictor.

**How:**

* **Metrics calculated within `calculate_relationship_metrics()` on (potentially differenced) data:**
    * **Pearson Correlation (`scipy.stats.pearsonr`):** Measures *linear* association (lag 0-3). Ranges from -1 to 1. Assumes normality.
    * **Spearman Rank Correlation (`scipy.stats.spearmanr`):** Measures *monotonic* association (non-linear, based on ranks) (lag 0-3). Ranges from -1 to 1. Non-parametric.
    * **Mutual Information (`sklearn.feature_selection.mutual_info_regression`):** Measures *general* dependence (linear and non-linear). Value >= 0 (0 means independence). Non-parametric.

* **Purpose:** Provide alternative views of the relationship, especially capturing non-linearities (MI, Spearman) or simple linear links (Pearson).

---

## Slide 14: Step 11: Signal Ranking

**Goal:** Combine the various statistical metrics into a single score to rank potential predictors based on their likely predictive utility.

**How:**

* **Function:** `rank_signals()`
* **Process:**
    * Uses a weighted scoring system (`weights` dictionary).
    * **Prioritizes:**
        * Low Granger p-value at optimal lag (strong negative weight).
        * Low Cointegration p-value (strong negative weight).
        * High Mutual Information (positive weight).
        * High absolute correlation at lags 0 and 1 (positive weight).
    * Calculates a final `score` for each FRED series.
    * Sorts the results DataFrame by score (descending).

* **Result:** A ranked list of the most promising predictor signals according to the defined criteria.

---

## Slide 15: Step 12: (Optional) Predictive Modeling - Prophet

**Goal:** Provide a baseline forecast using the top-ranked predictor as an external regressor in Facebook Prophet.

**How:**

* **Library:** `prophet`
* **Function:** `run_prophet_model()`
* **Process:**
    * Uses the *original* (non-differenced, non-scaled) target data (`y`) and top predictor data.
    * Initializes Prophet model (`m = Prophet()`).
    * Adds the predictor as a regressor (`m.add_regressor()`).
    * Fits the model (`m.fit()`).
    * Creates future dataframe (`m.make_future_dataframe()`).
    * **Challenge:** Requires future values of the regressor. Uses forward-fill (`ffill()`) as a simple assumption.
    * Predicts (`m.predict()`) and displays forecast (`yhat`, `yhat_lower`, `yhat_upper`).

---

## Slide 16: Step 13: (Optional) Predictive Modeling - XGBoost

**Goal:** Provide a baseline forecast using a gradient boosting model (XGBoost) with lagged values of the top predictor.

**How:**

* **Library:** `xgboost`
* **Function:** `run_xgboost_model()`
* **Process:**
    * Applies transformations (differencing/scaling) based on stationarity tests (`apply_transformations()`). Scaling is generally recommended.
    * Creates lagged features from the predictor series (e.g., predictor at t-1, t-2, ...).
    * **Time Series Cross-Validation (`sklearn.model_selection.TimeSeriesSplit`):** Crucial for evaluating time series models without lookahead bias. Splits data sequentially (train on past, test on immediate future).
    * Trains an `xgb.XGBRegressor` model on each training fold.
    * Evaluates using RMSE and MAE on the corresponding test fold.
    * Reports average cross-validated performance metrics.

---

## Slide 17: Step 14: Output & Results

**Goal:** Present the findings to the user.

**How:**

* **Display Top N:** Prints a formatted table (`logging.info()`, `pandas.to_string()`) showing the top N ranked signals and key metrics (rank, score, optimal lag, Granger p-value, cointegration status, MI, correlations, ADF results).
* **Save Results (Optional):** If an output file (`--output`) is specified:
    * Saves the full ranked DataFrame to a CSV or Excel file (`ranked_df.to_csv()`, `ranked_df.to_excel()`).

---

## Slide 18: Conclusion & Summary

**Recap:**

* Automated workflow to find potential equity predictors from FRED.
* Fetched stock and popular FRED data.
* Aligned time series to daily frequency.
* Applied rigorous statistical tests:
    * Stationarity (ADF) & Differencing
    * Cointegration
    * Optimal Lag Selection (AIC)
    * Granger Causality
    * Correlation & Mutual Information
* Ranked signals based on predictive potential.
* (Optional) Demonstrated baseline modeling with Prophet & XGBoost.

**Key Takeaways:**

* Provides a data-driven starting point for feature engineering.
* Highlights statistically significant relationships (Granger, Cointegration).
* Ranking helps prioritize further investigation.

---

## Slide 19: Q&A / Next Steps

**Discussion Points:**

* Interpretation of specific signals found.
* Refining the ranking weights.
* Exploring different lag structures or transformation methods.
* Robustness checks (different time periods, parameter sensitivity).

**Next Steps:**

* In-depth analysis of top-ranked signals (economic rationale).
* Feature engineering based on these signals for more complex predictive models.
* Out-of-sample testing and validation of models built using these signals.
* Consideration of transaction costs and implementation details for trading strategies.

