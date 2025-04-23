# --- Dependencies ---
# Suggestion: Create a requirements.txt file with:
# yfinance pandas numpy statsmodels scikit-learn scipy requests prophet xgboost openpyxl
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from statsmodels.tsa.api import VAR # Import VAR model
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from prophet import Prophet
import xgboost as xgb

import time
import requests
import warnings
import os
import logging
import argparse
from typing import Optional, List, Dict, Tuple, Any

# --- Configuration ---
warnings.filterwarnings("ignore") # Suppress specific warnings if needed, but be cautious

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
FRED_API_BASE_URL_SEARCH = "https://api.stlouisfed.org/fred/series/search"
FRED_API_BASE_URL_OBS    = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_DELAY_SECONDS    = 0.3 # Be respectful of API limits
MIN_OBS_FOR_VAR          = 50  # Min observations needed AFTER alignment/differencing for VAR/Granger
DEFAULT_GRANGER_MAX_LAG_SEARCH = 10  # Default max lag to search for optimal Granger lag using AIC/BIC
DEFAULT_FETCH_TOP_N      = 1000 # Default number of top popular series to fetch
FRED_API_SEARCH_LIMIT    = 1000 # Maximum allowed limit by FRED API for series search
ADF_P_VALUE_THRESHOLD    = 0.05 # Significance level for stationarity testing
COINT_P_VALUE_THRESHOLD  = 0.05 # Significance level for cointegration testing
GENERIC_SEARCH_TERM      = "data" # Term used to fetch broadly popular series

# --- Helper Functions ---

def get_fred_api_key() -> Optional[str]:
    """Retrieves the FRED API key from environment variables."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logging.error("FRED_API_KEY environment variable not set.")
        logging.info("Please set the FRED_API_KEY environment variable with your key.")
        logging.info("You can get a key from: https://fred.stlouisfed.org/docs/api/api_key.html")
        return None
    return api_key

# Modified: Added max_lag_search parameter
def test_stationarity(series: pd.Series, name: str = "Series", max_lag_search: int = DEFAULT_GRANGER_MAX_LAG_SEARCH) -> Tuple[float, bool]:
    """
    Performs the Augmented Dickey-Fuller test for stationarity.

    Args:
        series (pd.Series): The time series to test.
        name (str): Name of the series for logging.
        max_lag_search (int): The maximum lag used elsewhere (influences min data needed).

    Returns:
        Tuple[float, bool]: ADF p-value and boolean indicating stationarity (True if stationary).
    """
    # Ensure series is float type for ADF test
    try:
        series_float = series.astype(float)
    except ValueError:
        logging.warning(f"Could not convert series '{name}' to float for ADF test. Skipping.")
        return np.nan, False

    cleaned_series = series_float.dropna()

    # Use max_lag_search passed as parameter
    if len(cleaned_series) < max(15, max_lag_search + 5): # Need enough points for ADF and later VAR
        logging.debug(f"Insufficient data ({len(cleaned_series)}) for stationarity test on {name}.")
        return np.nan, False # Cannot determine

    # Check for constant series after dropping NaNs
    if np.ptp(cleaned_series.values) == 0:
         logging.debug(f"Series '{name}' is constant after dropna. Treating as non-stationary.")
         return 1.0, False # Treat constant series as non-stationary

    try:
        result = adfuller(cleaned_series, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < ADF_P_VALUE_THRESHOLD
        logging.debug(f"ADF test for {name}: p-value={p_value:.4f}, Stationary={is_stationary}")
        return p_value, is_stationary
    except Exception as e:
        # Catch potential LinAlgError or other issues
        logging.error(f"Error during ADF test for {name}: {e}")
        return np.nan, False # Error during test

def find_optimal_lag_var(data: pd.DataFrame, max_lag: int) -> Tuple[Optional[int], Optional[float]]:
    """
    Finds the optimal lag for a VAR model using AIC.

    Args:
        data (pd.DataFrame): DataFrame with the two time series (potentially differenced).
                             Column order matters for Granger interpretation later.
        max_lag (int): The maximum lag order to test.

    Returns:
        Tuple[Optional[int], Optional[float]]: Optimal lag based on AIC, and the AIC value.
                                               Returns (None, None) if unable to find optimal lag.
    """
    best_aic = np.inf
    optimal_lag = None

    # Check if data has at least MIN_OBS_FOR_VAR observations
    if len(data) < MIN_OBS_FOR_VAR:
         logging.warning(f"Insufficient data ({len(data)} < {MIN_OBS_FOR_VAR}) for VAR analysis. Skipping lag search.")
         return None, None

    # Check if data has enough points relative to max_lag for the VAR estimation loop
    if len(data) < max_lag + 5: # Need sufficient data relative to max lag
        logging.warning(f"Insufficient data ({len(data)}) to reliably test VAR up to lag {max_lag}. Reducing max_lag for search.")
        # Adjust max_lag down if possible, otherwise return None
        max_lag = max(0, len(data) - 5) # Ensure max_lag is at least 0
        if max_lag == 0:
             logging.warning("Not enough data to test any VAR lags.")
             return None, None


    for lag in range(1, max_lag + 1):
        try:
            # Ensure enough data points for the lag: T >= k*p + 1 where T=obs, k=vars, p=lag
            # Since k=2, we need T >= 2*lag + 1. Add buffer.
            if len(data) < 2 * lag + 5:
                logging.debug(f"Skipping VAR lag {lag}: Insufficient data points ({len(data)}).")
                continue # Stop searching if data becomes insufficient for higher lags

            model = VAR(data)
            results = model.fit(maxlags=lag, ic=None) # Fit for specific lag
            current_aic = results.aic
            logging.debug(f"VAR lag={lag}, AIC={current_aic:.4f}")

            # Check if current_aic is valid (not NaN or Inf)
            if pd.notna(current_aic) and np.isfinite(current_aic):
                if current_aic < best_aic:
                    best_aic = current_aic
                    optimal_lag = lag
            else:
                logging.warning(f"Invalid AIC value ({current_aic}) obtained for lag {lag}. Skipping lag.")


        except np.linalg.LinAlgError:
             logging.warning(f"Linear Algebra Error fitting VAR for lag {lag}. Might indicate collinearity. Skipping lag.")
             continue # Skip this lag if model fitting fails
        except ValueError as ve:
             # Catch specific ValueError related to insufficient data if not caught above
             if "sufficient observations" in str(ve) or "degrees of freedom" in str(ve):
                 logging.warning(f"Value Error fitting VAR for lag {lag}: {ve}. Skipping lag.")
                 continue
             else:
                 logging.error(f"Unexpected ValueError fitting VAR for lag {lag}: {ve}")
                 continue # Skip lag on other value errors too
        except Exception as e:
            logging.error(f"Unexpected error fitting VAR for lag {lag}: {e}")
            continue # Skip lag on any other error

    if optimal_lag is None:
        logging.warning(f"Could not determine optimal VAR lag up to {max_lag}.")
        return None, None

    logging.debug(f"Optimal VAR lag selected: {optimal_lag} (AIC: {best_aic:.4f})")
    return optimal_lag, best_aic


# --- Data Fetching ---

def fetch_stock_data(ticker: str, start: str, end: str, use_log_returns: bool = False) -> Optional[pd.Series]:
    """Fetches daily Close or Log Returns for a stock ticker via yfinance."""
    logging.info(f"Fetching stock data for {ticker} from {start} to {end}...")
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start, end=end, auto_adjust=False)['Adj Close']
        if data.empty:
            logging.warning(f"No data returned for {ticker}.")
            return None

        data.index = pd.to_datetime(data.index).tz_localize(None)
        data = data.asfreq("D")

        series_name = f"{ticker}_AdjClose"
        if use_log_returns:
            # Ensure prices are positive before taking log
            if (data <= 0).any():
                logging.warning(f"Ticker {ticker} has non-positive prices. Cannot calculate log returns accurately. Using Adj Close instead.")
            else:
                data = np.log(data / data.shift(1))
                series_name = f"{ticker}_LogReturn"
                data = data.iloc[1:]

        return data.rename(series_name)

    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return None

# Modified: Cap limit at FRED_API_SEARCH_LIMIT
def search_fred_popular_series(api_key: str, limit: int) -> List[str]:
    """
    Searches FRED for the most popular series using a generic search term,
    respecting the API's maximum limit.

    Args:
        api_key (str): Your FRED API key.
        limit (int): The desired maximum number of popular series IDs.

    Returns:
        List[str]: A list of FRED series IDs, ordered by popularity (up to API max).
    """
    actual_limit = limit
    if limit > FRED_API_SEARCH_LIMIT:
        logging.warning(f"Requested limit ({limit}) exceeds FRED API maximum ({FRED_API_SEARCH_LIMIT}). Capping limit at {FRED_API_SEARCH_LIMIT}.")
        actual_limit = FRED_API_SEARCH_LIMIT

    logging.info(f"Searching FRED for top {actual_limit} popular series using term '{GENERIC_SEARCH_TERM}'...")
    params = {
        "search_text": GENERIC_SEARCH_TERM, # Use generic term
        "api_key": api_key,
        "file_type": "json",
        "limit": actual_limit, # Use capped limit
        "order_by": "popularity",
        "sort_order": "desc",
    }
    try:
        response = requests.get(FRED_API_BASE_URL_SEARCH, params=params)
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        items = response.json().get("seriess", []) # More robust key access
        series_ids = [item["id"] for item in items if "id" in item]
        logging.info(f"  → Found {len(series_ids)} series IDs.")
        return series_ids
    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Error searching FRED popular series: {e}")
    except Exception as e:
        logging.error(f"Error parsing FRED popular series search results: {e}")
    return []

def fetch_fred_series_data(series_id: str, api_key: str, start_date: str, end_date: str) -> Optional[pd.Series]:
    """Fetches observations for a single FRED series."""
    logging.debug(f"Fetching data for FRED series: {series_id}...")
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date,
        "observation_end": end_date,
        "aggregation_method": "avg",
    }
    try:
        response = requests.get(FRED_API_BASE_URL_OBS, params=params)
        response.raise_for_status()
        data = response.json()
        obs = data.get("observations", [])
        if not obs:
            logging.warning(f"No observations found for {series_id} in the specified date range.")
            return None

        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().set_index("date")

        if df.empty:
            logging.warning(f"No valid numeric observations found for {series_id} after cleaning.")
            return None

        logging.debug(f"  → Fetched {len(df)} observations for {series_id}")
        return df["value"].rename(series_id)

    except requests.exceptions.RequestException as e:
        logging.error(f"HTTP Error fetching data for {series_id}: {e}")
    except Exception as e:
        logging.error(f"Error processing data for {series_id}: {e}")
    return None

# --- Data Processing & Alignment ---

def align_and_prepare_data(target_series: pd.Series, fred_series_dict: Dict[str, pd.Series]) -> pd.DataFrame:
    """Aligns target and FRED series to a common daily index."""
    logging.info("Aligning all time series to a common daily index...")
    if target_series is None or target_series.empty:
        logging.error("Target series is empty, cannot proceed.")
        return pd.DataFrame()

    daily_idx = pd.date_range(target_series.index.min(), target_series.index.max(), freq='D')
    combined_df = pd.DataFrame(index=daily_idx)
    combined_df[target_series.name] = target_series.reindex(daily_idx)

    dropped_fred_count = 0
    for sid, series in fred_series_dict.items():
        if series is None or series.empty:
            dropped_fred_count += 1
            continue
        series.index = pd.to_datetime(series.index).tz_localize(None)
        daily_s = series.resample('D').ffill() # Forward fill primarily
        combined_df[sid] = daily_s.reindex(daily_idx)

    logging.info(f"Initial FRED series count: {len(fred_series_dict)}. Series dropped due to being empty: {dropped_fred_count}")
    combined_df = combined_df.dropna(axis=1, how='all')

    rows_before = len(combined_df)
    combined_df = combined_df.dropna(subset=[target_series.name])
    rows_after = len(combined_df)
    logging.info(f"Aligned DataFrame shape: {combined_df.shape}. Dropped {rows_before - rows_after} rows where target was NaN.")

    if combined_df.empty:
        logging.warning("Combined DataFrame is empty after alignment and dropping target NaNs.")

    return combined_df

# --- Statistical Analysis ---

# Modified: Added max_lag_search parameter
def calculate_relationship_metrics(target_series: pd.Series, predictor_series: pd.Series, max_lag_search: int) -> Optional[Dict[str, Any]]:
    """
    Calculates various relationship metrics between two time series,
    handling stationarity, using optimal lags for Granger, and geared towards prediction.

    Args:
        target_series (pd.Series): The target variable (e.g., stock returns/price).
        predictor_series (pd.Series): The potential predictor variable (e.g., FRED series).
        max_lag_search (int): The maximum lag to search for optimal Granger/VAR lag.

    Returns:
        Optional[Dict[str, Any]]: Dictionary of calculated metrics or None if calculation fails.
    """
    metrics = {}
    series_name = predictor_series.name
    target_name = target_series.name

    # 1. Initial Data Prep & Stationarity Tests on Levels
    combined_levels = pd.concat([target_series, predictor_series], axis=1).dropna()
    if len(combined_levels) < MIN_OBS_FOR_VAR: # Use stricter minimum for VAR/Granger
        logging.debug(f"Skipping {series_name}: Insufficient overlapping data points ({len(combined_levels)} < {MIN_OBS_FOR_VAR}) for level analysis.")
        return None

    t_level = combined_levels.iloc[:, 0]
    p_level = combined_levels.iloc[:, 1]

    if np.ptp(t_level.values) == 0 or np.ptp(p_level.values) == 0:
         logging.warning(f"Skipping {series_name}: Target or predictor series is constant after alignment.")
         return None

    # Pass max_lag_search to test_stationarity
    metrics['adf_p_target_level'], target_is_stationary = test_stationarity(t_level, target_name, max_lag_search)
    metrics['adf_p_predictor_level'], p_is_stationary = test_stationarity(p_level, series_name, max_lag_search)

    # Handle cases where ADF test failed (returned NaN) - treat as non-stationary for safety
    if pd.isna(metrics['adf_p_target_level']): target_is_stationary = False
    if pd.isna(metrics['adf_p_predictor_level']): p_is_stationary = False


    # 2. Cointegration Test (only if both series are non-stationary on levels)
    metrics['coint_p_value'] = np.nan
    metrics['is_cointegrated'] = False
    if not target_is_stationary and not p_is_stationary:
        try:
            # Ensure series are of sufficient length for cointegration test
            if len(t_level) > MIN_OBS_FOR_VAR: # Cointegration might need more points
                 # Ensure data is float before cointegration test
                 t_level_float = t_level.astype(float)
                 p_level_float = p_level.astype(float)
                 score, p_value, _ = coint(t_level_float, p_level_float, trend='c') # 'c' includes constant term
                 metrics['coint_p_value'] = p_value
                 metrics['is_cointegrated'] = p_value < COINT_P_VALUE_THRESHOLD
                 logging.debug(f"Cointegration test for {series_name}: p-value={p_value:.4f}")
            else:
                 logging.debug(f"Skipping cointegration test for {series_name}: insufficient length after dropna.")
        except Exception as e:
            logging.warning(f"Error during cointegration test for {series_name}: {e}")


    # 3. Prepare Data for Correlation/Granger (Difference if non-stationary)
    # Also perform stationarity test *on the differenced series* if differencing is applied
    metrics['adf_p_target_diff'] = np.nan
    metrics['adf_p_predictor_diff'] = np.nan
    needs_diff = False

    if target_is_stationary and p_is_stationary:
        logging.debug(f"Using levels for correlation/Granger for {series_name} (both stationary).")
        analysis_df = combined_levels
    else:
        logging.debug(f"Using first differences for correlation/Granger for {series_name} (at least one non-stationary).")
        needs_diff = True
        analysis_df = combined_levels.diff().dropna()

        if len(analysis_df) >= MIN_OBS_FOR_VAR:
             # Test stationarity *after* differencing, passing max_lag_search
             metrics['adf_p_target_diff'], _ = test_stationarity(analysis_df.iloc[:, 0], f"{target_name}_diff", max_lag_search)
             metrics['adf_p_predictor_diff'], _ = test_stationarity(analysis_df.iloc[:, 1], f"{series_name}_diff", max_lag_search)
             # Check if differencing actually helped achieve stationarity (useful info, though we proceed anyway)
             if not pd.isna(metrics['adf_p_target_diff']) and metrics['adf_p_target_diff'] >= ADF_P_VALUE_THRESHOLD:
                 logging.warning(f"Target series {target_name} may still be non-stationary after differencing (p={metrics['adf_p_target_diff']:.4f}).")
             if not pd.isna(metrics['adf_p_predictor_diff']) and metrics['adf_p_predictor_diff'] >= ADF_P_VALUE_THRESHOLD:
                 logging.warning(f"Predictor series {series_name} may still be non-stationary after differencing (p={metrics['adf_p_predictor_diff']:.4f}).")
        else:
             logging.debug(f"Skipping post-difference stationarity check for {series_name}: insufficient data.")


    # Ensure enough data points *after potential differencing* for VAR/Granger
    if len(analysis_df) < MIN_OBS_FOR_VAR:
        logging.debug(f"Skipping {series_name}: Insufficient data points after differencing ({len(analysis_df)} < {MIN_OBS_FOR_VAR}).")
        # Return metrics calculated so far (ADF levels, potentially Cointegration)
        return metrics if any(not np.isnan(v) for v in metrics.values() if isinstance(v, (int, float))) else None


    t_analysis = analysis_df.iloc[:, 0]
    p_analysis = analysis_df.iloc[:, 1]

    # Check for constant series again after potential differencing
    if np.ptp(t_analysis.values) == 0 or np.ptp(p_analysis.values) == 0:
         logging.warning(f"Skipping {series_name}: Target or predictor series became constant after differencing.")
         return metrics if any(not np.isnan(v) for v in metrics.values() if isinstance(v, (int, float))) else None


    # 4. Mutual Information (on potentially differenced data)
    try:
        p_reshaped = p_analysis.values.reshape(-1, 1)
        t_ravelled = t_analysis.values.ravel()
        metrics["mutual_info"] = mutual_info_regression(p_reshaped, t_ravelled, discrete_features=False)[0]
    except Exception as e:
        logging.warning(f"Could not calculate Mutual Information for {series_name}: {e}")
        metrics["mutual_info"] = np.nan

    # 5. Optimal Lag Selection & Granger Causality (on potentially differenced data)
    metrics["optimal_lag_aic"] = np.nan
    metrics["granger_p_value_optimal_lag"] = np.nan
    metrics["granger_f_value_optimal_lag"] = np.nan

    # Ensure data is float for VAR
    try:
        var_data = analysis_df[[target_name, series_name]].astype(float)
    except ValueError:
        logging.error(f"Could not convert analysis data to float for VAR/Granger on {series_name}. Skipping.")
        # Return metrics calculated so far
        return metrics if any(not np.isnan(v) for v in metrics.values() if isinstance(v, (int, float))) else None

    # Pass max_lag_search to find_optimal_lag_var
    optimal_lag, _ = find_optimal_lag_var(var_data, max_lag_search)

    if optimal_lag is not None and optimal_lag > 0:
        metrics["optimal_lag_aic"] = optimal_lag
        # Run Granger test at the optimal lag
        try:
            # Test if predictor Granger-causes target (column index 1 -> column index 0)
            # Ensure enough observations for the chosen optimal lag
            if len(var_data) < optimal_lag + 5:
                 logging.warning(f"Insufficient data ({len(var_data)}) for Granger test at optimal lag {optimal_lag} for {series_name}. Skipping.")
            else:
                gc_result = grangercausalitytests(var_data, maxlag=[optimal_lag], verbose=False)

                # --- Start Fix ---
                # Check if the result for the optimal lag exists and has the expected structure
                if optimal_lag in gc_result and isinstance(gc_result[optimal_lag], tuple) and len(gc_result[optimal_lag]) > 0:
                    test_results_dict = gc_result[optimal_lag][0] # Get the dictionary of test results

                    if isinstance(test_results_dict, dict) and 'ssr_ftest' in test_results_dict:
                        ssr_ftest_result = test_results_dict['ssr_ftest']

                        # Check if ssr_ftest_result has the expected format (tuple/list of length 4)
                        if isinstance(ssr_ftest_result, (list, tuple)) and len(ssr_ftest_result) == 4:
                            # Correctly unpack all 4 values
                            f_val, p_val, _, _ = ssr_ftest_result
                            metrics["granger_f_value_optimal_lag"] = f_val
                            metrics["granger_p_value_optimal_lag"] = p_val
                            logging.debug(f"Granger test ({series_name} -> {target_name}, optimal_lag={optimal_lag}): p={p_val:.4f}")
                        else:
                            logging.warning(f"Granger test ({series_name}, lag={optimal_lag}): 'ssr_ftest' result had unexpected format: {ssr_ftest_result}")
                    else:
                        logging.warning(f"Granger test ({series_name}, lag={optimal_lag}): 'ssr_ftest' key not found in results dictionary.")
                else:
                    logging.warning(f"Granger test ({series_name}, lag={optimal_lag}): Result structure unexpected: {gc_result.get(optimal_lag)}")
                # --- End Fix ---

        except (ValueError, np.linalg.LinAlgError) as e:
             # This will catch errors during the grangercausalitytests call itself,
             # or potentially other ValueErrors if the structure checks above fail unexpectedly.
             logging.warning(f"Granger causality processing failed for {series_name} at optimal lag {optimal_lag}: {e}")
        except Exception as e:
             # Catch any other unexpected errors during the process
             logging.error(f"Unexpected error during Granger processing for {series_name} at lag {optimal_lag}: {e}")
    else:
        logging.debug(f"Skipping Granger causality for {series_name}: Could not determine optimal lag or optimal lag is 0.")


    # 6. Lagged Correlations (on potentially differenced data)
    for lag in range(4): # Lags 0, 1, 2, 3
        p_lagged = p_analysis.shift(lag)
        corr_df = pd.concat([t_analysis, p_lagged], axis=1).dropna()

        if len(corr_df) > 2:
            a = corr_df.iloc[:, 0].values
            b = corr_df.iloc[:, 1].values
            if np.ptp(a) == 0 or np.ptp(b) == 0:
                logging.debug(f"Skipping lag {lag} correlation for {series_name}: constant array after lagging.")
                metrics[f"pearson_lag_{lag}"] = np.nan
                metrics[f"spearman_lag_{lag}"] = np.nan
                continue
            try:
                metrics[f"pearson_lag_{lag}"] = pearsonr(a, b)[0]
            except ValueError: metrics[f"pearson_lag_{lag}"] = np.nan
            try:
                metrics[f"spearman_lag_{lag}"] = spearmanr(a, b)[0]
            except ValueError: metrics[f"spearman_lag_{lag}"] = np.nan
        else:
            metrics[f"pearson_lag_{lag}"] = np.nan
            metrics[f"spearman_lag_{lag}"] = np.nan

    return metrics

# --- Ranking ---

def rank_signals(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Ranks potential predictor signals based on calculated metrics,
    tuned for selecting features for prediction.
    """
    logging.info("Ranking signals based on metrics (tuned for prediction)...")
    if results_df.empty:
        logging.warning("Results DataFrame is empty, cannot rank signals.")
        return results_df

    # Define weights emphasizing predictive potential:
    weights = {
        'granger_p_value_optimal_lag': -0.40, # High importance, negative weight
        'coint_p_value':               -0.25, # Important for long-run, negative weight
        'mutual_info':                  0.20, # Captures non-linear links
        'pearson_lag_0':                0.10, # Contemporary linear link
        'spearman_lag_0':               0.10, # Contemporary non-parametric link
        'pearson_lag_1':                0.05, # Short-lag linear link
        'spearman_lag_1':               0.05, # Short-lag non-parametric link
    }

    ranked = results_df.copy()
    ranked['score'] = 0.0

    for metric, weight in weights.items():
        if metric in ranked.columns:
            col_data = ranked[metric]
            if pd.api.types.is_numeric_dtype(col_data): # Ensure column is numeric
                if 'p_value' in metric or 'adf_p' in metric: # Lower is better for p-values
                    score_contribution = col_data.fillna(1.0) * weight
                else: # Higher absolute value is better for MI, correlations
                    score_contribution = col_data.fillna(0.0).abs() * weight

                if score_contribution.isnull().any():
                     logging.warning(f"NaNs found in score contribution for metric '{metric}'. Filling with 0.")
                     score_contribution = score_contribution.fillna(0.0)

                ranked['score'] += score_contribution
            else:
                 logging.warning(f"Metric '{metric}' is not numeric. Skipping for scoring.")

        else:
            logging.debug(f"Metric '{metric}' not found in results, skipping for scoring.")


    # Sort by score descending
    ranked = ranked.sort_values(by='score', ascending=False)
    ranked['rank'] = range(1, len(ranked) + 1)

    if not ranked.empty:
        logging.info(f"Ranking complete. Top signal: {ranked.index[0]} with score {ranked['score'].iloc[0]:.4f}")
    else:
        logging.info("Ranking complete, but no signals were scored.")
    return ranked


# --- Predictive Modeling ---

def apply_transformations(df: pd.DataFrame, cols_to_transform: List[str], difference: bool = False, scale: bool = False) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """Applies differencing and/or scaling to specified columns."""
    df_t = df.copy()
    scaler = None

    if difference:
        logging.info(f"Applying first difference to columns: {cols_to_transform}")
        df_t[cols_to_transform] = df_t[cols_to_transform].diff()
        df_t = df_t.dropna(subset=cols_to_transform)

    if scale:
        logging.info(f"Applying standard scaling to columns: {cols_to_transform}")
        # Ensure data is float before scaling
        float_df = df_t[cols_to_transform].astype(float)
        if float_df.isnull().any().any():
             logging.warning(f"NaNs found in columns {cols_to_transform} before scaling. Filling with column mean.")
             float_df = float_df.fillna(float_df.mean()) # Fill NaNs before scaling

        # Check for constant columns after potential filling
        if (float_df.nunique() <= 1).any():
             constant_cols = float_df.columns[float_df.nunique() <= 1].tolist()
             logging.warning(f"Columns {constant_cols} are constant before scaling. Scaling might produce NaNs/errors. Skipping scaling for these.")
             scalable_cols = float_df.columns.difference(constant_cols)
             if not scalable_cols.empty:
                 scaler = StandardScaler()
                 df_t[scalable_cols] = scaler.fit_transform(float_df[scalable_cols])
             # Keep constant columns as they are
        else:
             scaler = StandardScaler()
             df_t[cols_to_transform] = scaler.fit_transform(float_df)

    return df_t, scaler


def run_prophet_model(df: pd.DataFrame, target_col: str, regressor_col: str, forecast_horizon: int):
    """Runs Facebook Prophet model with an external regressor."""
    logging.info(f"\n--- Running Prophet model ---")
    logging.info(f"Target: {target_col}, Regressor: {regressor_col}, Horizon: {forecast_horizon} days")

    prophet_df = df[[target_col, regressor_col]].reset_index().rename(columns={'index':'ds', target_col:'y'})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])

    prophet_df[regressor_col] = prophet_df[regressor_col].ffill().bfill()

    if prophet_df[regressor_col].isnull().any():
        logging.warning(f"Regressor column '{regressor_col}' still contains NaNs after filling. Prophet may fail.")
        prophet_df.dropna(subset=[regressor_col], inplace=True)
        if prophet_df.empty:
            logging.error("Prophet DataFrame empty after dropping regressor NaNs. Aborting Prophet.")
            return

    model = Prophet()
    model.add_regressor(regressor_col)

    try:
        model.fit(prophet_df)
    except Exception as e:
        logging.error(f"Prophet model fitting failed: {e}")
        return

    future = model.make_future_dataframe(periods=forecast_horizon)

    logging.warning(f"Prophet forecasting requires future regressor values. Using forward-fill for '{regressor_col}'. This is an assumption.")
    future = pd.merge(future, prophet_df[['ds', regressor_col]], on='ds', how='left')
    future[regressor_col] = future[regressor_col].ffill()
    future[regressor_col] = future[regressor_col].bfill()


    if future[regressor_col].isnull().any():
         logging.error(f"Could not create future values for regressor '{regressor_col}'. Prophet prediction aborted.")
         return

    try:
        forecast = model.predict(future)
        logging.info(f"\nProphet Forecast (last {forecast_horizon} periods):")
        if not forecast.empty:
             logging.info(forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(forecast_horizon).to_string())
        else:
             logging.warning("Prophet prediction returned an empty forecast.")
    except Exception as e:
        logging.error(f"Prophet prediction failed: {e}")


def run_xgboost_model(df: pd.DataFrame, target_col: str, predictor_col: str, forecast_horizon: int, lags: int = 5, n_splits: int = 5):
    """Runs XGBoost model using lagged predictor features and TimeSeriesSplit CV."""
    logging.info(f"\n--- Running XGBoost model ---")
    logging.info(f"Target: {target_col}, Predictor: {predictor_col}, Lags: {lags}, Horizon: {forecast_horizon}, CV Splits: {n_splits}")

    xgb_df = df[[target_col, predictor_col]].copy()

    logging.info(f"Creating {lags} lagged features for '{predictor_col}'...")
    for lag in range(1, lags + 1):
        xgb_df[f'{predictor_col}_lag_{lag}'] = xgb_df[predictor_col].shift(lag)

    xgb_df = xgb_df.dropna()

    if xgb_df.empty:
        logging.error("XGBoost DataFrame is empty after creating lags and dropping NaNs. Aborting.")
        return

    feature_cols = [f'{predictor_col}_lag_{lag}' for lag in range(1, lags + 1)]
    X = xgb_df[feature_cols]
    y = xgb_df[target_col]

    if len(X) < n_splits + forecast_horizon: # Ensure enough data for all splits
         logging.error(f"Insufficient data ({len(X)} points) for XGBoost with {n_splits} TimeSeriesSplits and test_size={forecast_horizon}. Aborting.")
         return

    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=forecast_horizon)
    rmses = []
    maes = []

    logging.info("Starting Time Series Cross-Validation for XGBoost...")
    for fold, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        logging.debug(f"Fold {fold+1}/{n_splits}: Train size={len(X_train)}, Test size={len(X_test)}")

        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)

        try:
             model.fit(X_train, y_train,
                       eval_set=[(X_test, y_test)],
                       early_stopping_rounds=10,
                       verbose=False)

             preds = model.predict(X_test)
             rmse = np.sqrt(np.mean((preds - y_test)**2))
             mae = np.mean(np.abs(preds - y_test))
             rmses.append(rmse)
             maes.append(mae)
             logging.debug(f"  Fold {fold+1} Test RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        except Exception as e:
             logging.error(f"Error during XGBoost training/prediction in fold {fold+1}: {e}")
             rmses.append(np.nan)
             maes.append(np.nan)

    if rmses:
         valid_rmses = [r for r in rmses if not np.isnan(r)]
         valid_maes = [m for m in maes if not np.isnan(m)]
         if valid_rmses:
              avg_rmse = np.mean(valid_rmses)
              std_rmse = np.std(valid_rmses)
              avg_mae = np.mean(valid_maes)
              std_mae = np.std(valid_maes)
              logging.info(f"\nXGBoost Cross-Validation Results ({len(valid_rmses)} valid folds):")
              logging.info(f"  Average Test RMSE: {avg_rmse:.4f} (+/- {std_rmse:.4f})")
              logging.info(f"  Average Test MAE:  {avg_mae:.4f} (+/- {std_mae:.4f})")
         else:
              logging.error("XGBoost CV failed for all folds.")
    else:
        logging.warning("No XGBoost CV results were generated.")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Find and analyze potential FRED predictors for a stock, optimized for prediction.")
    # Required positional arguments
    parser.add_argument("ticker", help="Stock ticker symbol (e.g., AAPL)")
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD)")

    # Optional arguments
    parser.add_argument("--fetch_top_n", type=int, default=DEFAULT_FETCH_TOP_N,
                        help=f"Number of top popular FRED series to fetch (API max is {FRED_API_SEARCH_LIMIT}, default: {DEFAULT_FETCH_TOP_N})") # Updated help text
    parser.add_argument("--use_log_returns", action='store_true',
                        help="Use log returns for the stock instead of adjusted close prices (Recommended for prediction).")
    parser.add_argument("--max_granger_lag", type=int, default=DEFAULT_GRANGER_MAX_LAG_SEARCH,
                        help=f"Maximum lag to search for optimal Granger causality lag (default: {DEFAULT_GRANGER_MAX_LAG_SEARCH})")
    parser.add_argument("-o", "--output",
                        help="Optional CSV or Excel file path to save ranked results (e.g., results.csv or results.xlsx)")
    parser.add_argument("--run_models", action='store_true',
                        help="Run predictive models (Prophet, XGBoost) on the top-ranked predictor.")
    parser.add_argument("--forecast_horizon", type=int, default=30,
                        help="Forecast horizon in days for predictive models")
    parser.add_argument("--xgboost_lags", type=int, default=5,
                        help="Number of lags for XGBoost predictor features")


    args = parser.parse_args()

    FRED_API_KEY = get_fred_api_key()
    if not FRED_API_KEY:
        return

    # 1. Fetch Stock Data
    stock_data = fetch_stock_data(args.ticker, args.start_date, args.end_date, args.use_log_returns)
    if stock_data is None or stock_data.empty:
        logging.error(f"Could not fetch or process stock data for {args.ticker}. Exiting.")
        return

    # 2. Search FRED for Top N Popular Series
    # The function search_fred_popular_series now handles the API limit internally
    series_to_fetch = search_fred_popular_series(FRED_API_KEY, args.fetch_top_n)
    if not series_to_fetch:
        logging.warning(f"Could not find popular FRED series using term '{GENERIC_SEARCH_TERM}'. Exiting.")
        return
    logging.info(f"Attempting to fetch data for {len(series_to_fetch)} popular FRED series IDs.")

    # 3. Fetch FRED Series Data
    fred_data_dict = {}
    fetched_count = 0
    api_errors = 0
    for i, sid in enumerate(series_to_fetch):
        # Add progress logging
        if (i + 1) % 50 == 0:
             logging.info(f"Fetching FRED series {i+1}/{len(series_to_fetch)}...")

        try:
            series = fetch_fred_series_data(sid, FRED_API_KEY, args.start_date, args.end_date)
            if series is not None and not series.empty:
                fred_data_dict[sid] = series
                fetched_count += 1
            time.sleep(REQUEST_DELAY_SECONDS) # Rate limiting
        except Exception as e:
            logging.error(f"Unhandled error fetching series {sid}: {e}")
            api_errors += 1
            # Optional: Implement retry logic here
            time.sleep(REQUEST_DELAY_SECONDS * 2) # Longer delay after error

    logging.info(f"Successfully fetched data for {fetched_count}/{len(series_to_fetch)} FRED series. Encountered {api_errors} API/fetch errors.")

    if not fred_data_dict:
        logging.error("No FRED data could be successfully fetched. Exiting.")
        return

    # 4. Align Data
    combined_df = align_and_prepare_data(stock_data, fred_data_dict)
    if combined_df.empty or stock_data.name not in combined_df.columns:
        logging.error("Data alignment resulted in an empty DataFrame or missing target column. Exiting.")
        return

    # 5. Calculate Relationship Metrics
    logging.info("Calculating relationship metrics for each potential predictor...")
    results = {}
    valid_predictors = combined_df.columns.drop(stock_data.name)
    errors_calculating = 0

    # Get the max lag value from arguments
    max_lag_to_use = args.max_granger_lag

    for i, sid in enumerate(valid_predictors):
         # Add progress logging
        if (i + 1) % 50 == 0:
             logging.info(f"Analyzing series {i+1}/{len(valid_predictors)} ({sid})...")

        predictor_series = combined_df[sid]
        target_series = combined_df[stock_data.name]

        if predictor_series.isnull().all() or predictor_series.nunique() <= 1 :
             logging.warning(f"Skipping {sid}: All NaNs or constant value after alignment.")
             errors_calculating += 1
             continue

        # Pass the max lag value from args to the function
        metrics = calculate_relationship_metrics(target_series, predictor_series, max_lag_to_use)
        if metrics:
            results[sid] = metrics
            logging.debug(f"Calculated metrics for {sid}")
        else:
            logging.debug(f"Could not calculate metrics for {sid} (likely insufficient data or errors).")
            errors_calculating += 1

    logging.info(f"Finished calculating metrics. Successfully processed: {len(results)}. Failed/Skipped: {errors_calculating}")

    if not results:
        logging.error("No relationship metrics could be computed for any series. Exiting.")
        return

    # 6. Rank Signals
    results_df = pd.DataFrame.from_dict(results, orient='index')
    ranked_df = rank_signals(results_df)

    # Display Top N
    top_n = 20
    logging.info(f"\n--- Top {top_n} Potential Predictor Signals (Ranked for Prediction) ---")
    # Updated display columns
    display_cols = [
        'rank', 'score', 'optimal_lag_aic', 'granger_p_value_optimal_lag',
        'is_cointegrated', 'coint_p_value', 'mutual_info',
        'pearson_lag_0', 'spearman_lag_0', 'pearson_lag_1',
        'adf_p_predictor_level', 'adf_p_predictor_diff' # Show stationarity info
    ]
    display_cols = [col for col in display_cols if col in ranked_df.columns] # Filter existing columns
    # Use pandas option context for better display
    with pd.option_context('display.max_rows', top_n,
                           'display.max_columns', None,
                           'display.width', 1000,
                           'display.float_format', '{:.4f}'.format):
        logging.info("\n" + ranked_df[display_cols].head(top_n).to_string())


    # 7. Save Results (Optional) - Added Excel support
    if args.output:
        try:
            if args.output.endswith('.csv'):
                ranked_df.to_csv(args.output)
                logging.info(f"Ranked results saved to CSV: {args.output}")
            elif args.output.endswith('.xlsx'):
                 # Ensure target column name is filesystem-safe for sheet name
                 safe_target_name = "".join(c if c.isalnum() else "_" for c in stock_data.name)
                 ranked_df.to_excel(args.output, sheet_name=f'Ranked_Signals_{safe_target_name}')
                 logging.info(f"Ranked results saved to Excel: {args.output}")
            else:
                 logging.warning("Output file extension not recognized (use .csv or .xlsx). Saving as CSV.")
                 ranked_df.to_csv(args.output + ".csv")
                 logging.info(f"Ranked results saved to CSV: {args.output}.csv")

        except Exception as e:
            logging.error(f"Failed to save results to {args.output}: {e}")

    # 8. Run Predictive Models (Optional)
    if args.run_models and not ranked_df.empty:
        top_predictor_sid = ranked_df.index[0]
        logging.info(f"\n--- Running Predictive Models on Top Predictor: {top_predictor_sid} ---")

        # Determine if differencing was used for the top predictor during analysis
        top_pred_results = results_df.loc[top_predictor_sid]
        target_adf_p = top_pred_results.get('adf_p_target_level', 1.0) # Default to non-stationary if missing
        pred_adf_p = top_pred_results.get('adf_p_predictor_level', 1.0)

        needs_diff_model = (target_adf_p >= ADF_P_VALUE_THRESHOLD) or \
                           (pred_adf_p >= ADF_P_VALUE_THRESHOLD)
        needs_scale_model = True # Scaling generally good for XGBoost

        logging.info(f"Modeling transformations: Differencing={needs_diff_model}, Scaling={needs_scale_model}")

        model_df, _ = apply_transformations(combined_df,
                                          cols_to_transform=[stock_data.name, top_predictor_sid],
                                          difference=needs_diff_model,
                                          scale=needs_scale_model)

        if model_df.empty:
             logging.error("DataFrame for modeling is empty after transformations. Skipping modeling.")
        else:
            # Run Prophet
            prophet_input_df = combined_df[[stock_data.name, top_predictor_sid]].copy()
            run_prophet_model(prophet_input_df,
                              target_col=stock_data.name,
                              regressor_col=top_predictor_sid,
                              forecast_horizon=args.forecast_horizon)

            # Run XGBoost
            run_xgboost_model(model_df,
                            target_col=stock_data.name,
                            predictor_col=top_predictor_sid,
                            forecast_horizon=args.forecast_horizon,
                            lags=args.xgboost_lags)

    logging.info("\n--- Script execution finished ---")


if __name__ == "__main__":
    main()
