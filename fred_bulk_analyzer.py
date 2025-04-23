# --- fred_bulk_analyzer.py ---

import argparse
import logging
import os
import sys # To increase recursion depth limit if needed
import time
import warnings
from typing import List, Dict, Optional, Tuple, Any, Set

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests, coint
from statsmodels.tools.eval_measures import rmse as sm_rmse

# --- Configuration ---
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        # Optional: Add a FileHandler to log to a file as well
                        # logging.FileHandler("bulk_analyzer.log")
                    ])

# --- Constants ---
FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"
REQUEST_DELAY_SECONDS = 0.6 # INCREASED DELAY to be more respectful of API limits during long run
API_RETRY_DELAY = 5 # Seconds to wait after a rate limit error
API_MAX_RETRIES = 5 # Max retries for API calls
FRED_API_OFFSET_LIMIT = 1000 # Max results per call for series/list type endpoints
MIN_OBS_FOR_VAR = 50
DEFAULT_GRANGER_MAX_LAG_SEARCH = 10
ADF_P_VALUE_THRESHOLD = 0.05
COINT_P_VALUE_THRESHOLD = 0.05
DEFAULT_TOP_N_SAVE = 1000
BATCH_SAVE_SIZE = 500 # How often to save results to the CSV
TARGET_FREQUENCY = 'D' # Filter for Daily frequency series

# --- Helper Functions ---

def get_fred_api_key() -> Optional[str]:
    """Retrieves the FRED API key from environment variables."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logging.error("FRED_API_KEY environment variable not set.")
        return None
    return api_key

def make_api_request(endpoint: str, params: Dict[str, Any], api_key: str) -> Optional[Dict]:
    """Makes a request to the FRED API with rate limit handling and retries."""
    if 'api_key' not in params:
        params['api_key'] = api_key
    if 'file_type' not in params:
        params['file_type'] = 'json'

    url = f"{FRED_API_BASE_URL}/{endpoint}"
    retries = 0
    while retries < API_MAX_RETRIES:
        try:
            response = requests.get(url, params=params, timeout=30) # Added timeout
            if response.status_code == 429: # Rate limit exceeded
                logging.warning(f"Rate limit exceeded (429). Waiting {API_RETRY_DELAY} seconds...")
                time.sleep(API_RETRY_DELAY)
                retries += 1
                continue # Retry the request
            response.raise_for_status() # Raise HTTPError for other bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.Timeout:
             logging.warning(f"Request timed out for {endpoint} with params {params}. Retrying ({retries+1}/{API_MAX_RETRIES})...")
             retries += 1
             time.sleep(API_RETRY_DELAY)
        except requests.exceptions.RequestException as e:
            logging.error(f"HTTP Error for {endpoint} with params {params}: {e}")
            retries += 1
            if retries >= API_MAX_RETRIES:
                 logging.error("Max retries reached. Skipping this request.")
                 return None
            logging.warning(f"Retrying ({retries}/{API_MAX_RETRIES}) after error...")
            time.sleep(API_RETRY_DELAY) # Wait before retrying after error
        except Exception as e:
            logging.error(f"Unexpected error during API request for {endpoint}: {e}")
            return None # Non-recoverable error for this request
    return None # Failed after retries

# --- Functions to Get All Series IDs (via Category Traversal) ---

def get_child_categories(api_key: str, category_id: int = 0) -> List[int]:
    """Gets child category IDs for a given category ID."""
    endpoint = "category/children"
    params = {"category_id": category_id}
    data = make_api_request(endpoint, params, api_key)
    if data and "categories" in data:
        return [cat.get("id") for cat in data["categories"] if cat.get("id") is not None]
    return []

# Modified: Filters for TARGET_FREQUENCY
def get_category_series(api_key: str, category_id: int, limit: int = FRED_API_OFFSET_LIMIT) -> List[str]:
    """
    Gets series IDs within a specific category, filtering by frequency
    and handling pagination.
    """
    all_series_ids_in_cat = []
    offset = 0
    logging.debug(f"Fetching series for category {category_id} (frequency={TARGET_FREQUENCY})...")
    while True:
        endpoint = "category/series"
        params = {"category_id": category_id, "limit": limit, "offset": offset}
        data = make_api_request(endpoint, params, api_key)

        if data and "seriess" in data and data["seriess"]:
            series_list = data["seriess"]
            ids_in_batch = []
            # --- Filter for daily frequency ---
            for s_info in series_list:
                if s_info.get("id") and s_info.get("frequency_short") == TARGET_FREQUENCY:
                    ids_in_batch.append(s_info["id"])
            # ---------------------------------
            all_series_ids_in_cat.extend(ids_in_batch)
            logging.debug(f"  Category {category_id}, Offset {offset}: Found {len(ids_in_batch)} daily series in this batch of {len(series_list)}.")

            # Check if we got fewer *total* results than the limit, meaning we reached the end for this category
            if len(series_list) < limit:
                break
            # Otherwise, prepare for the next page
            offset += limit
            time.sleep(REQUEST_DELAY_SECONDS) # Delay between pages
        else:
            # No more series or an error occurred
            if data is None: # Logged error in make_api_request
                 logging.warning(f"Failed to fetch series for category {category_id} at offset {offset}.")
            else:
                 logging.debug(f"No more series found for category {category_id} at offset {offset}.")
            break # Exit loop

    return all_series_ids_in_cat


def get_all_series_ids_recursive(api_key: str, category_id: int = 0, existing_ids: Optional[Set[str]] = None) -> Set[str]:
    """Recursively traverses categories to find all series IDs matching TARGET_FREQUENCY."""
    if existing_ids is None:
        existing_ids = set()

    logging.info(f"Processing Category ID: {category_id} for {TARGET_FREQUENCY} series...")

    # Get series (already filtered by frequency) in the current category
    series_in_cat = get_category_series(api_key, category_id)
    new_series_count = 0
    if series_in_cat: # Only update if list is not empty
        new_series = set(series_in_cat) - existing_ids
        new_series_count = len(new_series)
        if new_series_count > 0:
            logging.info(f"  Found {new_series_count} new {TARGET_FREQUENCY}-frequency series in category {category_id}.")
            existing_ids.update(new_series)
        else:
            logging.info(f"  No new {TARGET_FREQUENCY}-frequency series found in category {category_id}.")
    else:
         logging.info(f"  No {TARGET_FREQUENCY}-frequency series returned for category {category_id}.")


    # Get child categories and recurse
    child_ids = get_child_categories(api_key, category_id)
    time.sleep(REQUEST_DELAY_SECONDS) # Delay before processing children

    for child_id in child_ids:
        get_all_series_ids_recursive(api_key, child_id, existing_ids)

    return existing_ids


# --- Functions for Data Fetching and Analysis (Adapted from previous scripts) ---

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
            if (data <= 0).any():
                logging.warning(f"Ticker {ticker} has non-positive prices. Cannot calculate log returns accurately. Using Adj Close.")
            else:
                data = np.log(data / data.shift(1))
                series_name = f"{ticker}_LogReturn"
                data = data.iloc[1:]
        return data.rename(series_name)
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return None

def fetch_fred_series_data_robust(series_id: str, api_key: str, start_date: str, end_date: str) -> Optional[pd.Series]:
    """Fetches observations for a single FRED series using robust API call."""
    logging.debug(f"Fetching data for FRED series: {series_id}...")
    params = {
        "series_id": series_id,
        "observation_start": start_date,
        "observation_end": end_date,
        "aggregation_method": "avg", # Keep avg, though some series might benefit from 'eop'
    }
    data = make_api_request("series/observations", params, api_key)

    if data and "observations" in data:
        obs = data["observations"]
        if not obs:
            logging.debug(f"No observations found for {series_id} in the specified date range.")
            return None
        try:
            df = pd.DataFrame(obs)[["date", "value"]]
            df["date"] = pd.to_datetime(df["date"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna().set_index("date")
            return df["value"].rename(series_id) if not df.empty else None
        except Exception as e:
             logging.error(f"Error processing data for {series_id} after fetch: {e}")
             return None
    else:
        # Error logged in make_api_request if data is None
        if data is not None: # Log only if request succeeded but data was bad
             logging.warning(f"Failed to fetch or parse observations for {series_id}.")
        return None

def test_stationarity(series: pd.Series, name: str = "Series", max_lag_search: int = DEFAULT_GRANGER_MAX_LAG_SEARCH) -> Tuple[float, bool]:
    """Performs the Augmented Dickey-Fuller test for stationarity."""
    try:
        # Convert to float and drop NaNs *before* checking length/constant
        series_float = series.astype(float).dropna()
        if len(series_float) < max(15, max_lag_search + 5):
             # logging.debug(f"ADF test skipped for {name}: Insufficient data ({len(series_float)} points)")
             return np.nan, False
        if np.ptp(series_float.values) == 0:
             # logging.debug(f"ADF test skipped for {name}: Series is constant")
             return 1.0, False # Treat constant as non-stationary
        result = adfuller(series_float, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < ADF_P_VALUE_THRESHOLD
        return p_value, is_stationary
    except Exception as e: # Catch LinAlgError, ValueError etc.
        logging.warning(f"ADF test failed for {name}: {e}") # Log actual error now
        return np.nan, False

def find_optimal_lag_var(data: pd.DataFrame, max_lag: int) -> Tuple[Optional[int], Optional[float]]:
    """Finds the optimal lag for a VAR model using AIC."""
    best_aic = np.inf
    optimal_lag = None
    # Ensure data is float and drop NaNs introduced by differencing etc.
    data = data.astype(float).dropna()
    if len(data) < MIN_OBS_FOR_VAR:
        # logging.debug(f"VAR lag search skipped: Insufficient data ({len(data)} < {MIN_OBS_FOR_VAR})")
        return None, None
    # Adjust max_lag based on available data
    max_lag = min(max_lag, (len(data) - 1) // data.shape[1] - 1 )
    if max_lag < 1:
        # logging.debug(f"VAR lag search skipped: Not enough data for any lags.")
        return None, None

    for lag in range(1, max_lag + 1):
        try:
            # Check required observations: T >= k*p + 1 (k=num_vars, p=lag)
            # Add buffer: T >= k*p + 5
            if len(data) < data.shape[1] * lag + 5:
                # logging.debug(f"Skipping VAR lag {lag}: Insufficient observations.")
                continue
            model = VAR(data)
            results = model.fit(maxlags=lag, ic=None) # Fit for specific lag
            current_aic = results.aic
            if pd.notna(current_aic) and np.isfinite(current_aic) and current_aic < best_aic:
                best_aic = current_aic
                optimal_lag = lag
        except Exception as e:
            # logging.debug(f"VAR lag search failed for lag {lag}: {e}") # Reduce noise
            continue # Ignore errors during lag search
    return optimal_lag, best_aic

# --- Modified calculate_relationship_metrics with more robust error handling ---
def calculate_relationship_metrics(target_series: pd.Series, predictor_series: pd.Series, max_lag_search: int) -> Optional[Dict[str, Any]]:
    """
    Calculates relationship metrics, handling stationarity and optimal lags more robustly.
    Returns None only if initial checks fail, otherwise returns dict with NaNs for failed steps.
    """
    metrics = {}
    series_name = predictor_series.name
    target_name = target_series.name

    # 1. Initial Data Prep & Checks
    try:
        combined_levels = pd.concat([target_series, predictor_series], axis=1).dropna()
        if len(combined_levels) < MIN_OBS_FOR_VAR:
            logging.debug(f"Skipping {series_name}: Insufficient overlap ({len(combined_levels)} < {MIN_OBS_FOR_VAR}).")
            return None # Exit early if not enough overlap

        t_level = combined_levels.iloc[:, 0]
        p_level = combined_levels.iloc[:, 1]

        if np.ptp(t_level.values) == 0 or np.ptp(p_level.values) == 0:
            logging.debug(f"Skipping {series_name}: Target or predictor series is constant at level.")
            return None # Exit early if constant
    except Exception as e:
        logging.warning(f"Error during initial data prep for {series_name}: {e}")
        return None # Exit if basic prep fails

    # 2. Stationarity Tests (Levels)
    metrics['adf_p_target_level'], target_is_stationary = test_stationarity(t_level, target_name, max_lag_search)
    metrics['adf_p_predictor_level'], p_is_stationary = test_stationarity(p_level, series_name, max_lag_search)
    if pd.isna(metrics['adf_p_target_level']): target_is_stationary = False # Treat NaN p-value as non-stationary
    if pd.isna(metrics['adf_p_predictor_level']): p_is_stationary = False

    # 3. Cointegration Test
    metrics['coint_p_value'], metrics['is_cointegrated'] = np.nan, False
    if not target_is_stationary and not p_is_stationary and len(t_level) > MIN_OBS_FOR_VAR:
        try:
            # Ensure float type for cointegration test
            t_level_float = t_level.astype(float)
            p_level_float = p_level.astype(float)
            score, p_value, _ = coint(t_level_float, p_level_float, trend='c')
            metrics['coint_p_value'] = p_value
            metrics['is_cointegrated'] = p_value < COINT_P_VALUE_THRESHOLD
        except Exception as e:
            logging.warning(f"Cointegration test failed for {series_name}: {e}")
            # Keep metrics as NaN, continue analysis

    # 4. Prepare Differenced Data & Check Stationarity (if needed)
    metrics['adf_p_target_diff'], metrics['adf_p_predictor_diff'] = np.nan, np.nan
    analysis_df = None
    differenced = False
    if not target_is_stationary or not p_is_stationary:
        differenced = True
        try:
            analysis_df = combined_levels.diff().dropna()
            # Test stationarity *after* differencing
            if len(analysis_df) >= MIN_OBS_FOR_VAR:
                 metrics['adf_p_target_diff'], _ = test_stationarity(analysis_df.iloc[:, 0], f"{target_name}_diff", max_lag_search)
                 metrics['adf_p_predictor_diff'], _ = test_stationarity(analysis_df.iloc[:, 1], f"{series_name}_diff", max_lag_search)
            else:
                 logging.debug(f"Skipping post-difference ADF check for {series_name}: insufficient data after diff.")
        except Exception as e:
             logging.warning(f"Error during differencing or post-diff ADF for {series_name}: {e}")
             analysis_df = pd.DataFrame() # Ensure it's empty if error occurs
    else:
        analysis_df = combined_levels # Use levels if both stationary

    # Check length and constants *after* potential differencing
    if analysis_df is None or len(analysis_df) < MIN_OBS_FOR_VAR:
        logging.debug(f"Skipping further analysis for {series_name}: Insufficient data after differencing ({len(analysis_df if analysis_df is not None else 0)}).")
        return metrics # Return partial metrics (ADF levels, maybe Coint)

    try:
        t_analysis = analysis_df.iloc[:, 0]
        p_analysis = analysis_df.iloc[:, 1]
        if np.ptp(t_analysis.values) == 0 or np.ptp(p_analysis.values) == 0:
            logging.debug(f"Skipping further analysis for {series_name}: Series became constant after differencing.")
            return metrics # Return partial metrics
    except Exception as e:
         logging.warning(f"Error checking constants after differencing for {series_name}: {e}")
         return metrics # Return partial metrics


    # 5. Mutual Information
    metrics["mutual_info"] = np.nan
    try:
        p_reshaped = p_analysis.values.reshape(-1, 1)
        t_ravelled = t_analysis.values.ravel()
        metrics["mutual_info"] = mutual_info_regression(p_reshaped, t_ravelled, discrete_features=False)[0]
    except Exception as e:
        logging.warning(f"Mutual Information calculation failed for {series_name}: {e}")

    # 6. Optimal Lag & Granger Causality
    metrics["optimal_lag_aic"], metrics["granger_p_value_optimal_lag"] = np.nan, np.nan
    optimal_lag = None # Initialize optimal_lag
    try:
        var_data = analysis_df[[target_name, series_name]] # No need to astype float here, find_optimal_lag handles it
        optimal_lag, _ = find_optimal_lag_var(var_data, max_lag_search)
        if optimal_lag is not None and optimal_lag > 0:
            metrics["optimal_lag_aic"] = optimal_lag
            # Ensure data is float for grangercausalitytests
            var_data_float = var_data.astype(float).dropna()
            # Check length again after potential NaNs from float conversion
            if len(var_data_float) >= optimal_lag + 5:
                 gc_result = grangercausalitytests(var_data_float, maxlag=[optimal_lag], verbose=False)
                 if optimal_lag in gc_result and gc_result[optimal_lag][0].get('ssr_ftest'):
                      ssr_ftest_result = gc_result[optimal_lag][0]['ssr_ftest']
                      if isinstance(ssr_ftest_result, (list, tuple)) and len(ssr_ftest_result) == 4:
                           metrics["granger_p_value_optimal_lag"] = ssr_ftest_result[1] # p-value
                      else:
                           logging.warning(f"Granger ssr_ftest format unexpected for {series_name}, lag {optimal_lag}")
                 else:
                      logging.warning(f"Granger result structure unexpected for {series_name}, lag {optimal_lag}")
            else:
                 logging.debug(f"Skipping Granger test for {series_name} at lag {optimal_lag}: insufficient data after float conversion/dropna.")
        elif optimal_lag is None:
             logging.debug(f"Skipping Granger test for {series_name}: could not determine optimal lag.")

    except Exception as e:
        logging.warning(f"VAR/Granger analysis failed for {series_name}: {e}")


    # 7. Lagged Correlations
    for lag in range(2): # Only compute lag 0 & 1 for speed
        metrics[f"pearson_lag_{lag}"], metrics[f"spearman_lag_{lag}"] = np.nan, np.nan # Initialize
        try:
            p_lagged = p_analysis.shift(lag)
            corr_df = pd.concat([t_analysis, p_lagged], axis=1).dropna()
            if len(corr_df) > 2:
                a, b = corr_df.iloc[:, 0].values, corr_df.iloc[:, 1].values
                if np.ptp(a) > 0 and np.ptp(b) > 0: # Check constants again
                    # Add inner try-except for each correlation type
                    try: metrics[f"pearson_lag_{lag}"] = pearsonr(a, b)[0]
                    except ValueError: pass # Keep NaN on ValueError
                    try: metrics[f"spearman_lag_{lag}"] = spearmanr(a, b)[0]
                    except ValueError: pass # Keep NaN on ValueError
        except Exception as e:
            logging.warning(f"Correlation calculation failed for {series_name}, lag {lag}: {e}")
            # Keep metrics as NaN

    return metrics # Return the dictionary, possibly with NaNs


def rank_signals(results_df: pd.DataFrame) -> pd.DataFrame:
    """Ranks signals based on metrics, tuned for prediction."""
    if results_df.empty: return results_df
    weights = { # Simplified weights for bulk analysis
        'granger_p_value_optimal_lag': -0.40,
        'coint_p_value':               -0.25,
        'mutual_info':                  0.20,
        'pearson_lag_0':                0.10,
        'spearman_lag_0':               0.10,
    }
    ranked = results_df.copy()
    ranked['score'] = 0.0
    for metric, weight in weights.items():
        if metric in ranked.columns:
            col_data = ranked[metric]
            if pd.api.types.is_numeric_dtype(col_data):
                score_contribution = col_data.fillna(1.0 if 'p_value' in metric else 0.0)
                if 'p_value' in metric: score_contribution *= weight
                else: score_contribution = score_contribution.abs() * weight
                ranked['score'] += score_contribution.fillna(0.0)
    ranked = ranked.sort_values(by='score', ascending=False)
    ranked['rank'] = range(1, len(ranked) + 1)
    return ranked

# --- Main Execution ---
def main():
    # Set higher recursion depth limit for category traversal if needed
    # sys.setrecursionlimit(2000) # Use with caution

    parser = argparse.ArgumentParser(description="Analyze ALL FRED series against a target stock with checkpointing.")
    parser.add_argument("ticker", help="Target stock ticker symbol (e.g., AAPL)")
    parser.add_argument("start_date", help="Start date for analysis (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date for analysis (YYYY-MM-DD)")
    parser.add_argument("-o", "--output_csv", required=True, help="CSV file path to save/resume ranked results")
    parser.add_argument("--top_n_save", type=int, default=DEFAULT_TOP_N_SAVE, help=f"Number of top ranked series to save in the final ranked file (default: {DEFAULT_TOP_N_SAVE})")
    parser.add_argument("--use_log_returns", action='store_true', help="Use log returns for the stock (recommended)")
    parser.add_argument("--max_series_process", type=int, default=None, help="Optional: Stop after processing this many *new* series (for testing)")
    parser.add_argument("--max_granger_lag", type=int, default=DEFAULT_GRANGER_MAX_LAG_SEARCH, help=f"Max lag for Granger/VAR tests (default: {DEFAULT_GRANGER_MAX_LAG_SEARCH})")

    args = parser.parse_args()

    api_key = get_fred_api_key()
    if not api_key: return

    # --- Checkpointing Setup ---
    processed_ids = set()
    all_results = {} # Store results from previous runs if file exists
    output_file = args.output_csv
    file_exists = os.path.exists(output_file)

    if file_exists:
        logging.info(f"Output file '{output_file}' exists. Loading previous results...")
        try:
            # Read existing data, assuming first column is the series ID index
            existing_df = pd.read_csv(output_file, index_col=0)
            # Convert DataFrame back to dictionary format expected by the script
            all_results = existing_df.to_dict(orient='index')
            processed_ids = set(existing_df.index)
            logging.info(f"Loaded {len(processed_ids)} previously processed series IDs.")
        except Exception as e:
            logging.error(f"Error loading existing results from {output_file}: {e}. Starting fresh analysis.")
            # Reset in case of error reading the file
            processed_ids = set()
            all_results = {}
            file_exists = False # Treat as if file doesn't exist for writing header later
    else:
        logging.info(f"Output file '{output_file}' not found. Starting fresh analysis.")

    # --- Fetch Target Data ---
    target_data = fetch_stock_data(args.ticker, args.start_date, args.end_date, args.use_log_returns)
    if target_data is None or target_data.empty:
        logging.error(f"Could not fetch target data for {args.ticker}. Exiting.")
        return
    target_col_name = target_data.name
    logging.info(f"Target data fetched: {len(target_data)} points for {target_col_name}")

    # --- Get All FRED IDs ---
    logging.info(f"Attempting to fetch all FRED series IDs with frequency '{TARGET_FREQUENCY}' via category traversal...") # Updated log
    logging.warning("This process can take a very long time and may hit API limits.")
    try:
        all_fred_series_ids = get_all_series_ids_recursive(api_key)
        logging.info(f"Finished category traversal. Found {len(all_fred_series_ids)} unique {TARGET_FREQUENCY}-frequency series IDs.") # Updated log
    except RecursionError:
         logging.error("Hit Python's recursion depth limit during category traversal.")
         return
    except Exception as e:
         logging.error(f"An error occurred during category traversal: {e}")
         return

    if not all_fred_series_ids:
        logging.error(f"No {TARGET_FREQUENCY}-frequency FRED series IDs were collected. Exiting.") # Updated log
        return

    # --- Filter out already processed IDs ---
    fred_series_list_full = list(all_fred_series_ids)
    series_to_process_list = [sid for sid in fred_series_list_full if sid not in processed_ids]
    logging.info(f"Total unique {TARGET_FREQUENCY}-frequency IDs found: {len(all_fred_series_ids)}. Previously processed: {len(processed_ids)}. Remaining to process: {len(series_to_process_list)}") # Updated log

    total_series_to_process_limit = len(series_to_process_list)
    if args.max_series_process is not None:
         total_series_to_process_limit = min(total_series_to_process_limit, args.max_series_process)
         series_to_process_list = series_to_process_list[:total_series_to_process_limit]
         logging.warning(f"Limiting processing to {total_series_to_process_limit} new series based on --max_series_process.")

    # --- Process Series Iteratively with Checkpointing ---
    logging.info(f"Starting analysis of {len(series_to_process_list)} new {TARGET_FREQUENCY}-frequency FRED series against {target_col_name}...") # Updated log
    new_results_batch = {} # Store results for the current batch
    processed_in_session = 0
    fetch_errors = 0
    analysis_errors_skipped = 0 # Count series skipped due to analysis errors
    start_time_analysis = time.time()

    for i, series_id in enumerate(series_to_process_list):
        current_series_num = i + 1
        # Log progress periodically
        if current_series_num % 100 == 0:
            elapsed_time = time.time() - start_time_analysis
            rate = current_series_num / elapsed_time if elapsed_time > 0 else 0
            logging.info(f"Progress: {current_series_num}/{len(series_to_process_list)} ({rate:.2f} series/sec). Successful this session: {processed_in_session}. Fetch Errors: {fetch_errors}. Analysis Skips: {analysis_errors_skipped}.")

        # Fetch predictor data
        predictor_data = fetch_fred_series_data_robust(series_id, api_key, args.start_date, args.end_date)
        time.sleep(REQUEST_DELAY_SECONDS) # Crucial delay

        if predictor_data is None or predictor_data.empty:
            fetch_errors += 1
            continue # Skip if data fetch failed

        # Align (only target and current predictor)
        aligned_df = pd.concat([target_data, predictor_data], axis=1).dropna()

        # Calculate metrics (more robust function)
        metrics = calculate_relationship_metrics(aligned_df[target_col_name], aligned_df[series_id], args.max_granger_lag)

        if metrics is not None:
            # Check if at least some key metrics were calculable (e.g., correlations or MI)
            # This avoids saving entries that failed very early (e.g., due to insufficient overlap)
            key_metrics_present = any(k in metrics and pd.notna(metrics[k]) for k in ['mutual_info', 'pearson_lag_0', 'spearman_lag_0', 'granger_p_value_optimal_lag'])
            if key_metrics_present:
                new_results_batch[series_id] = metrics
                processed_in_session += 1 # Increment count of successfully processed series
            else:
                 analysis_errors_skipped += 1 # Count as skipped if no useful metrics generated
        else:
             analysis_errors_skipped += 1 # Count as skipped if initial checks failed (returned None)


        # --- Save batch results periodically ---
        # Trigger save based on processed_in_session counter
        if processed_in_session > 0 and processed_in_session % BATCH_SAVE_SIZE == 0:
            logging.info(f"Successfully processed {processed_in_session} series in session. Saving batch results to {output_file}...")
            try:
                batch_df = pd.DataFrame.from_dict(new_results_batch, orient='index')
                # Append to CSV, write header only if file didn't exist initially
                batch_df.to_csv(output_file, mode='a', header=not file_exists)
                file_exists = True # Ensure header is not written again
                logging.info(f"Successfully appended {len(new_results_batch)} results.")
                # Add newly saved IDs to processed set and clear batch
                processed_ids.update(new_results_batch.keys())
                new_results_batch = {} # Reset batch
                # Reset session counter after successful save to trigger next save correctly
                processed_in_session = 0
            except Exception as e:
                logging.error(f"Failed to save batch results to {output_file}: {e}")
                # Keep results in new_results_batch to try saving again next time

    # --- Final Save and Ranking ---
    logging.info("Analysis loop finished.")

    # Save any remaining results from the last batch
    if new_results_batch:
        logging.info(f"Saving remaining {len(new_results_batch)} results...")
        try:
            batch_df = pd.DataFrame.from_dict(new_results_batch, orient='index')
            batch_df.to_csv(output_file, mode='a', header=not file_exists)
            processed_ids.update(new_results_batch.keys()) # Update processed IDs
            logging.info("Successfully appended final batch results.")
        except Exception as e:
            logging.error(f"Failed to save final batch results to {output_file}: {e}")

    # Now, read the complete results file for final ranking
    logging.info("Reading complete results file for final ranking...")
    try:
         final_results_df = pd.read_csv(output_file, index_col=0)
         # Handle potential duplicate indices if script was restarted improperly
         final_results_df = final_results_df[~final_results_df.index.duplicated(keep='last')]
    except Exception as e:
         logging.error(f"Could not read final results file {output_file} for ranking: {e}")
         return # Cannot rank if file cannot be read

    logging.info(f"Ranking {len(final_results_df)} total processed series...")
    ranked_df = rank_signals(final_results_df)

    # Select top N to save (overwrite the file with the final ranked version)
    top_n_df = ranked_df.head(args.top_n_save)

    logging.info(f"Top 5 Ranked Series:\n{top_n_df[['rank', 'score']].head().to_string()}")

    # Save the final ranked top N list (overwrite previous file)
    try:
        top_n_df.to_csv(output_file) # Overwrite the file now
        logging.info(f"Successfully saved final top {len(top_n_df)} ranked series to {output_file}")
    except Exception as e:
        logging.error(f"Failed to save final ranked results to {output_file}: {e}")

    end_time_total = time.time()
    # Calculate total processed this run more accurately
    total_processed_this_run = len(processed_ids) - (len(all_results) if file_exists else 0)

    logging.info(f"--- Total Script Execution Time (this run): {(end_time_total - start_time_analysis):.2f} seconds ---")
    logging.info(f"--- Summary: Processed {total_processed_this_run} new series this run. Fetch Errors: {fetch_errors}. Analysis Skips: {analysis_errors_skipped}. ---")


if __name__ == "__main__":
    main()
