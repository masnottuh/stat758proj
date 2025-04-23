# --- run_models.py ---

import argparse
import logging
import os
import warnings
import time
import itertools # For SARIMAX grid search
from typing import List, Dict, Optional, Tuple, Any

# --- Imports ---
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import statsmodels.api as sm
import yfinance as yf
import xgboost as xgb
from scipy.stats import normaltest, uniform, randint # For RandomizedSearchCV distributions
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, make_scorer
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse as sm_rmse
# For hyperparameter tuning
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
# Optional: For GARCH features
try:
    from arch import arch_model
    arch_available = True
except ImportError:
    arch_available = False
    # No warning here, will be checked later if --add_garch is used


# --- Configuration ---
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants ---
FRED_API_BASE_URL_OBS = "https://api.stlouisfed.org/fred/series/observations"
REQUEST_DELAY_SECONDS = 0.3
ADF_P_VALUE_THRESHOLD = 0.05
DEFAULT_XGB_LAGS = 5
DEFAULT_GRANGER_MAX_LAG_SEARCH = 10
# SARIMAX Grid Search Ranges (Expanded Slightly)
SARIMAX_P_RANGE = range(0, 4) # Increased range
SARIMAX_D_RANGE = range(0, 2) # Keep d range small (0 or 1 usually sufficient)
SARIMAX_Q_RANGE = range(0, 4) # Increased range
# Set seasonal order to non-seasonal for simplicity in auto-tuning
SARIMAX_SEASONAL_ORDER = (0, 0, 0, 0)
# XGBoost Randomized Search Params
XGB_N_ITER_SEARCH = 50 # Increased iterations
XGB_CV_SPLITS = 5 # Number of splits for TimeSeriesSplit in hyperparameter tuning
# Rolling window sizes for volatility features
VOLATILITY_WINDOWS = [5, 21, 63] # Added longer window
# Quantiles for XGBoost prediction interval
XGB_QUANTILE_LOW = 0.30 # 10th percentile
XGB_QUANTILE_HIGH = 0.70 # 90th percentile
XGB_QUANTILE_MEDIAN = 0.50 # Median

# --- Helper Functions ---

def get_fred_api_key() -> Optional[str]:
    """Retrieves the FRED API key from environment variables."""
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logging.error("FRED_API_KEY environment variable not set.")
        return None
    return api_key

def test_stationarity(series: pd.Series, name: str = "Series") -> Tuple[float, bool]:
    """Performs the Augmented Dickey-Fuller test."""
    try:
        series_float = series.astype(float).dropna()
        if len(series_float) < 15: return np.nan, False
        if np.ptp(series_float.values) == 0: return 1.0, False # Constant series
        result = adfuller(series_float, autolag='AIC')
        p_value = result[1]
        is_stationary = p_value < ADF_P_VALUE_THRESHOLD
        logging.debug(f"ADF test for {name}: p-value={p_value:.4f}, Stationary={is_stationary}")
        return p_value, is_stationary
    except Exception as e:
        logging.error(f"Error during ADF test for {name}: {e}")
        return np.nan, False

# Custom scorer for RandomizedSearchCV - using MAE for quantile regression might be more robust
def mae_scorer(y_true, y_pred):
     y_true = np.asarray(y_true)
     y_pred = np.asarray(y_pred)
     mask = np.isfinite(y_true) & np.isfinite(y_pred)
     if not np.all(mask):
         y_true = y_true[mask]
         y_pred = y_pred[mask]
         if len(y_true) == 0: return -np.inf
     if len(y_true) == 0: return -np.inf
     return -mean_absolute_error(y_true, y_pred) # Negative MAE

def calculate_metrics(actual: np.ndarray, predicted: np.ndarray, model_name: str = "") -> Dict[str, float]:
    """Calculates standard regression, directional accuracy, and variance ratio metrics."""
    metrics = {}
    try:
        actual = np.asarray(actual).flatten()
        predicted = np.asarray(predicted).flatten()
        # Ensure finite values before calculation
        finite_mask = np.isfinite(actual) & np.isfinite(predicted)
        if not np.all(finite_mask):
            logging.warning(f"Non-finite values found in actual/predicted for {model_name}. Calculating metrics on finite subset.")
            actual = actual[finite_mask]
            predicted = predicted[finite_mask]

        if actual.shape != predicted.shape:
             logging.warning(f"Shape mismatch for metrics calculation ({model_name}): Actual {actual.shape}, Predicted {predicted.shape}. Aligning...")
             min_len = min(len(actual), len(predicted))
             actual = actual[:min_len]
             predicted = predicted[:min_len]

        if len(actual) == 0: raise ValueError("Actual/Predicted arrays are empty after processing.")

        metrics['RMSE'] = sm_rmse(actual, predicted)
        metrics['MAE'] = mean_absolute_error(actual, predicted)
        mask = actual != 0
        if np.any(mask):
             metrics['MAPE'] = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
        else:
             metrics['MAPE'] = np.nan

        actual_diff = np.diff(actual)
        predicted_diff = np.diff(predicted)
        min_len_diff = min(len(actual_diff), len(predicted_diff))
        if min_len_diff > 0:
            actual_diff = actual_diff[:min_len_diff]
            predicted_diff = predicted_diff[:min_len_diff]
            correct_direction = (np.sign(actual_diff) == np.sign(predicted_diff))
            zero_mask = (actual_diff == 0)
            correct_direction[zero_mask] = (predicted_diff[zero_mask] == 0)
            metrics['Directional_Accuracy'] = np.mean(correct_direction) * 100
        else:
            metrics['Directional_Accuracy'] = np.nan

        var_actual = np.var(actual)
        var_predicted = np.var(predicted)
        if var_actual > 1e-9:
             metrics['Variance_Ratio'] = var_predicted / var_actual
        else:
             metrics['Variance_Ratio'] = np.nan

    except Exception as e:
        logging.error(f"Error calculating metrics for {model_name}: {e}")
        metrics = {k: np.nan for k in ['RMSE', 'MAE', 'MAPE', 'Directional_Accuracy', 'Variance_Ratio']}
    return metrics

# --- Helper Function for Price Conversion ---
def convert_logret_forecast_to_price(
    log_ret_forecast: pd.Series,
    last_actual_price: float
) -> Optional[pd.Series]:
    """Converts a forecast of log returns back to price levels."""
    if log_ret_forecast is None or log_ret_forecast.empty: return None
    if not np.isfinite(last_actual_price):
        logging.error("Last actual price is non-finite. Cannot convert log returns.")
        return None
    try:
        log_ret_forecast = log_ret_forecast.replace([np.inf, -np.inf], 0).fillna(0)
        # Use numpy's cumsum for potential speedup and broadcasting
        price_forecast_values = np.exp(np.cumsum(log_ret_forecast.values)) * last_actual_price
        # Check for non-finite values *after* calculation
        if not np.all(np.isfinite(price_forecast_values)):
            logging.warning("Non-finite values encountered after converting log returns to price. Attempting to handle.")
            temp_series = pd.Series(price_forecast_values) # Convert to series for filling
            price_forecast_values = temp_series.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill').values
            if np.isnan(price_forecast_values).any(): # Check if NaNs remain
                 logging.error("Could not fully resolve non-finite values in price forecast.")
                 return None
        return pd.Series(price_forecast_values, index=log_ret_forecast.index)
    except Exception as e:
        logging.error(f"Error converting log return forecast to price: {e}")
        return None

# Modified: Added more robust checks
def convert_logprice_forecast_to_price(
    log_price_forecast: pd.Series
) -> Optional[pd.Series]:
    """Converts a forecast of log prices back to price levels."""
    if log_price_forecast is None or log_price_forecast.empty: return None
    try:
        # Check for non-finite values *before* exponentiating
        if not np.all(np.isfinite(log_price_forecast)):
             logging.warning("Non-finite values found in log price forecast before exponentiation. Attempting to fill...")
             log_price_forecast = log_price_forecast.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
             if log_price_forecast.isnull().any():
                  logging.error("Could not fill all NaNs in log price forecast. Price conversion may fail.")
                  # Return None if critical NaNs remain before exp
                  return None

        logging.debug(f"Min/Max log price forecast before exp: {log_price_forecast.min():.4f} / {log_price_forecast.max():.4f}")

        with np.errstate(over='ignore', invalid='ignore'): # Ignore invalid value encountered in exp
             price_forecast = np.exp(log_price_forecast)

        # Check for Inf/NaN/Zero *after* exponentiating
        if not np.all(np.isfinite(price_forecast)) or (price_forecast < 1e-9).any():
            logging.warning("Infinite, NaN, or near-zero values encountered after exponentiating log price forecast. Check integration step or input forecast. Attempting to clean.")
            price_forecast = price_forecast.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
            # If still NaN, it means the fill failed (e.g., all NaNs initially)
            if price_forecast.isnull().any():
                 logging.error("Could not fully resolve non-finite values in final price forecast.")
                 return None

        return price_forecast
    except Exception as e:
        logging.error(f"Error converting log price forecast to price: {e}")
        return None


# --- Data Loading and Fetching ---

def load_top_predictors(csv_path: str, top_n: int) -> List[str]:
    """Loads top N predictor IDs from the bulk analyzer's CSV output file."""
    logging.info(f"Loading top {top_n} predictors from: {csv_path}")
    try:
        df = pd.read_csv(csv_path, index_col=0)
        logging.info(f"Read CSV file: {csv_path}")
        if 'rank' not in df.columns:
            if 'score' in df.columns:
                 logging.warning("'rank' column not found, sorting by 'score' instead.")
                 df = df.sort_values(by='score', ascending=False)
                 df['rank'] = range(1, len(df) + 1)
            else:
                 logging.error(f"'rank' or 'score' column not found in file '{csv_path}'. Cannot determine top predictors.")
                 return []
        else:
             df = df.sort_values(by='rank', ascending=True)
        top_predictors = df.head(top_n).index.tolist()
        logging.info(f"Selected Top {len(top_predictors)} Predictors: {top_predictors}")
        return top_predictors
    except FileNotFoundError:
        logging.error(f"Input CSV file not found at: {csv_path}")
        return []
    except Exception as e:
        logging.error(f"Error reading input CSV file {csv_path}: {e}")
        return []

def fetch_stock_data(ticker: str, start: str, end: str, use_log_returns: bool = False) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Fetches daily Adjusted Close price and optionally Log Returns for a stock.
    Returns (price_series, log_return_series)
    """
    logging.info(f"Fetching stock data for {ticker} from {start} to {end}...")
    price_series = None
    log_return_series = None
    try:
        stock = yf.Ticker(ticker)
        data_hist = stock.history(start=start, end=end, auto_adjust=False) # Get OHLCV
        if data_hist.empty:
            logging.warning(f"No data returned for {ticker}.")
            return None, None

        data = data_hist['Adj Close'] # Keep using Adj Close for price series base
        data.index = pd.to_datetime(data.index).tz_localize(None)
        data = data.asfreq("D")

        price_series_name = f"{ticker}_AdjClose"
        price_series = data.rename(price_series_name)

        if use_log_returns:
            if (data <= 0).any():
                logging.warning(f"Ticker {ticker} has non-positive prices. Cannot calculate log returns.")
            else:
                log_data = np.log(data / data.shift(1))
                log_return_series_name = f"{ticker}_LogReturn"
                log_return_series = log_data.iloc[1:].rename(log_return_series_name)

        return price_series, log_return_series

    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return None, None

def fetch_fred_series_data(series_id: str, api_key: str, start_date: str, end_date: str) -> Optional[pd.Series]:
    """Fetches observations for a single FRED series."""
    logging.debug(f"Fetching data for FRED series: {series_id}...")
    params = {
        "series_id": series_id, "api_key": api_key, "file_type": "json",
        "observation_start": start_date, "observation_end": end_date,
        "aggregation_method": "avg",
    }
    try:
        response = requests.get(FRED_API_BASE_URL_OBS, params=params)
        response.raise_for_status()
        data = response.json()
        obs = data.get("observations", [])
        if not obs: return None
        df = pd.DataFrame(obs)[["date", "value"]]
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna().set_index("date")
        return df["value"].rename(series_id) if not df.empty else None
    except requests.exceptions.RequestException as req_e:
        logging.error(f"HTTP Error fetching data for {series_id}: {req_e}")
        return None
    except Exception as e:
        logging.error(f"Error processing data for {series_id}: {e}")
        return None

def fetch_modeling_data(
    target_ticker: str,
    predictor_ids: List[str],
    start: str,
    end: str,
    use_log_returns: bool
) -> Tuple[Optional[pd.DataFrame], Optional[str], Optional[str]]:
    """
    Fetches target (price and optionally log returns) and predictor data, aligns them.
    Returns (combined_df, target_price_col, target_logret_col)
    """
    logging.info("--- Fetching Data for Modeling ---")
    price_series, log_return_series = fetch_stock_data(target_ticker, start, end, use_log_returns)

    if price_series is None:
        logging.error("Failed to fetch target price series data. Exiting.")
        return None, None, None

    target_price_col = price_series.name
    target_logret_col = log_return_series.name if log_return_series is not None else None

    api_key = get_fred_api_key()
    if not api_key:
        logging.error("FRED API Key not found. Exiting.")
        return None, None, None

    predictor_dict = {}
    fetched_count = 0
    for i, sid in enumerate(predictor_ids):
        if (i + 1) % 25 == 0: logging.info(f"Fetching predictor {i+1}/{len(predictor_ids)}...")
        series = fetch_fred_series_data(sid, api_key, start, end)
        if series is not None:
            predictor_dict[sid] = series
            fetched_count += 1
        time.sleep(REQUEST_DELAY_SECONDS)
    logging.info(f"Successfully fetched {fetched_count}/{len(predictor_ids)} predictor series.")

    logging.info("Aligning data...")
    primary_target_series = log_return_series if use_log_returns and log_return_series is not None else price_series
    if primary_target_series is None:
         logging.error("Primary target series for alignment is missing.")
         return None, None, None

    daily_idx = pd.date_range(primary_target_series.index.min(), primary_target_series.index.max(), freq='D')
    combined_df = pd.DataFrame(index=daily_idx)

    combined_df[target_price_col] = price_series.reindex(daily_idx)
    if target_logret_col and log_return_series is not None:
         combined_df[target_logret_col] = log_return_series.reindex(daily_idx)

    for sid, series in predictor_dict.items():
        series.index = pd.to_datetime(series.index).tz_localize(None)
        daily_s = series.resample('D').ffill()
        combined_df[sid] = daily_s.reindex(daily_idx)

    primary_target_col = target_logret_col if use_log_returns and target_logret_col else target_price_col
    combined_df = combined_df.dropna(subset=[primary_target_col])
    combined_df = combined_df.ffill().bfill()
    combined_df = combined_df.dropna(axis=1, how='all')

    logging.info(f"Aligned data shape: {combined_df.shape}")
    if primary_target_col not in combined_df.columns:
         logging.error(f"Primary target column '{primary_target_col}' lost during alignment.")
         return None, None, None

    for col in combined_df.columns:
         try:
             combined_df[col] = pd.to_numeric(combined_df[col])
         except ValueError:
             logging.warning(f"Could not convert column '{col}' to numeric. Dropping it.")
             combined_df = combined_df.drop(columns=[col])
    combined_df = combined_df.dropna(axis=1, how='all')

    return combined_df, target_price_col, target_logret_col


# --- Data Preparation ---
def prepare_data_for_model(
    df: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
    test_size: float = 0.001,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    """
    Splits data, performs stationarity tests on the target_col, and differencing if needed.
    Returns: Tuple: train_df, test_df, train_df_diff, test_df_diff, target_differenced
    """
    logging.info(f"--- Preparing Data for Models (Target: {target_col}) ---")
    all_cols = [target_col] + predictor_cols
    price_col = target_col.replace('_LogReturn', '_AdjClose')
    if price_col != target_col and price_col not in df.columns:
         logging.warning(f"Price column '{price_col}' not found, needed for conversions.")
    if price_col != target_col and price_col not in all_cols:
         all_cols.append(price_col)


    missing_cols = [col for col in all_cols if col not in df.columns]
    if missing_cols:
        logging.error(f"Columns not found in DataFrame: {missing_cols}. Aborting preparation.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False

    model_df = df[all_cols].apply(pd.to_numeric, errors='coerce').copy()
    rows_before_na = len(model_df)
    model_df = model_df.dropna(subset=[target_col])
    rows_after_na = len(model_df)
    if rows_before_na > rows_after_na:
         logging.warning(f"Dropped {rows_before_na - rows_after_na} rows where target '{target_col}' was NaN before splitting.")
    if model_df.empty:
         logging.error("DataFrame is empty after dropping target NaNs. Cannot proceed.")
         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False

    if isinstance(test_size, float) and 0 < test_size < 1:
        n_test = int(len(model_df) * test_size)
    elif isinstance(test_size, int) and 0 < test_size < len(model_df):
        n_test = test_size
    else:
        logging.warning(f"Invalid test_size '{test_size}'. Defaulting to 20%.")
        n_test = int(len(model_df) * 0.2)
    if n_test == 0: n_test = 1
    if n_test >= len(model_df):
         logging.error("Test set size is >= total data size. Cannot split.")
         return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), False

    train_df = model_df.iloc[:-n_test]
    test_df = model_df.iloc[-n_test:]
    logging.info(f"Train/Test Split: Train={train_df.shape[0]} samples, Test={test_df.shape[0]} samples")

    logging.info(f"Checking stationarity of target '{target_col}' on training data...")
    target_p, target_stationary = test_stationarity(train_df[target_col], target_col)

    predictors_stationary = True
    for col in predictor_cols:
        if col in train_df.columns:
             p_val, p_stationary = test_stationarity(train_df[col], col)
             if not p_stationary:
                 predictors_stationary = False
        else:
             logging.warning(f"Predictor column '{col}' not found in train_df for stationarity check.")


    target_differenced = False
    needs_diff_for_var_xgb = False
    if not target_stationary:
         target_differenced = True
         needs_diff_for_var_xgb = True
         logging.info(f"Target '{target_col}' is non-stationary. Will use differenced data for VAR/XGBoost.")
    elif not predictors_stationary:
         needs_diff_for_var_xgb = True
         logging.info(f"Target '{target_col}' is stationary, but predictors are not. Will use differenced data for VAR/XGBoost.")
    else:
        logging.info(f"Target '{target_col}' and predictors appear stationary on training data.")

    modeling_cols = [target_col] + predictor_cols
    train_df_diff = train_df[modeling_cols].diff().dropna()
    test_df_diff = test_df[modeling_cols].diff().dropna()

    if train_df_diff.empty and needs_diff_for_var_xgb:
         logging.error("Differencing resulted in an empty training dataframe.")
         train_df_diff = pd.DataFrame()
         test_df_diff = pd.DataFrame()

    return train_df, test_df, train_df_diff, test_df_diff, target_differenced


# --- Model Training & Evaluation ---

def train_evaluate_sarimax(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    top_predictor_col: Optional[str] = None
) -> Tuple[Optional[Any], Optional[pd.Series], Optional[Dict[str, float]], Optional[pd.Series]]:
    """
    Trains and evaluates a SARIMAX model using statsmodels.
    Performs a grid search over p, d, q orders to find the best fit based on AIC.
    """
    logging.info("--- Training and Evaluating SARIMAX (statsmodels, Grid Search) ---")

    y_train = train_df[target_col]
    y_test = test_df[target_col]
    X_train = train_df[[top_predictor_col]].astype(float) if top_predictor_col and top_predictor_col in train_df.columns else None
    X_test = test_df[[top_predictor_col]].astype(float) if top_predictor_col and top_predictor_col in test_df.columns else None


    if X_train is not None and X_train.isnull().values.any():
         logging.warning(f"NaNs found in SARIMAX exogenous train variable '{top_predictor_col}'. Filling with mean.")
         X_train = X_train.fillna(X_train.mean())
    if X_test is not None and X_test.isnull().values.any():
         logging.warning(f"NaNs found in SARIMAX exogenous test variable '{top_predictor_col}'. Filling with train mean.")
         fill_value = X_train.mean() if X_train is not None else X_test.mean()
         X_test = X_test.fillna(fill_value)

    best_aic = np.inf
    best_order = None
    best_model_fitted = None
    # Expanded grid search ranges
    logging.info(f"Starting SARIMAX grid search (p={list(SARIMAX_P_RANGE)}, d={list(SARIMAX_D_RANGE)}, q={list(SARIMAX_Q_RANGE)})... (This may take time)")
    pdq_combinations = list(itertools.product(SARIMAX_P_RANGE, SARIMAX_D_RANGE, SARIMAX_Q_RANGE))
    search_start_time = time.time()

    for i, order in enumerate(pdq_combinations):
        if (i + 1) % 10 == 0: logging.info(f"  Grid search progress: {i+1}/{len(pdq_combinations)}")
        try:
            if order == (0,0,0): continue
            model_temp = sm.tsa.statespace.SARIMAX(endog=y_train, exog=X_train,
                                                   order=order, seasonal_order=SARIMAX_SEASONAL_ORDER,
                                                   enforce_stationarity=False, enforce_invertibility=False)
            results_temp = model_temp.fit(disp=False)
            if results_temp.aic < best_aic:
                best_aic = results_temp.aic
                best_order = order
                best_model_fitted = results_temp
        except Exception: continue # Ignore errors during search

    search_duration = time.time() - search_start_time
    logging.info(f"SARIMAX grid search finished in {search_duration:.2f} seconds.")

    if best_model_fitted is None or best_order is None:
        logging.error("SARIMAX grid search failed to find a valid model.")
        return None, None, None, None

    logging.info(f"Best SARIMAX order found: {best_order} with AIC: {best_aic:.4f}")
    logging.info(best_model_fitted.summary())

    try:
        logging.info("Generating SARIMAX forecast on test set using best model...")
        forecast_obj = best_model_fitted.get_forecast(steps=len(y_test), exog=X_test)
        forecast_series = forecast_obj.predicted_mean
        forecast_series.index = y_test.index
        metrics = calculate_metrics(y_test.values, forecast_series.values, model_name="SARIMAX")
        logging.info(f"SARIMAX Metrics (Target: {target_col}): {metrics}")
        residuals = best_model_fitted.resid
        return best_model_fitted, forecast_series, metrics, residuals
    except Exception as e:
        logging.error(f"Error during SARIMAX forecasting/evaluation with best model: {e}", exc_info=True)
        return best_model_fitted, None, None, None

def train_evaluate_var(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df_diff: pd.DataFrame,
    test_df_diff: pd.DataFrame,
    target_col: str, # This is the column VAR is trained on (potentially differenced)
    predictor_cols: List[str],
    target_was_differenced: bool,
    target_level_col: str # The original level column name (price or logret)
) -> Tuple[Optional[Any], Optional[pd.Series], Optional[Dict[str, float]], Optional[pd.Series]]:
    """
    Trains and evaluates a VAR model. Integrates forecast back to levels
    for plotting and comparable evaluation using target_level_col.
    """
    logging.info("--- Training and Evaluating VAR ---")
    var_cols = [target_col] + predictor_cols
    missing_cols = [col for col in var_cols if col not in train_df_diff.columns]
    if missing_cols:
         logging.error(f"Missing columns in differenced training data for VAR: {missing_cols}")
         return None, None, None, None
    if train_df_diff.empty or test_df_diff.empty:
         logging.error("Differenced train or test data is empty for VAR.")
         return None, None, None, None

    df_train_var = train_df_diff[var_cols].copy()
    df_test_var = test_df_diff[var_cols].copy()
    df_train_var = df_train_var.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    df_test_var = df_test_var.astype(float)
    if df_train_var.empty:
         logging.error("Training data for VAR became empty after handling NaNs/Infs.")
         return None, None, None, None

    model_fitted = None
    forecast_series_level = None
    metrics_level = None
    residuals = None

    try:
        logging.info("Selecting optimal VAR lag order using AIC...")
        model_select = VAR(df_train_var)
        n_vars = len(var_cols)
        maxlags_to_test = min(DEFAULT_GRANGER_MAX_LAG_SEARCH, int(len(df_train_var)**(1/3)) )
        maxlags_to_test = min(maxlags_to_test, (len(df_train_var) - 1) // n_vars -1)
        if maxlags_to_test < 1: maxlags_to_test = 1
        logging.info(f"Testing VAR lags up to {maxlags_to_test}")
        best_lag = 1
        try:
             if maxlags_to_test > 0:
                  selected_orders = model_select.select_order(maxlags=maxlags_to_test)
                  best_lag = selected_orders.aic
                  logging.info(f"Optimal VAR lag selected by AIC: {best_lag}")
             else:
                  logging.warning("Not enough data for VAR lag selection. Defaulting to lag 1.")
        except Exception as e:
             logging.warning(f"VAR lag selection failed: {e}. Defaulting to lag 1.")

        logging.info(f"Fitting VAR({best_lag})...")
        model = VAR(df_train_var)
        model_fitted = model.fit(best_lag)
        logging.info(model_fitted.summary())
        logging.info("Generating VAR forecast on test set (differenced scale)...")
        lag_order = model_fitted.k_ar
        if len(df_train_var) < lag_order:
             logging.error(f"Training data length ({len(df_train_var)}) < VAR lag order ({lag_order}). Cannot forecast.")
             return model_fitted, None, None, None
        forecast_input = df_train_var.values[-lag_order:]
        try:
            forecast_diff = model_fitted.forecast(y=forecast_input, steps=len(df_test_var))
        except Exception as fc_e:
            logging.error(f"VAR forecast failed: {fc_e}")
            return model_fitted, None, None, None

        target_col_index = df_train_var.columns.get_loc(target_col)
        target_forecast_diff = forecast_diff[:, target_col_index]
        forecast_series_diff = pd.Series(target_forecast_diff, index=df_test_var.index)

        logging.info("Integrating VAR forecast back to level scale...")
        # Ensure the level column exists in train_df
        if target_level_col not in train_df.columns:
             logging.error(f"Target level column '{target_level_col}' not found in train_df for integration.")
             return model_fitted, None, None, model_fitted.resid[target_col] if model_fitted else None

        last_actual_level = train_df[target_level_col].iloc[-1]
        # Check if last level is valid
        if not np.isfinite(last_actual_level):
             logging.error(f"Last actual level value for '{target_level_col}' is not finite. Cannot integrate VAR forecast.")
             return model_fitted, None, None, model_fitted.resid[target_col] if model_fitted else None

        # Ensure forecast differences are finite before cumsum
        forecast_series_diff = forecast_series_diff.replace([np.inf, -np.inf], 0).fillna(0)
        integrated_forecast = last_actual_level + forecast_series_diff.cumsum()
        forecast_series_level = integrated_forecast
        # Log integrated forecast stats
        logging.debug(f"Integrated VAR forecast (level) stats: Min={forecast_series_level.min():.4f}, Max={forecast_series_level.max():.4f}, Mean={forecast_series_level.mean():.4f}")


        y_test_level = test_df[target_level_col]
        eval_df_level = pd.concat([y_test_level, forecast_series_level], axis=1).dropna()
        if not eval_df_level.empty:
             metrics_level = calculate_metrics(eval_df_level.iloc[:, 0].values, eval_df_level.iloc[:, 1].values, model_name="VAR Level")
             logging.info(f"VAR Metrics (on Level Scale - Target: {target_level_col}): {metrics_level}")
        else:
             logging.warning("Could not evaluate VAR on level scale due to alignment issues after integration.")
             metrics_level = {k: np.nan for k in ['RMSE', 'MAE', 'MAPE', 'Directional_Accuracy', 'Variance_Ratio']}
        residuals = model_fitted.resid[target_col]

    except Exception as e:
        logging.error(f"Error during VAR training/evaluation: {e}", exc_info=True)
        return model_fitted, None, None, None

    return model_fitted, forecast_series_level, metrics_level, residuals

# Modified: Includes feature engineering and hyperparameter tuning
def train_evaluate_xgboost_quantile(
    train_df: pd.DataFrame, # Contains target_col and predictors
    test_df: pd.DataFrame,
    target_col: str,
    predictor_cols: List[str],
    lags: int = DEFAULT_XGB_LAGS,
    scale_features: bool = True,
    tune_hyperparams: bool = True,
    add_garch_feature: bool = True, # Flag to add GARCH feature
    quantile: float = 0.5 # Quantile to predict (0.5 for median)
) -> Tuple[Optional[Any], Optional[pd.Series], Optional[Dict[str, float]], Optional[pd.DataFrame], Optional[pd.Series]]:
    """
    Trains and evaluates an XGBoost model for a specific quantile.
    Includes feature engineering and optional hyperparameter tuning.
    """
    logging.info(f"--- Training and Evaluating XGBoost (Quantile={quantile:.2f}) ---")
    logging.info(f"Creating {lags} lags and volatility features...")

    # Combine for feature engineering
    df_full = pd.concat([train_df, test_df])
    feature_names = []

    # --- Feature Engineering ---
    # 1. Lagged Target
    for i in range(1, lags + 1):
        feat_name = f'{target_col}_lag_{i}'
        df_full[feat_name] = df_full[target_col].shift(i)
        feature_names.append(feat_name)
    # 2. Lagged Predictors
    for pred_col in predictor_cols:
        if pred_col in df_full.columns:
            for i in range(1, lags + 1):
                feat_name = f'{pred_col}_lag_{i}'
                df_full[feat_name] = df_full[pred_col].shift(i)
                feature_names.append(feat_name)
        else:
            logging.warning(f"Predictor column '{pred_col}' not found for lagging in XGBoost.")

    # 3. Rolling Mean/Std (Target) - Captures trend and volatility
    for window in VOLATILITY_WINDOWS:
         feat_name_ma = f'{target_col}_MA_{window}'
         feat_name_std = f'{target_col}_STD_{window}'
         # Use the original target column for rolling features
         df_full[feat_name_ma] = df_full[target_col].rolling(window=window, min_periods=1).mean().shift(1) # Shift to avoid lookahead
         df_full[feat_name_std] = df_full[target_col].rolling(window=window, min_periods=1).std().shift(1) # Use std() for volatility
         feature_names.extend([feat_name_ma, feat_name_std])

    # 4. Time Features
    if isinstance(df_full.index, pd.DatetimeIndex):
         df_full['day_of_week'] = df_full.index.dayofweek
         df_full['month'] = df_full.index.month
         feature_names.extend(['day_of_week', 'month'])
    else:
         logging.warning("DataFrame index is not DatetimeIndex, cannot create time features.")

    # --- Optional: Add GARCH Volatility Forecast Feature ---
    if add_garch_feature and arch_available:
        try:
            logging.info("Attempting to add GARCH volatility feature...")
            garch_target = train_df[target_col].dropna() * 100
            if not garch_target.empty and garch_target.var() > 1e-6: # Check variance before fitting
                garch_model = arch_model(garch_target, mean='Constant', vol='Garch', p=1, q=1, rescale=False)
                garch_results = garch_model.fit(disp='off', show_warning=False)
                forecast_horizon = len(df_full) - len(train_df)
                garch_forecast = garch_results.forecast(horizon=forecast_horizon, start=train_df.index[-1], reindex=False)
                cond_variance_forecast = garch_forecast.variance.iloc[-1]
                cond_vol_forecast = np.sqrt(cond_variance_forecast) / 100
                forecast_index = df_full.index[len(train_df):len(train_df)+len(cond_vol_forecast)]
                if len(forecast_index) == len(cond_vol_forecast):
                    cond_vol_series = pd.Series(cond_vol_forecast.values, index=forecast_index)
                    df_full['garch_vol_forecast'] = cond_vol_series.shift(1).reindex(df_full.index)
                    df_full['garch_vol_forecast'] = df_full['garch_vol_forecast'].fillna(method='bfill').fillna(method='ffill') # Fill NaNs robustly
                    if 'garch_vol_forecast' in df_full.columns:
                         feature_names.append('garch_vol_forecast')
                         logging.info("Added GARCH volatility feature.")
                    else:
                         logging.warning("GARCH volatility feature column creation failed.")
                else:
                     logging.warning("GARCH forecast length/index mismatch. Skipping feature.")
            else:
                logging.warning("Target series empty or constant for GARCH fitting.")
        except Exception as garch_e:
            logging.warning(f"Failed to fit GARCH model or add feature: {garch_e}")

    df_full = df_full.dropna()
    train_featured = df_full.loc[train_df.index.intersection(df_full.index)]
    test_featured = df_full.loc[test_df.index.intersection(df_full.index)]
    if train_featured.empty or test_featured.empty:
        logging.error("Train or test set empty after creating features. Cannot train XGBoost.")
        return None, None, None, None, None

    X_train = train_featured[feature_names]
    y_train = train_featured[target_col]
    X_test = test_featured[feature_names]
    y_test = test_featured[target_col]

    # --- Optional Scaling ---
    scaler = None
    if scale_features:
        logging.info("Scaling features for XGBoost...")
        try:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            X_train = X_train_scaled
            X_test = X_test_scaled
        except Exception as e:
            logging.error(f"Error during feature scaling: {e}. Proceeding without scaling.")
            X_train = train_featured[feature_names].values
            X_test = test_featured[feature_names].values

    logging.info(f"XGBoost features created. Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    try:
        # --- Hyperparameter Tuning (Randomized Search) ---
        if tune_hyperparams:
            logging.info(f"Starting XGBoost hyperparameter tuning (RandomizedSearchCV, n_iter={XGB_N_ITER_SEARCH}, Quantile={quantile:.2f})...")
            # Increased parameter ranges
            param_dist = {
                'n_estimators': randint(100, 1501),
                'learning_rate': uniform(0.005, 0.295),
                'max_depth': randint(3, 16),
                'subsample': uniform(0.5, 0.5), # 0.5 to 1.0
                'colsample_bytree': uniform(0.5, 0.5),
                'gamma': uniform(0, 0.6),
                'reg_alpha': uniform(0, 0.5),
                'reg_lambda': uniform(0, 1.5)
            }
            tscv = TimeSeriesSplit(n_splits=XGB_CV_SPLITS)
            # Removed early stopping from RandomizedSearchCV fit_params
            # It's complex to set up correctly with CV and might lead to issues.
            xgb_model_base = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile,
                                              random_state=42, n_jobs=-1)
            random_search = RandomizedSearchCV(
                estimator=xgb_model_base, param_distributions=param_dist,
                n_iter=XGB_N_ITER_SEARCH, scoring=make_scorer(mae_scorer, greater_is_better=True), # Use MAE for quantile
                cv=tscv, verbose=1, random_state=42, n_jobs=-1, refit=True # refit=True is default
            )
            search_start_time = time.time()
            random_search.fit(X_train, y_train) # Fit without early stopping during search
            search_duration = time.time() - search_start_time
            logging.info(f"Hyperparameter search finished in {search_duration:.2f} seconds.")
            logging.info(f"Best XGBoost params found for Quantile {quantile:.2f}: {random_search.best_params_}")
            logging.info(f"Best XGBoost score (Negative MAE on CV): {random_search.best_score_:.4f}")
            model = random_search.best_estimator_ # Already fitted with best params on full X_train

        else:
             logging.info(f"Using default XGBoost parameters for Quantile {quantile:.2f}.")
             model = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=quantile,
                                     n_estimators=100, learning_rate=0.1, max_depth=5,
                                     subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
             logging.info("Fitting XGBoost model with default parameters...")
             model.fit(X_train, y_train, verbose=False) # No early stopping if not tuning


        # --- Final Evaluation ---
        logging.info(f"Generating XGBoost Quantile={quantile:.2f} forecast on test set...")
        predictions = model.predict(X_test)
        forecast_series = pd.Series(predictions, index=y_test.index)

        metrics = None
        feature_importance = None
        residuals = None

        # Calculate metrics only for the median forecast
        if quantile == XGB_QUANTILE_MEDIAN:
            metrics = calculate_metrics(y_test.values, forecast_series.values, model_name=f"XGBoost Median")
            logging.info(f"XGBoost Median Metrics (Target: {target_col}): {metrics}")
            if hasattr(model, 'feature_importances_'):
                # Determine correct feature names (handle scaling)
                current_feature_names = feature_names # Default if no scaling/other issues
                if scaler is not None:
                    try: current_feature_names = scaler.get_feature_names_out(feature_names)
                    except: pass # Use original names if scaler fails
                elif isinstance(X_train, pd.DataFrame): # Check if original DF was used
                     current_feature_names = X_train.columns.tolist()

                # Ensure importance array length matches feature names length
                if len(current_feature_names) == len(model.feature_importances_):
                     feature_importance = pd.DataFrame({'Feature': current_feature_names, 'Importance': model.feature_importances_})
                     feature_importance = feature_importance.sort_values(by='Importance', ascending=False).reset_index(drop=True)
                     logging.debug(f"XGBoost Feature Importances (Median Model):\n{feature_importance.head()}")
                else:
                     logging.warning(f"Feature name length ({len(current_feature_names)}) mismatch with importance length ({len(model.feature_importances_)}). Skipping importance calculation.")

            residuals = y_test - forecast_series


        # Return model, forecast, metrics (only for median), importance (only for median), residuals (only for median)
        return model, forecast_series, metrics, feature_importance, residuals
    except Exception as e:
        logging.error(f"Error during XGBoost Quantile={quantile:.2f} training/evaluation: {e}", exc_info=True)
        return None, None, None, None, None


# --- Plotting Functions ---

# Modified: Handles quantile forecasts for XGBoost
def plot_forecast_comparison(
    actual_train: pd.Series,
    actual_test: pd.Series,
    forecasts_dict: Dict[str, Optional[pd.Series]],  # Includes 'XGBoost_Avg', 'XGBoost Q0.25', 'XGBoost Q0.75'
    target_col: str,
    output_dir: str,
    plot_title_suffix: str = ""
):
    """Plots actual vs predicted values, showing XGBoost quantiles as a band."""
    logging.info(f"Generating forecast comparison plot ({plot_title_suffix or 'Levels'})...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 7))
    plt.plot(actual_train.index, actual_train, label='Train Actual', color='black', alpha=0.7)
    plt.plot(actual_test.index, actual_test, label='Test Actual', color='blue', linewidth=2)

    # Define keys explicitly
    xgb_low_key = f'XGBoost Q{XGB_QUANTILE_LOW:.2f}'
    xgb_high_key = f'XGBoost Q{XGB_QUANTILE_HIGH:.2f}'
    xgb_avg_key = 'XGBoost_Avg'

    xgb_low = forecasts_dict.get(xgb_low_key, None)
    xgb_high = forecasts_dict.get(xgb_high_key, None)
    xgb_avg = forecasts_dict.get(xgb_avg_key, None)

    colors = plt.cm.viridis(np.linspace(0, 1, len(forecasts_dict)))

    plot_idx = 0
    # Plot forecasts (excluding XGBoost quantiles to plot separately)
    for model_name, forecast in forecasts_dict.items():
        if model_name in [xgb_low_key, xgb_high_key, xgb_avg_key]:
            continue  # Skip XGBoost keys for separate plotting
        if forecast is not None:
            common_index = actual_test.index.intersection(forecast.index)
            if not common_index.empty:
                plt.plot(common_index, forecast.loc[common_index], label=f'{model_name} Forecast',
                         linestyle='--', color=colors[plot_idx])
                plot_idx += 1
            else:
                logging.warning(f"Index mismatch for {model_name}; skipping plot.")

    # Plot XGBoost average forecast
    if xgb_avg is not None:
        common_index_avg = actual_test.index.intersection(xgb_avg.index)
        plt.plot(common_index_avg, xgb_avg.loc[common_index_avg], label='XGBoost Average Forecast',
                 linestyle='-', color='orange', linewidth=2)

    # Plot quantile range band
    if xgb_low is not None and xgb_high is not None:
        common_index_band = actual_test.index.intersection(xgb_low.index).intersection(xgb_high.index)
        if not common_index_band.empty:
            low_bound = np.minimum(xgb_low.loc[common_index_band], xgb_high.loc[common_index_band])
            high_bound = np.maximum(xgb_low.loc[common_index_band], xgb_high.loc[common_index_band])
            plt.fill_between(common_index_band, low_bound, high_bound,
                             color='orange', alpha=0.3,
                             label=f'XGBoost {int(XGB_QUANTILE_LOW*100)}-{int(XGB_QUANTILE_HIGH*100)}th Percentile')
        else:
            logging.warning("Index mismatch for XGBoost quantile bands; skipping plot.")
    else:
        logging.warning("Missing XGBoost quantile forecasts; skipping quantile band.")

    title = f'Actual vs. Forecasted {target_col}'
    if plot_title_suffix:
        title += f" ({plot_title_suffix})"
    plt.title(title, fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    try:
        filename_suffix = plot_title_suffix.lower().replace(" ", "_") if plot_title_suffix else "levels"
        filename = os.path.join(output_dir, f"{target_col}_forecast_comparison_{filename_suffix}.png")
        plt.savefig(filename, dpi=300)
        logging.info(f"Saved forecast comparison plot to {filename}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to save forecast comparison plot: {e}")

# Modified: Handles quantile forecasts for XGBoost price conversion
def plot_price_forecast_comparison(
    actual_train_price: pd.Series,
    actual_test_price: pd.Series,
    price_forecasts_dict: Dict[str, Optional[pd.Series]],  # Contains SARIMAX, VAR, XGBoost_Avg, XGB_Low, XGB_High prices
    target_price_col: str,
    output_dir: str
):
    """Plots actual prices vs predicted prices, showing XGBoost quantiles as a band."""
    logging.info("Generating PRICE forecast comparison plot...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(14, 7))
    plt.plot(actual_train_price.index, actual_train_price, label='Train Actual Price', color='black', alpha=0.7)
    plt.plot(actual_test_price.index, actual_test_price, label='Test Actual Price', color='blue', linewidth=2)

    # Define XGB forecast keys explicitly
    xgb_low_key = f'XGBoost Q{XGB_QUANTILE_LOW:.2f}'
    xgb_high_key = f'XGBoost Q{XGB_QUANTILE_HIGH:.2f}'
    xgb_avg_key = 'XGBoost_Avg'

    xgb_low_price = price_forecasts_dict.get(xgb_low_key, None)
    xgb_high_price = price_forecasts_dict.get(xgb_high_key, None)
    xgb_avg_price = price_forecasts_dict.get(xgb_avg_key, None)

    colors = plt.cm.viridis(np.linspace(0, 1, len(price_forecasts_dict)))
    plot_idx = 0

    # Plot other models (SARIMAX, VAR)
    for model_name, forecast in price_forecasts_dict.items():
        if model_name in [xgb_low_key, xgb_high_key, xgb_avg_key]:
            continue  # Skip XGBoost forecasts for separate plotting
        if forecast is not None:
            common_index = actual_test_price.index.intersection(forecast.index)
            if not common_index.empty:
                plt.plot(common_index, forecast.loc[common_index], label=f'{model_name} Price Forecast', linestyle='--', color=colors[plot_idx])
                plot_idx += 1
            else:
                logging.warning(f"Could not plot PRICE forecast for {model_name} due to index mismatch or empty forecast.")
        else:
            logging.warning(f"PRICE forecast for {model_name} is None, skipping plot.")

    # Plot XGBoost average forecast
    if xgb_avg_price is not None:
        common_index_avg = actual_test_price.index.intersection(xgb_avg_price.index)
        plt.plot(common_index_avg, xgb_avg_price.loc[common_index_avg], label='XGBoost Avg Price Forecast', linestyle='-', color='orange', linewidth=2)

    # Plot XGBoost quantile range band
    if xgb_low_price is not None and xgb_high_price is not None:
        common_index_band = actual_test_price.index.intersection(xgb_low_price.index).intersection(xgb_high_price.index)
        if not common_index_band.empty:
            low_bound = np.minimum(xgb_low_price.loc[common_index_band], xgb_high_price.loc[common_index_band])
            high_bound = np.maximum(xgb_low_price.loc[common_index_band], xgb_high_price.loc[common_index_band])
            plt.fill_between(common_index_band, low_bound, high_bound,
                             color='orange', alpha=0.3,
                             label=f'XGBoost {int(XGB_QUANTILE_LOW*100)}-{int(XGB_QUANTILE_HIGH*100)}th Percentile Price')
        else:
            logging.warning("Could not plot XGBoost PRICE quantile band due to index mismatch.")

    plt.title(f'Actual vs. Forecasted Price ({target_price_col})', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price', fontsize=12)
    plt.legend(fontsize=10)
    plt.tight_layout()
    try:
        filename = os.path.join(output_dir, f"{target_price_col}_forecast_comparison_PRICE.png")
        plt.savefig(filename, dpi=300)
        logging.info(f"Saved PRICE forecast comparison plot to {filename}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to save PRICE forecast comparison plot: {e}")


def plot_residuals_diagnostics(
    model_name: str,
    residuals: Optional[pd.Series],
    output_dir: str,
    target_col: str
):
    """Plots residual histogram, Q-Q plot, and ACF."""
    if residuals is None or residuals.empty:
        logging.warning(f"No residuals provided for {model_name}. Skipping diagnostics plot.")
        return
    logging.info(f"Generating residual diagnostics plot for {model_name} (Scale: {target_col})...")
    residuals = residuals.dropna()
    if residuals.empty:
        logging.warning(f"Residuals are empty after dropping NaNs for {model_name}. Skipping.")
        return

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'{model_name} Residual Diagnostics ({target_col})', fontsize=16)
    sns.histplot(residuals, kde=True, ax=axes[0], bins=30)
    axes[0].set_title('Residual Histogram')
    try:
         if len(residuals) >= 8:
              stat, p_norm = normaltest(residuals)
              axes[0].text(0.05, 0.95, f'Normality p={p_norm:.3f}', transform=axes[0].transAxes, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
         else:
              axes[0].text(0.05, 0.95, 'Normality test skipped (N<8)', transform=axes[0].transAxes, ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
    except Exception as e:
         logging.warning(f"Normality test failed for {model_name}: {e}")
    try:
        sm.qqplot(residuals, line='s', ax=axes[1])
        axes[1].set_title('Q-Q Plot')
    except Exception as e:
        axes[1].set_title('Q-Q Plot (Error)')
        logging.error(f"Q-Q plot failed for {model_name}: {e}")
    try:
        nlags = min(40, len(residuals)//2 - 1)
        if nlags > 0:
             plot_acf(residuals, ax=axes[2], lags=nlags)
             axes[2].set_title('Residual ACF')
        else:
             axes[2].set_title('Residual ACF (N too small)')
             logging.warning(f"Not enough residuals to plot ACF for {model_name}")
    except Exception as e:
        axes[2].set_title('Residual ACF (Error)')
        logging.error(f"ACF plot failed for {model_name}: {e}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    try:
        filename = os.path.join(output_dir, f"{target_col}_{model_name}_residuals.png")
        plt.savefig(filename, dpi=300)
        logging.info(f"Saved residual diagnostics plot to {filename}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to save residual plot for {model_name}: {e}")


def plot_feature_importance(
    feature_importance_df: Optional[pd.DataFrame],
    output_dir: str,
    target_col: str
):
    """Plots XGBoost feature importance."""
    if feature_importance_df is None or feature_importance_df.empty:
        logging.warning("No feature importance data provided for XGBoost. Skipping plot.")
        return
    logging.info("Generating XGBoost feature importance plot...")
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.figure(figsize=(10, max(6, len(feature_importance_df) // 3)))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(30), palette='viridis')
    plt.title(f'XGBoost Feature Importance (Top 30) for {target_col}', fontsize=16)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    try:
        filename = os.path.join(output_dir, f"{target_col}_XGBoost_feature_importance.png")
        plt.savefig(filename, dpi=300)
        logging.info(f"Saved feature importance plot to {filename}")
        plt.close()
    except Exception as e:
        logging.error(f"Failed to save feature importance plot: {e}")


# --- Main Orchestration ---
def main():
    parser = argparse.ArgumentParser(description="Train and evaluate models using top FRED predictors.")
    parser.add_argument("csv_path", help="Path to the ranked predictors CSV file")
    parser.add_argument("target_ticker", help="Target stock ticker symbol (must match analysis)")
    parser.add_argument("start_date", help="Start date for modeling data (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date for modeling data (YYYY-MM-DD)")
    parser.add_argument("-n", "--top_n", type=int, default=5, help="Number of top predictors to use (default: 5)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Fraction of data for test set (e.g., 0.2 for 20%)")
    parser.add_argument("--use_log_returns", action='store_true', help="Use log returns for the target stock (recommended)")
    parser.add_argument("--xgb_lags", type=int, default=DEFAULT_XGB_LAGS, help=f"Number of lags for XGBoost features (default: {DEFAULT_XGB_LAGS})")
    parser.add_argument("--xgb_scale", action='store_true', help="Scale features before training XGBoost")
    parser.add_argument("--tune_xgb", action='store_true', help="Perform hyperparameter tuning for XGBoost (can be slow)")
    parser.add_argument("--add_garch", action='store_true', help="Attempt to add GARCH volatility feature to XGBoost (requires 'arch' library)")
    parser.add_argument("--output_dir", default="model_results", help="Directory to save plots and results (default: model_results)")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        logging.info(f"Created output directory: {args.output_dir}")

    # 1. Load Top Predictor IDs
    top_predictor_ids = load_top_predictors(args.csv_path, args.top_n)
    if not top_predictor_ids: return

    # 2. Fetch Modeling Data (Gets both price and logret if requested)
    base_ticker = args.target_ticker
    combined_df, target_price_col, target_logret_col = fetch_modeling_data(
        base_ticker, top_predictor_ids, args.start_date, args.end_date, args.use_log_returns
    )
    if combined_df is None: return

    # Determine the primary target column for modeling
    modeling_target_col = target_logret_col if args.use_log_returns and target_logret_col else target_price_col
    if modeling_target_col not in combined_df.columns:
         logging.error(f"Modeling target column '{modeling_target_col}' not found after data fetching/alignment.")
         return
    if target_price_col not in combined_df.columns:
         logging.error(f"Price column '{target_price_col}' not found after data fetching/alignment. Cannot convert forecasts to price.")
         return

    actual_predictor_cols = [pid for pid in top_predictor_ids if pid in combined_df.columns]
    if not actual_predictor_cols:
         logging.error("None of the top predictor IDs were found in the final aligned data.")
         return
    if len(actual_predictor_cols) < len(top_predictor_ids):
         logging.warning(f"Using {len(actual_predictor_cols)} predictors out of the requested {args.top_n} due to fetch/alignment issues.")

    # 3. Prepare Data for the specific target being modeled
    train_df, test_df, train_df_diff, test_df_diff, target_differenced = prepare_data_for_model(
        combined_df, modeling_target_col, actual_predictor_cols, args.test_size
    )
    if train_df.empty or test_df.empty:
        logging.error("Train or test DataFrame is empty after preparation. Exiting.")
        return

    # --- Model Runs ---
    model_results = {}
    # Store forecasts on the scale they were generated (logret or price level)
    level_forecasts = {}
    residuals = {}
    xgb_feat_importance = None

    # 4. SARIMAX
    top_1_predictor = actual_predictor_cols[0] if actual_predictor_cols else None
    sarimax_model, sarimax_fc, sarimax_metrics, sarimax_resid = train_evaluate_sarimax(
        train_df, test_df, modeling_target_col, top_1_predictor
    )
    if sarimax_metrics:
        model_results["SARIMAX"] = sarimax_metrics
        level_forecasts["SARIMAX"] = sarimax_fc # Forecast is on modeling_target_col scale
        residuals["SARIMAX"] = sarimax_resid

    # 5. VAR
    if not train_df_diff.empty and not test_df_diff.empty:
        var_model, var_fc_level, var_metrics_level, var_resid_diff = train_evaluate_var(
            train_df, test_df, train_df_diff, test_df_diff,
            modeling_target_col, # Target VAR was trained on (potentially differenced)
            actual_predictor_cols,
            target_differenced,
            modeling_target_col # Target level col for integration/evaluation
        )
        if var_metrics_level:
            model_results["VAR"] = var_metrics_level # Store level metrics
            level_forecasts["VAR"] = var_fc_level # Store level forecast
            residuals["VAR"] = var_resid_diff # Store diff residuals
    else:
         logging.warning("Skipping VAR model as differenced data is empty.")

    # 6. XGBoost (Quantile Regression)
    xgb_train_df = train_df_diff if target_differenced else train_df
    xgb_test_df = test_df_diff if target_differenced else test_df
    xgb_forecasts_quantiles = {}

    if not xgb_train_df.empty and not xgb_test_df.empty:
        quantiles_to_run = [XGB_QUANTILE_LOW, XGB_QUANTILE_HIGH]
        quantile_forecasts = []

        for q in quantiles_to_run:
            q_key = f'XGBoost Q{q:.2f}'
            _, xgb_fc_q, _, _, _ = train_evaluate_xgboost_quantile(
                xgb_train_df, xgb_test_df, modeling_target_col, actual_predictor_cols,
                lags=args.xgb_lags,
                scale_features=args.xgb_scale,
                tune_hyperparams=args.tune_xgb,
                add_garch_feature=args.add_garch,
                quantile=q
            )
            xgb_forecasts_quantiles[q_key] = xgb_fc_q
            quantile_forecasts.append(xgb_fc_q)

        # Compute average forecast from narrowed quantiles
        if all(fc is not None for fc in quantile_forecasts):
            avg_xgb_forecast = sum(quantile_forecasts) / len(quantile_forecasts)
            level_forecasts["XGBoost_Avg"] = avg_xgb_forecast

            # Calculate metrics for averaged forecast
            y_true = xgb_test_df[modeling_target_col]
            avg_metrics = calculate_metrics(y_true.values, avg_xgb_forecast.values, model_name="XGBoost_Avg")
            model_results["XGBoost_Avg"] = avg_metrics
            logging.info(f"XGBoost Average Metrics (Target: {modeling_target_col}): {avg_metrics}")
        else:
            logging.error("One or more XGBoost quantile forecasts failed; skipping average calculation.")

        # Integration back to level scale (if necessary)
        if target_differenced:
            logging.info("Integrating XGBoost averaged forecast back to level scale...")
            last_actual_level_xgb = train_df[modeling_target_col].iloc[-1]
            if np.isfinite(last_actual_level_xgb):
                level_forecasts["XGBoost_Avg"] = last_actual_level_xgb + level_forecasts["XGBoost_Avg"].cumsum()
    else:
        logging.warning("Skipping XGBoost model as input dataframes are empty.")


    # --- Reporting ---
    logging.info("\n--- Model Comparison ---")
    # Report metrics based on Median XGBoost forecast if available
    results_df = pd.DataFrame.from_dict(model_results, orient='index')
    logging.info("\nMetrics Summary (on prediction scale, XGBoost uses Median):\n" + results_df.to_string(float_format="%.4f"))
    try:
         results_filename = os.path.join(args.output_dir, f"{modeling_target_col}_model_metrics.csv")
         results_df.to_csv(results_filename)
         logging.info(f"Saved metrics summary to {results_filename}")
    except Exception as e:
         logging.error(f"Failed to save metrics summary: {e}")

    # --- Plotting ---
    logging.info("\n--- Generating Plots ---")
    # 1. Plot comparison on the modeling scale (logret or price) - includes XGB quantiles
    plot_forecast_comparison(
        train_df[modeling_target_col], test_df[modeling_target_col],
        level_forecasts.copy(), # Pass a copy so popping doesn't affect price conversion
        modeling_target_col, args.output_dir,
        plot_title_suffix="Log Returns" if args.use_log_returns else "Price Levels"
    )

    # 2. Convert forecasts to Price scale and plot
    logging.info("Converting level forecasts to Price scale for plotting...")
    price_forecasts = {}
    last_actual_price = train_df[target_price_col].iloc[-1]
    if not np.isfinite(last_actual_price):
         logging.error(f"Last actual price ({target_price_col}) is not finite. Cannot convert forecasts to price scale.")
    else:
        for model_name, fc_level in level_forecasts.items(): # fc_level is on modeling_target_col scale
            if fc_level is None: continue
            if args.use_log_returns and target_logret_col is not None:
                # If modeling target was log returns
                # --- Corrected Price Conversion Logic ---
                if model_name.startswith("XGBoost") or model_name == "SARIMAX":
                     # These models predicted log returns (fc_level is log returns)
                     price_forecasts[model_name] = convert_logret_forecast_to_price(fc_level, last_actual_price)
                elif model_name == "VAR":
                     # VAR fc_level is integrated log returns (log price)
                     price_forecasts[model_name] = convert_logprice_forecast_to_price(fc_level)
                # -----------------------------------------
            else:
                # If modeling target was price (fc_level is already price)
                price_forecasts[model_name] = fc_level

    # Plot price comparison - includes XGB quantiles converted to price
    plot_price_forecast_comparison(
        train_df[target_price_col],
        test_df[target_price_col],
        price_forecasts, # This now contains price forecasts for all models/quantiles
        target_price_col,
        args.output_dir
    )

    # 3. Plot Residuals (Residuals are on the scale the model was trained/evaluated on)
    if "SARIMAX" in residuals: plot_residuals_diagnostics("SARIMAX", residuals["SARIMAX"], args.output_dir, modeling_target_col)
    if "VAR" in residuals: plot_residuals_diagnostics("VAR", residuals["VAR"], args.output_dir, modeling_target_col + " (Diff)")
    if "XGBoost Median" in residuals: plot_residuals_diagnostics("XGBoost Median", residuals["XGBoost Median"], args.output_dir, modeling_target_col) # Use median residuals

    # 4. Plot Feature Importance (from median XGBoost model)
    if xgb_feat_importance is not None: plot_feature_importance(xgb_feat_importance, args.output_dir, modeling_target_col)

    logging.info(f"\n--- Script execution finished. Results saved in '{args.output_dir}' ---")

if __name__ == "__main__":
    main()
