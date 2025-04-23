import logging
import os
import sys
import time
import requests
from typing import Dict, List, Optional, Set, Any

# --- Configuration ---
FRED_API_BASE_URL = "https://api.stlouisfed.org/fred"
REQUEST_DELAY_SECONDS = 0.6
API_RETRY_DELAY = 5
API_MAX_RETRIES = 5
FRED_API_OFFSET_LIMIT = 1000

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- API Setup ---
def get_fred_api_key() -> Optional[str]:
    api_key = os.getenv("FRED_API_KEY")
    if not api_key:
        logging.error("FRED_API_KEY environment variable not set.")
        return None
    return api_key

def make_api_request(endpoint: str, params: Dict[str, Any], api_key: str) -> Optional[Dict]:
    if 'api_key' not in params:
        params['api_key'] = api_key
    if 'file_type' not in params:
        params['file_type'] = 'json'

    url = f"{FRED_API_BASE_URL}/{endpoint}"
    retries = 0
    while retries < API_MAX_RETRIES:
        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 429:
                logging.warning("Rate limit exceeded. Retrying...")
                time.sleep(API_RETRY_DELAY)
                retries += 1
                continue
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logging.warning(f"Request failed: {e}. Retrying {retries + 1}/{API_MAX_RETRIES}")
            time.sleep(API_RETRY_DELAY)
            retries += 1
    return None

# --- Category Traversal ---
def get_child_categories(api_key: str, category_id: int = 0) -> List[int]:
    endpoint = "category/children"
    params = {"category_id": category_id}
    data = make_api_request(endpoint, params, api_key)
    return [c['id'] for c in data.get("categories", []) if 'id' in c]

def get_series_by_frequency(api_key: str, category_id: int, frequency: str) -> List[str]:
    series_ids = []
    offset = 0
    while True:
        endpoint = "category/series"
        params = {
            "category_id": category_id,
            "limit": FRED_API_OFFSET_LIMIT,
            "offset": offset,
        }
        data = make_api_request(endpoint, params, api_key)
        if not data or "seriess" not in data:
            break

        filtered = [s['id'] for s in data["seriess"] if s.get("frequency_short") == frequency and 'id' in s]
        logging.debug(f"Category {category_id}, offset {offset}: Found {len(filtered)} {frequency} series.")
        series_ids.extend(filtered)
        if len(data["seriess"]) < FRED_API_OFFSET_LIMIT:
            break
        offset += FRED_API_OFFSET_LIMIT
        time.sleep(REQUEST_DELAY_SECONDS)
    return series_ids

def collect_series_recursive(api_key: str, category_id: int = 0, existing_ids: Optional[Set[str]] = None) -> Set[str]:
    if existing_ids is None:
        existing_ids = set()

    logging.info(f"Scanning Category ID {category_id}...")

    # Daily and Weekly series
    for freq in ('D', 'W'):
        ids = get_series_by_frequency(api_key, category_id, freq)
        if ids:
            new = set(ids) - existing_ids
            existing_ids.update(new)
            logging.info(f"  + Found {len(new)} new {freq} frequency series in category {category_id}.")

    # Recurse
    for child_id in get_child_categories(api_key, category_id):
        collect_series_recursive(api_key, child_id, existing_ids)
        time.sleep(REQUEST_DELAY_SECONDS)

    return existing_ids

# --- Script Entry ---
def main():
    api_key = get_fred_api_key()
    if not api_key:
        return

    logging.info("Starting recursive FRED series collection for daily and weekly frequencies...")
    series_ids = collect_series_recursive(api_key)

    logging.info(f"Total unique series collected: {len(series_ids)}")
    output_file = "fred_series_ids_daily_and_weekly.txt"
    with open(output_file, "w") as f:
        for sid in sorted(series_ids):
            f.write(f"{sid}\n")
    logging.info(f"Series IDs saved to {output_file}")

if __name__ == "__main__":
    main()
