"""
Single entry point for loading churn data. All scripts should call load_data() from here.
Switch between BigQuery (SQL) and local CSV/Excel (pandas) via config.DATA_SOURCE.
"""
from config import DATA_SOURCE
import main_data_loading
import backup_data_loading


def load_data():
    """Load train/val/test splits. Uses BigQuery if DATA_SOURCE == 'bigquery', else pandas (local CSV/Excel)."""
    if DATA_SOURCE == "bigquery":
        return main_data_loading.load_data()
    return backup_data_loading.load_data()
