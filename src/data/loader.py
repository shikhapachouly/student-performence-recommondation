import logging
import pandas as pd
from src.config import DATASET_PATH

logger = logging.getLogger(__name__)

EXPECTED_COLS = 52
EXPECTED_ROWS = 1378


def load_dataset(path=None):
    """Load the student dataset CSV, strip column whitespace, and validate shape."""
    if path is None:
        path = DATASET_PATH

    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    logger.info("Dataset shape: %d rows x %d columns", df.shape[0], df.shape[1])
    logger.info("Columns: %s", list(df.columns))

    # Validate shape
    if df.shape[0] != EXPECTED_ROWS:
        logger.warning(
            "Expected %d rows but got %d", EXPECTED_ROWS, df.shape[0]
        )
    if df.shape[1] != EXPECTED_COLS:
        logger.warning(
            "Expected %d columns but got %d", EXPECTED_COLS, df.shape[1]
        )

    return df
