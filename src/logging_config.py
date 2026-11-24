
import logging
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Logs directory
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = LOG_DIR / "predictions.log"

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

# Create logger
logger = logging.getLogger("BankMarketingPrediction")


def log_prediction(customer_data, prediction, probabilities):

    result = "Yes" if prediction == 1 else "No"

    logger.info(
        "Customer Data: %s | Prediction: %s | "
        "No Subscription Probability: %.4f | "
        "Subscription Probability: %.4f",
        customer_data,
        result,
        probabilities[0],
        probabilities[1]
    )
