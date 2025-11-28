import logging
import os

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")


class ProjectFilter(logging.Filter):
    def filter(self, record):
        return record.name.startswith("myproject")


file_handler = logging.FileHandler(LOG_FILE)
file_handler.addFilter(ProjectFilter())

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[file_handler],
)

logger = logging.getLogger(__name__)
