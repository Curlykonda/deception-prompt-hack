import logging

log_format = "%(asctime)s | %(levelname)s | %(message)s"
date_format = "%d/%m/%y-%H:%M"
logging.basicConfig(level=logging.INFO, format=log_format, datefmt=date_format)
LOG = logging.getLogger(__name__)
