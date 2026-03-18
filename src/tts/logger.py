import os
import logging
from logging.handlers import TimedRotatingFileHandler
from datetime import datetime
from typing import Dict, Any

class Logger():
    def __init__(
            self,
            log_dir: str = "logs",
            print_to_console: bool = True
        ):
        super().__init__()
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.log_file_name = os.path.join(log_dir, f"log_{current_date}.log")

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.logger = logging.getLogger("TSSLogger")
        self.logger.setLevel(logging.DEBUG)

        if self.logger.handlers:
            self.logger.handlers.clear()

        file_handler = TimedRotatingFileHandler(
            self.log_file_name, when="midnight", interval=1, backupCount=7, encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        if print_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        return self.logger
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        return {
            "print_to_console": False,
            "log_dir": "tts_logs"
        }