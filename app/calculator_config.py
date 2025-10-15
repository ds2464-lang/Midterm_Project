# Calculator Config #

from dataclasses import dataclass
from decimal import Decimal
from numbers import Number
from pathlib import Path
import os
from typing import Optional

from dotenv import load_dotenv

from app.exceptions import ConfigurationError

# Load environment variables from a .env file into the program's environment
load_dotenv()

def get_project_root() -> Path:
    # Get the directory of the current file (app/calculator_config.py)
    current_file = Path(__file__)
    # Navigate up two levels to reach the project root (from app/calculator_config.py to project root)
    return current_file.parent.parent

@dataclass
class CalculatorConfig:
    def __init__(
        self,
        base_dir: Optional[Path] = None,
        max_history_size: Optional[int] = None,
        auto_save: Optional[bool] = None,
        precision: Optional[int] = None,
        max_input_value: Optional[Number] = None,
        default_encoding: Optional[str] = None
    ):
        # Set base directory to project root by default
        project_root = get_project_root()
        self.base_dir = base_dir or Path(
            os.getenv('CALCULATOR_BASE_DIR', str(project_root))
        ).resolve()
        # Maximum history size
        self.max_history_size = max_history_size or int(
            os.getenv('CALCULATOR_MAX_HISTORY_SIZE', '1000')
        )
        # Auto-save preference
        auto_save_env = os.getenv('CALCULATOR_AUTO_SAVE', 'true').lower()
        self.auto_save = auto_save if auto_save is not None else (
            auto_save_env == 'true' or auto_save_env == '1'
        )
        # Calculation precision
        self.precision = precision or int(
            os.getenv('CALCULATOR_PRECISION', '10')
        )
        # Maximum input value allowed
        self.max_input_value = max_input_value or Decimal(
            os.getenv('CALCULATOR_MAX_INPUT_VALUE', '1e999')
        )
        # Default encoding for file operations
        self.default_encoding = default_encoding or os.getenv(
            'Calculator_Default_Encoding', 'utf - 8'
        )
    @property
    def log_dir(self) -> Path:
        return Path(os.getenv(
            'CALCULATOR_LOG_DIR',
            str(self.base_dir / "logs")
        )).resolve()
    
    @property
    def history_dir(self) -> Path:
        return Path(os.getenv(
            'CALCULATOR_HISTORY_DIR',
            str(self.base_dir / "history")
        )).resolve()
    
    @property
    def history_files(self) -> Path:
        return Path(os.getenv(
            'CALCULATOR_HISTORY_FILE',
            str(self.history_dir / "calculator_history.csv")
        )).resolve()
    
    @property
    def log_file(self) -> Path:
        return Path(os.getenv(
            'CALCULATOR_LOG_FILE',
            str(self.log_dir / "calculator.log")
        )).resolve()
    
    def validate(self) -> None:
        if self.max_history_size <=0:
            raise ConfigurationError("max_history_size must be positive")
        if self.precision <= 0:
            raise ConfigurationError("precision must be positive")
        if self.max_input_value <= 0:
            raise ConfigurationError("max_input_value must be positive")