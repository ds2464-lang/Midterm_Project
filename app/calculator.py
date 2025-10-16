# Calculator Class #

from decimal import Decimal
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from app.calculation import Calculation
from app.calculator_config import CalculatorConfig
from app.calculator_memento import CalculatorMemento
from app.exceptions import OperationError, ValidationError
from app.history import HistoryObserver
from app.input_validators import InputValidator
from app.operations import Operation

# Type aliases for better readability
Number = Union[int, float, Decimal]
CalculationResult = Union[Number, str]


class Calculator:

    def __init__(self, config: Optional[CalculatorConfig] = None):
        if config is None:
            # Determine the project root directory if no configuration is provided
            current_file = Path(__file__)
            project_root = current_file.parent.parent
            config = CalculatorConfig(base_dir=project_root)

        # Assign the configuration and validate its parameters
        self.config = config
        self.config.validate()

        # Ensure that the log directory exists
        os.makedirs(self.config.log_dir, exist_ok=True)

        # Set up the logging system
        self._setup_logging()

        # Initialize calculation history and operation strategy
        self.history: List[Calculation] = []
        self.operation_strategy: Optional[Operation] = None

        # Initialize observer list for the Observer pattern
        self.observers: List[HistoryObserver] = []

        # Initialize stacks for undo and redo functionality using the Memento pattern
        self.undo_stack: List[CalculatorMemento] = []
        self.redo_stack: List[CalculatorMemento] = []

        # Create required directories for history management
        self._setup_directories()

        try:
            # Attempt to load existing calculation history from file
            self.load_history()
        except Exception as e:
            # Log a warning if history could not be loaded
            logging.warning(f"Could not load existing history: {e}")

        # Log the successful initialization of the calculator
        logging.info("Calculator initialized with configuration")

    def _setup_logging(self) -> None:
        try:
            # Ensure the log directory exists
            os.makedirs(self.config.log_dir, exist_ok=True)
            log_file = self.config.log_file.resolve()

            # Configure the basic logging settings
            logging.basicConfig(
                filename=str(log_file),
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                force=True  # Overwrite any existing logging configuration
            )
            logging.info(f"Logging initialized at: {log_file}")
        except Exception as e:
            # Print an error message and re-raise the exception if logging setup fails
            print(f"Error setting up logging: {e}")
            raise

    def _setup_directories(self) -> None:
        self.config.history_dir.mkdir(parents=True, exist_ok=True)

    def add_observer(self, observer: HistoryObserver) -> None:
        self.observers.append(observer)
        logging.info(f"Added observer: {observer.__class__.__name__}")

    def remove_observer(self, observer: HistoryObserver) -> None:
        self.observers.remove(observer)
        logging.info(f"Removed observer: {observer.__class__.__name__}")

    def notify_observers(self, calculation: Calculation) -> None:
        for observer in self.observers:
            observer.update(calculation)

    def set_operation(self, operation: Operation) -> None:
        self.operation_strategy = operation
        logging.info(f"Set operation: {operation}")

    def perform_operation(
        self,
        a: Union[str, Number],
        b: Union[str, Number]
    ) -> CalculationResult:
        if not self.operation_strategy:
            raise OperationError("No operation set")

        try:
            # Validate and convert inputs to Decimal
            validated_a = InputValidator.validate_number(a, self.config)
            validated_b = InputValidator.validate_number(b, self.config)

            # Execute the operation strategy
            result = self.operation_strategy.execute(validated_a, validated_b)

            # Create a new Calculation instance with the operation details
            calculation = Calculation(
                operation=str(self.operation_strategy),
                operand1=validated_a,
                operand2=validated_b
            )

            # Save the current state to the undo stack before making changes
            self.undo_stack.append(CalculatorMemento(self.history.copy()))

            # Clear the redo stack since new operation invalidates the redo history
            self.redo_stack.clear()

            # Append the new calculation to the history
            self.history.append(calculation)

            # Ensure the history does not exceed the maximum size
            if len(self.history) > self.config.max_history_size:
                self.history.pop(0)

            # Notify all observers about the new calculation
            self.notify_observers(calculation)

            return result

        except ValidationError as e:
            # Log and re-raise validation errors
            logging.error(f"Validation error: {str(e)}")
            raise
        except Exception as e:
            # Log and raise operation errors for any other exceptions
            logging.error(f"Operation failed: {str(e)}")
            raise OperationError(f"Operation failed: {str(e)}")

    def save_history(self) -> None:
        try:
            # Ensure the history directory exists
            self.config.history_dir.mkdir(parents=True, exist_ok=True)

            history_data = []
            for calc in self.history:
                # Serialize each Calculation instance to a dictionary
                history_data.append({
                    'operation': str(calc.operation),
                    'operand1': str(calc.operand1),
                    'operand2': str(calc.operand2),
                    'result': str(calc.result),
                    'timestamp': calc.timestamp.isoformat()
                })

            if history_data:
                # Create a pandas DataFrame from the history data
                df = pd.DataFrame(history_data)
                # Write the DataFrame to a CSV file without the index
                df.to_csv(self.config.history_file, index=False)
                logging.info(f"History saved successfully to {self.config.history_file}")
            else:
                # If history is empty, create an empty CSV with headers
                pd.DataFrame(columns=['operation', 'operand1', 'operand2', 'result', 'timestamp']
                           ).to_csv(self.config.history_file, index=False)
                logging.info("Empty history saved")

        except Exception as e:
            # Log and raise an OperationError if saving fails
            logging.error(f"Failed to save history: {e}")
            raise OperationError(f"Failed to save history: {e}")

    def load_history(self) -> None:
        try:
            if self.config.history_file.exists():
                # Read the CSV file into a pandas DataFrame
                df = pd.read_csv(self.config.history_file)
                if not df.empty:
                    # Deserialize each row into a Calculation instance
                    self.history = [
                        Calculation.from_dict({
                            'operation': row['operation'],
                            'operand1': row['operand1'],
                            'operand2': row['operand2'],
                            'result': row['result'],
                            'timestamp': row['timestamp']
                        })
                        for _, row in df.iterrows()
                    ]
                    logging.info(f"Loaded {len(self.history)} calculations from history")
                else:
                    logging.info("Loaded empty history file")
            else:
                # If no history file exists, start with an empty history
                logging.info("No history file found - starting with empty history")
        except Exception as e:
            # Log and raise an OperationError if loading fails
            logging.error(f"Failed to load history: {e}")
            raise OperationError(f"Failed to load history: {e}")

    def get_history_dataframe(self) -> pd.DataFrame:
        history_data = []
        for calc in self.history:
            history_data.append({
                'operation': str(calc.operation),
                'operand1': str(calc.operand1),
                'operand2': str(calc.operand2),
                'result': str(calc.result),
                'timestamp': calc.timestamp
            })
        return pd.DataFrame(history_data)

    def show_history(self) -> List[str]:
        return [
            f"{calc.operation}({calc.operand1}, {calc.operand2}) = {calc.result}"
            for calc in self.history
        ]

    def clear_history(self) -> None:
        self.history.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()
        logging.info("History cleared")

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        # Pop the last state from the undo stack
        memento = self.undo_stack.pop()
        # Push the current state onto the redo stack
        self.redo_stack.append(CalculatorMemento(self.history.copy()))
        # Restore the history from the memento
        self.history = memento.history.copy()
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        # Pop the last state from the redo stack
        memento = self.redo_stack.pop()
        # Push the current state onto the undo stack
        self.undo_stack.append(CalculatorMemento(self.history.copy()))
        # Restore the history from the memento
        self.history = memento.history.copy()
        return True