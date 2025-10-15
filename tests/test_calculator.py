import datetime
from pathlib import Path
import pandas as pd
import pytest
from unittest.mock import Mock, patch, PropertyMock, MagicMock
from decimal import Decimal
from tempfile import TemporaryDirectory
from app.calculator import Calculator
from app.calculator_repl import calculator_repl
from app.calculator_config import CalculatorConfig
from app.exceptions import OperationError, ValidationError
from app.history import LoggingObserver, AutoSaveObserver
from app.operations import OperationFactory
from app.calculator_memento import CalculatorMemento
import logging
from app.calculation import Calculation

#Fixture to initialize Calculator with a temporary directory for file paths
@pytest
def calculator():
    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        config = CalculatorConfig(base_dir=temp_path)

        # Patch properties to use the temporary directory paths
        with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
             patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file, \
             patch.object(CalculatorConfig, 'history_dir', new_callable=PropertyMock) as mock_history_dir, \
             patch.object(CalculatorConfig, 'history_file', new_callable=PropertyMock) as mock_history_file:
            
            # Set return values to use paths within the temporary directory
            mock_log_dir.return_value = temp_path / "logs"
            mock_log_file.return_value = temp_path / "logs/calculator.log"
            mock_history_dir.return_value = temp_path / "history"
            mock_history_file.return_value = temp_path / "history/calculator_history.csv"
            
            # Return an instance of Calculator with the mocked config
            yield Calculator(config=config)

# Test Calculator Initialization

def test_calculator_initialization(calculator):
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []
    assert calculator.operation_strategy is None

# Test Logging Setup

@patch('app.calculator.logging.info')
def test_logging_setup(logging_info_mock):
    with patch.object(CalculatorConfig, 'log_dir', new_callable=PropertyMock) as mock_log_dir, \
         patch.object(CalculatorConfig, 'log_file', new_callable=PropertyMock) as mock_log_file:
        mock_log_dir.return_value = Path('/tmp/logs')
        mock_log_file.return_value = Path('/tmp/logs/calculator.log')
        
        # Instantiate calculator to trigger logging
        calculator = Calculator(CalculatorConfig())
        logging_info_mock.assert_any_call("Calculator initialized with configuration")

@patch('app.calculator.logging.info')
def test_setup_logging_exception(tmp_path):
    """
    Test that _setup_logging prints an error and re-raises the exception on failure.
    """
    config = CalculatorConfig(base_dir=tmp_path)
    calculator = Calculator(config=config)

    # Mock os.makedirs to raise an exception
    with patch('app.calculator.os.makedirs', side_effect=Exception("Permission denied")), \
         patch('builtins.print') as mock_print:
        
        with pytest.raises(Exception, match="Permission denied"):
            calculator._setup_logging()

        mock_print.assert_called_once_with("Error setting up logging: Permission denied")

# Test Adding and Removing Observers

def test_add_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    assert observer in calculator.observers

def test_remove_observer(calculator):
    observer = LoggingObserver()
    calculator.add_observer(observer)
    calculator.remove_observer(observer)
    assert observer not in calculator.observers

# Test Setting Operations

def test_set_operation(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    assert calculator.operation_strategy == operation

# Test Performing Operations

def test_perform_operation_addition(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    result = calculator.perform_operation(2, 3)
    assert result == Decimal('5')

def test_perform_operation_validation_error(calculator):
    calculator.set_operation(OperationFactory.create_operation('add'))
    with pytest.raises(ValidationError):
        calculator.perform_operation('invalid', 3)

def test_perform_operation_operation_error(calculator):
    with pytest.raises(OperationError, match="No operation set"):
        calculator.perform_operation(2, 3)

def test_perform_operation_trims_history(calculator):
    """
    Test that perform_operation trims history when it exceeds max_history_size.
    """
    # Arrange
    calculator.config.max_history_size = 3  # Small limit for testing
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)

    # Fill the history to the limit
    for i in range(3):
        calculator.perform_operation(i, i + 1)

    assert len(calculator.history) == 3  # confirm initial fill

    # Act — this next operation should push it over the limit
    calculator.perform_operation(10, 20)

    # Assert — history should still have only 3 items
    assert len(calculator.history) == 3

    # Assert — the *first* (oldest) entry was removed
    # The oldest was the result of (0 + 1)
    oldest_operation = calculator.history[0]
    assert oldest_operation.operand1 != Decimal('0')
    assert oldest_operation.operand2 != Decimal('1')

def test_perform_operation_unexpected_exception(calculator):
    """
    Test that perform_operation logs an error and raises OperationError
    when an unexpected exception occurs inside the operation.
    """
    # Arrange
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)

    # Patch the operation's execute() to raise an unexpected Exception
    with patch.object(operation, 'execute', side_effect=Exception("Unexpected math failure")), \
         patch('app.calculator.logging.error') as mock_log:
        
        # Act & Assert
        with pytest.raises(OperationError, match="Operation failed: Unexpected math failure"):
            calculator.perform_operation(Decimal('2'), Decimal('3'))

        # Verify that logging.error was called with the expected message
        mock_log.assert_called_once_with("Operation failed: Unexpected math failure")

# Test Undo/Redo Functionality

def test_undo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    assert calculator.history == []

def test_redo(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.undo()
    calculator.redo()
    assert len(calculator.history) == 1

# Test History Management

@patch('app.calculator.Calculator.load_history', side_effect=Exception("File read error"))
@patch('app.calculator.logging.warning')
def test_calculator_init_load_history_warning(mock_warning, mock_load_history):
    """
    Test that Calculator initialization logs a warning if load_history() fails.
    """
    calc = Calculator(CalculatorConfig())  # triggers the try/except internally

    mock_warning.assert_called_with("Could not load existing history: File read error")



@patch('app.calculator.pd.DataFrame.to_csv')
def test_save_history(mock_to_csv, calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.save_history()
    mock_to_csv.assert_called_once()


def test_save_history_empty_logs_and_creates_empty_csv(tmp_path, calculator):
    """
    Test that save_history() creates an empty CSV and logs when history is empty.
    """
    # Arrange
    calculator.history = []  # simulate empty history
    calculator.config.history_file = tmp_path / "history.csv"

    with patch("app.calculator.logging.info") as mock_log, \
         patch("pandas.DataFrame.to_csv") as mock_to_csv:

        # Act
        calculator.save_history()

        # Assert
        mock_to_csv.assert_called_once_with(calculator.config.history_file, index=False)
        mock_log.assert_called_once_with("Empty history saved")


@patch('app.calculator.pd.read_csv')
@patch('app.calculator.Path.exists', return_value=True)
def test_load_history(mock_exists, mock_read_csv, calculator):
    # Mock CSV data to match the expected format in from_dict
    mock_read_csv.return_value = pd.DataFrame({
        'operation': ['Addition'],
        'operand1': ['2'],
        'operand2': ['3'],
        'result': ['5'],
        'timestamp': [datetime.datetime.now().isoformat()]
    })
    
    # Test the load_history functionality
    try:
        calculator.load_history()
        # Verify history length after loading
        assert len(calculator.history) == 1
        # Verify the loaded values
        assert calculator.history[0].operation == "Addition"
        assert calculator.history[0].operand1 == Decimal("2")
        assert calculator.history[0].operand2 == Decimal("3")
        assert calculator.history[0].result == Decimal("5")
    except OperationError:
        pytest.fail("Loading history failed due to OperationError")
        
            
# Test Clearing History

def test_clear_history(calculator):
    operation = OperationFactory.create_operation('add')
    calculator.set_operation(operation)
    calculator.perform_operation(2, 3)
    calculator.clear_history()
    assert calculator.history == []
    assert calculator.undo_stack == []
    assert calculator.redo_stack == []

# Test REPL Commands (using patches for input/output handling)

@patch('builtins.input', side_effect=['exit'])
@patch('builtins.print')
def test_calculator_repl_exit(mock_print, mock_input):
    with patch('app.calculator.Calculator.save_history') as mock_save_history:
        calculator_repl()
        mock_save_history.assert_called_once()
        mock_print.assert_any_call("History saved successfully.")
        mock_print.assert_any_call("Goodbye!")

@patch('builtins.input', side_effect=['help', 'exit'])
@patch('builtins.print')
def test_calculator_repl_help(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nAvailable commands:")

@patch('builtins.input', side_effect=['add', '2', '3', 'exit'])
@patch('builtins.print')
def test_calculator_repl_addition(mock_print, mock_input):
    calculator_repl()
    mock_print.assert_any_call("\nResult: 5")


def test_to_dict_returns_expected_structure():
    """
    Test that CalculatorMemento.to_dict() correctly serializes history and timestamp.
    """
    # Arrange: create mock Calculation objects with known to_dict() return values
    calc1 = MagicMock()
    calc1.to_dict.return_value = {'operation': 'add', 'operand1': 2, 'operand2': 3, 'result': 5}
    
    calc2 = MagicMock()
    calc2.to_dict.return_value = {'operation': 'multiply', 'operand1': 4, 'operand2': 5, 'result': 20}
    
    # Fixed timestamp for test predictability
    test_timestamp = datetime.datetime(2025, 10, 6, 12, 0, 0)

    memento = CalculatorMemento(history=[calc1, calc2])
    memento.timestamp = test_timestamp  # Override timestamp for stable comparison

    # Act
    result = memento.to_dict()

    # Assert
    expected = {
        'history': [
            {'operation': 'add', 'operand1': 2, 'operand2': 3, 'result': 5},
            {'operation': 'multiply', 'operand1': 4, 'operand2': 5, 'result': 20},
        ],
        'timestamp': test_timestamp.isoformat(),
    }

    assert result == expected
    calc1.to_dict.assert_called_once()
    calc2.to_dict.assert_called_once()


def test_from_dict_restores_memento_correctly():
    """
    Test that CalculatorMemento.from_dict correctly restores history and timestamp.
    """
    # Arrange
    fake_data = {
        "history": [
            {"operation": "add", "operand1": 2, "operand2": 3, "result": 5},
            {"operation": "multiply", "operand1": 4, "operand2": 5, "result": 20},
        ],
        "timestamp": "2025-10-06T12:00:00"
    }

    mock_calc1 = MagicMock()
    mock_calc2 = MagicMock()

    # Patch Calculation.from_dict to return mock Calculation objects
    with patch("app.calculator_memento.Calculation.from_dict", side_effect=[mock_calc1, mock_calc2]) as mock_from_dict:
        # Act
        memento = CalculatorMemento.from_dict(fake_data)

        # Assert
        assert isinstance(memento, CalculatorMemento)
        assert memento.history == [mock_calc1, mock_calc2]
        assert memento.timestamp == datetime.datetime(2025, 10, 6, 12, 0, 0)

        # Verify from_dict was called correctly for each history item
        assert mock_from_dict.call_count == 2
        mock_from_dict.assert_any_call(fake_data["history"][0])
        mock_from_dict.assert_any_call(fake_data["history"][1])


@patch("builtins.print")
@patch("builtins.input", side_effect=["history", "exit"])
def test_repl_history_no_entries(mock_input, mock_print):
    """Test that REPL prints message when there is no calculation history."""
    with patch("app.calculator.Calculator.show_history", return_value=[]), \
         patch("app.calculator.Calculator.save_history"):
        calculator_repl()

    # Verify correct messages are printed
    mock_print.assert_any_call("No calculations in history")
    mock_print.assert_any_call("Goodbye!")

@patch("builtins.print")
@patch("builtins.input", side_effect=["history", "exit"])
def test_repl_prints_history_entries(mock_input, mock_print):
    """Ensure that non-empty history triggers the loop that prints each entry."""
    # Create a fake Calculator with a populated history
    fake_calc = MagicMock()
    fake_calc.show_history.return_value = [
        "Addition(2, 3) = 5",
        "Multiplication(4, 2) = 8"
    ]
    fake_calc.save_history.return_value = None

    # Patch the Calculator class used inside calculator_repl
    with patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()

    # Verify that the history header and entries were printed
    mock_print.assert_any_call("\nCalculation History:")
    mock_print.assert_any_call("1. Addition(2, 3) = 5")
    mock_print.assert_any_call("2. Multiplication(4, 2) = 8")

@patch("builtins.print")
@patch("builtins.input", side_effect=["clear", "exit"])
def test_repl_clear_history_command(mock_input, mock_print):
    """Ensure the 'clear' command clears history and prints confirmation."""
    # Create a fake Calculator with a clear_history method
    fake_calc = MagicMock()
    fake_calc.clear_history.return_value = None
    fake_calc.save_history.return_value = None

    # Patch Calculator class used in the REPL
    with patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()

    # Verify that clear_history was called once
    fake_calc.clear_history.assert_called_once()

    # Verify that confirmation message was printed
    mock_print.assert_any_call("History cleared")

    # Verify exit message is printed after loop ends
    mock_print.assert_any_call("Goodbye!")

@pytest.mark.parametrize(
    "undo_result, expected_message",
    [
        (True, "Operation undone"),
        (False, "Nothing to undo"),
    ],
)
@patch("builtins.print")
def test_repl_undo_command(mock_print, undo_result, expected_message):
    """Test 'undo' command prints correct message depending on undo result."""
    # Create a fake Calculator with undo() returning True or False
    fake_calc = MagicMock()
    fake_calc.undo.return_value = undo_result
    fake_calc.save_history.return_value = None

    # Simulate 'undo' then 'exit' commands
    with patch("builtins.input", side_effect=["undo", "exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()

    # Ensure undo() was called once
    fake_calc.undo.assert_called_once()

    # Check that correct message was printed
    mock_print.assert_any_call(expected_message)

    # Verify goodbye message prints on exit
    mock_print.assert_any_call("Goodbye!")

@pytest.mark.parametrize(
    "redo_result, expected_message",
    [
        (True, "Operation redone"),
        (False, "Nothing to redo"),
    ],
)    
@patch("builtins.print")
def test_repl_redo_command(mock_print, redo_result, expected_message):
    """Test 'redo' command prints correct message depending on redo result."""
    # Create a fake Calculator with undo() returning True or False
    fake_calc = MagicMock()
    fake_calc.redo.return_value = redo_result
    fake_calc.save_history.return_value = None

    # Simulate 'undo' then 'exit' commands
    with patch("builtins.input", side_effect=["redo", "exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()

    fake_calc.redo.assert_called_once()
    mock_print.assert_any_call(expected_message)
    mock_print.assert_any_call("Goodbye!")

@patch("builtins.print")
def test_repl_cancel_first_number(mock_print):
    """Covers the branch where user cancels input for 'a'."""
    fake_calc = MagicMock()

    # User types 'add', then 'cancel' for first number, then exit
    with patch("builtins.input", side_effect=["add", "cancel", "exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()

    # Confirm that the cancel message was printed
    mock_print.assert_any_call("Operation cancelled")


@patch("builtins.print")
def test_repl_cancel_second_number(mock_print):
    """Covers the branch where user cancels input for 'b'."""
    fake_calc = MagicMock()

    with patch("builtins.input", side_effect=["add", "2", "cancel", "exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()

    mock_print.assert_any_call("Operation cancelled")


@patch("builtins.print")
def test_repl_known_exceptions(mock_print):
    """Trigger ValidationError during operation."""
    fake_calc = MagicMock()
    fake_calc.perform_operation.side_effect = ValidationError("Invalid input")
    fake_calc.set_operation.return_value = None

    with patch("builtins.input", side_effect=["add", "2", "3", "exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc), \
         patch("app.calculator_repl.OperationFactory.create_operation", return_value=MagicMock()):
        calculator_repl()

    mock_print.assert_any_call("Error: Invalid input")

@patch("builtins.print")
def test_repl_unexpected_exception(mock_print):
    """Trigger unexpected exception during operation."""
    fake_calc = MagicMock()
    fake_calc.perform_operation.side_effect = Exception("error")
    fake_calc.set_operation.return_value = None

    with patch("builtins.input", side_effect=["add", "2", "3", "exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc), \
         patch("app.calculator_repl.OperationFactory.create_operation", return_value=MagicMock()):
        calculator_repl()

    mock_print.assert_any_call("Unexpected error: error")

@patch("builtins.print")
def test_repl_unknown_command(mock_print):
    fake_calc = MagicMock()

    with patch("builtins.input", side_effect=["modulus", "exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()

    mock_print.assert_any_call("Unknown command: 'modulus'. Type 'help' for available commands.")




@patch("builtins.print")
def test_repl_eof_error(mock_print):
    fake_calc = MagicMock()

    with patch("builtins.input", side_effect=EOFError), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()

    mock_print.assert_any_call("\nInput terminated. Exiting...")

@patch("builtins.print")
@patch("logging.error")
def test_repl_fatal_error(mock_logging_error, mock_print):
    # Patch Calculator to raise an exception during initialization
    with patch("app.calculator_repl.Calculator", side_effect=Exception("Init fail")):
        with pytest.raises(Exception, match="Init fail"):
            calculator_repl()

    mock_print.assert_any_call("Fatal error: Init fail")
    mock_logging_error.assert_called_with("Fatal error in calculator REPL: Init fail")


@patch("builtins.print")
def test_repl_exit_save_history_exception(mock_print):
    """Covers the 'Warning: Could not save history' and 'Goodbye!' lines."""
    fake_calc = MagicMock()
    # Make save_history raise an Exception
    fake_calc.save_history.side_effect = Exception("Disk write error")

    # Patch Calculator to return our fake_calc
    with patch("builtins.input", side_effect=["exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()

    # Check that the warning for failed save was printed
    mock_print.assert_any_call("Warning: Could not save history: Disk write error")
    # Check that "Goodbye!" is always printed
    mock_print.assert_any_call("Goodbye!")

@patch("builtins.print")
def test_repl_save_history_success(mock_print):
    fake_calc = MagicMock()
    
    # save_history works normally (does not raise)
    with patch("builtins.input", side_effect=["save", "exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()
    
    # Confirm success message
    mock_print.assert_any_call("History saved successfully")

@patch("builtins.print")
def test_repl_save_history_exception(mock_print):
    fake_calc = MagicMock()
    # Force save_history to raise
    fake_calc.save_history.side_effect = Exception("Disk write error")
    
    with patch("builtins.input", side_effect=["save", "exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()
    
    # Confirm that exception handler prints error message
    mock_print.assert_any_call("Error saving history: Disk write error")

@ patch("builtins.print")
def test_repl_load_history_success(mock_print):
    fake_calc = MagicMock()

    with patch("builtins.input", side_effect=["load", "exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()

    mock_print.assert_any_call("History loaded successfully")

@patch("builtins.print")
def test_repl_load_history_exception(mock_print):
    fake_calc = MagicMock()
    fake_calc.load_history.side_effect = Exception("Disk read error")

    with patch("builtins.input", side_effect=["load", "exit"]), \
         patch("app.calculator_repl.Calculator", return_value=fake_calc):
        calculator_repl()
    
    mock_print.assert_any_call("Error loading history: Disk read error")


def test_save_history_exception_block(calculator):
    """Test that the except block in save_history logs error and raises OperationError."""
    # Arrange: add a calculation to history
    calc_entry = Calculation(operation="Addition", operand1=Decimal("2"), operand2=Decimal("3"))
    calculator.history.append(calc_entry)

    # Patch DataFrame.to_csv to raise an exception
    with patch("pandas.DataFrame.to_csv", side_effect=Exception("Disk write error")), \
         patch("logging.error") as mock_log_error:
        # Act & Assert: OperationError is raised with correct message
        with pytest.raises(OperationError, match="Failed to save history: Disk write error"):
            calculator.save_history()

        # Assert: logging.error was called with the exception message
        mock_log_error.assert_called_with("Failed to save history: Disk write error")


def test_get_history_dataframe_with_data(calculator):
    """Test that get_history_dataframe returns a valid DataFrame when history is populated."""
    # Arrange: create fake Calculation objects
    calc1 = Calculation(operation="Addition", operand1=1, operand2=2)
    calc1.result = 3
    calc2 = Calculation(operation="Multiplication", operand1=2, operand2=3)
    calc2.result = 6

    calculator.history = [calc1, calc2]

    # Act
    df = calculator.get_history_dataframe()

    # Assert: ensure DataFrame structure and content
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ['operation', 'operand1', 'operand2', 'result', 'timestamp']
    assert len(df) == 2
    assert df.iloc[0]['operation'] == 'Addition'
    assert df.iloc[1]['result'] == '6'

def test_show_history_return_format(calculator):
    """
    Test that show_history returns a correctly formatted list of strings.
    This specifically targets the list comprehension return line.
    """
    # Arrange: create a single Calculation instance with known values
    calc = Calculation(operation="Addition", operand1=2, operand2=3)
    calc.result = 5

    calculator.history = [calc]

    # Act: call the method (this executes the list comprehension)
    result = calculator.show_history()

    # Assert: verify the returned list and its string format
    assert isinstance(result, list), "Expected a list return type"
    assert len(result) == 1, "Expected one history entry"
    assert result[0] == "Addition(2, 3) = 5", "Output string format is incorrect"

def test_undo_returns_false_when_undo_stack_empty(calculator):
    """
    Test that undo() returns False when there is nothing to undo.
    """
    # Arrange: ensure undo_stack is empty
    calculator.undo_stack.clear()

    # Act: call undo
    result = calculator.undo()

    # Assert: verify early return behavior
    assert result is False, "Expected undo() to return False when undo_stack is empty"

def test_redo_returns_false_when_redo_stack_empty(calculator):
    calculator.redo_stack.clear()
    result = calculator.redo()
    assert result is False, "Expected redo() to return False when redo_stack is empty"