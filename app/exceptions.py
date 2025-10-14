# Execeptions #

class CalculatorError(Exception):
    """
    Base exception class for calculator-specific error.
    """
    pass

class ValidationError(CalculatorError):
    """
    Raised when input validation fails.
    """
    pass

class OperationError(CalculatorError):
    """
    Raised when a calculation operation fails.
    """
    pass

class ConfigurationError(CalculatorError):
    """
    Raised when calculator configuration is invalid.
    """
    pass