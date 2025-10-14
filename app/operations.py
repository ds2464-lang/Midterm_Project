# Operations

from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Dict
from app.exceptions import ValidationError

class Operation(ABC):
    """
    Abstract base class for calculator operations.
    """
    @abstractmethod
    def execute(self, a: Decimal, b: Decimal) -> Decimal:
        """
        Execute the operation.
        """
        pass # pragma: no cover
    
    def validate_operands(self, a: Decimal, b: Decimal) -> None:
        """
        Validate operands before execution.
        """
        pass # pragma: no cover
    
    def __str__(self) -> str:
        """
        Return operation name for display
        """
        return self.__class__.__name__

class Addition(Operation):
    """
    Addition operation implementation.
    """
    
    def execute(self, a: Decimal, b: Decimal) -> Decimal:
        self.validate_operands(a, b)
        return a + b
    
class Subtraction(Operation):
    """
    Subtraction operation implementation.
    """

    def execute(self, a: Decimal, b: Decimal) -> Decimal:
        self.validate_operands(a, b)
        return a - b
    
class Multiplication(Operation):
    """
    Multiplication operation implementation.
    """

    def execute(self, a: Decimal, b: Decimal) -> Decimal:
        self.validate_operands(a, b)
        return a * b

class Division(Operation):
    """
    Division operation implementation.
    """
    
    def validate_operands(self, a: Decimal, b: Decimal) -> Decimal:
        super().validate_operands(a, b)
        if b == 0:
            raise ValidationError("Division by zero is not allowed")

    def execute(self, a: Decimal, b: Decimal) -> Decimal:
        self.validate_operands(a, b)
        return a / b

class Power(Operation):
    """
    Power operation implementation. Raises one number to the power of another.
    """

    def validate_operands(self, a: Decimal, b: Decimal) -> Decimal:
        super().validate_operands(a, b)
        if b < 0:
            raise ValidationError("Negative exponent not supported")

    def execute(self, a: Decimal, b: Decimal) -> Decimal:
        self.validate_operands(a, b)
        return Decimal(pow(float(a), float(b)))

class Root(Operation):
    """
    Root operation implementation. Calculates the nth root of a number.
    """

    def validate_operands(self, a: Decimal, b: Decimal) -> Decimal:
        super().validate_operands(a, b)
        if a < 0:
            raise ValidationError("Cannot calculate root of negative number")
        if b == 0:
            raise ValidationError("Zero root is undefined")

    def execute(self, a: Decimal, b: Decimal) -> Decimal:
        self.validate_operands(a, b)
        return Decimal(pow(float(a), 1 / float(b)))
    
class OperationFactory:
    """
    Factory class for creating operation instances.
    """
    # Dictionary mapping operation identifiers to their corresponding classes
    _operations: Dict[str, type] = {
        'add': Addition,
        'subtract': Subtraction,
        'multiply': Multiplication,
        'divide': Division,
        'power': Power,
        'root': Root,
    }

    @classmethod
    def register_operation(cls, name: str, operation_class: type) -> None:
        """
        Register a new operation type. Allows dynamic addition of new operations to the factory.
        """
        if not issubclass(operation_class, Operation):
            raise TypeError("Operation class must inherit from Operation.")
        cls.operations[name.lower()] = operation_class
    
    @classmethod
    def create_operation(cls, operation_type: str) -> Operation:
        """
        Creates an operation instance based on the operation type.
        """
        operation_class = cls._operations.get(operation_type())
        if not operation_class:
            raise ValueError(f"Unkown operation: {operation_type}")
        return operation_class()