# Calculator Memento #

from dataclasses import dataclass, field
import datetime
from typing import Any, Dict, List
from app.calculation import Calculation


@dataclass
class CalculatorMemento:
    history: List[Calculation] # List of Calculation instances representing the calcutor's history
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now) # Time when the memento was created

    def to_dict(self) -> Dict[str, Any]:
        return {
            'history' : [calc.to_dict() for calc in self.history],
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CalculatorMemento':
        return cls(
            history=[Calculation.from_dict(calc) for calc in data['history']],
            timestamp=datetime.datetime.fromisoformat(data['timestamp'])
        )