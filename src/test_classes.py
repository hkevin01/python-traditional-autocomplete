"""
Test File 2: Class AutoComplete Testing

INSTRUCTIONS:
- Type 'self.' inside methods to see attribute suggestions
- Type class name + '.' to see class methods
- Hover over method names to see docstrings
- NO Copilot ghost text should appear

Expected: Only traditional IntelliSense suggestions
"""

from typing import List, Optional


class Calculator:
    """A simple calculator class for testing autocomplete."""
    
    def __init__(self, initial_value: float = 0.0):
        """Initialize calculator with optional starting value."""
        self.value = initial_value
        self.history: List[str] = []
        self.operations_count = 0
    
    def add(self, number: float) -> float:
        """
        Add a number to current value.
        
        TEST: When you type 'self.' inside this method, you should see:
        - self.value
        - self.history
        - self.operations_count
        - self.add, self.subtract, etc.
        """
        # Type 'self.' here and see suggestions
        self.value += number
        self.history.append(f"Added {number}")
        self.operations_count += 1
        return self.value
    
    def subtract(self, number: float) -> float:
        """Subtract a number from current value."""
        # Type 'self.' here
        self.value -= number
        self.history.append(f"Subtracted {number}")
        self.operations_count += 1
        return self.value
    
    def multiply(self, number: float) -> float:
        """Multiply current value by number."""
        # Type 'self.h' and see 'history' suggestion
        self.value *= number
        self.history.append(f"Multiplied by {number}")
        return self.value
    
    def get_history(self) -> List[str]:
        """Get operation history."""
        # Type 'self.hi' to see history autocomplete
        return self.history
    
    def reset(self) -> None:
        """Reset calculator to zero."""
        self.value = 0.0
        self.history.clear()
        self.operations_count = 0


class DataAnalyzer:
    """Test class with more complex type hints."""
    
    def __init__(self, data: List[float]):
        self.data = data
        self.stats: Optional[dict] = None
    
    def calculate_mean(self) -> float:
        """Calculate mean of data."""
        # TEST: Type 'self.d' and see 'data' suggestion
        # Type 'self.data.' to see list methods
        return sum(self.data) / len(self.data) if self.data else 0.0
    
    def calculate_median(self) -> float:
        """Calculate median of data."""
        sorted_data = sorted(self.data)
        n = len(sorted_data)
        # TEST: Type 'sorted_data.' to see list methods
        if n % 2 == 0:
            return (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        return sorted_data[n//2]


# TEST: Instance methods
# Type 'calc.' and you should see all Calculator methods
calc = Calculator(100)
# calc.

# Type 'analyzer.' to see DataAnalyzer methods
analyzer = DataAnalyzer([1, 2, 3, 4, 5])
# analyzer.

# TEST: Call method and see parameter hints
# When you type 'calc.add(' you should see parameter hint: number: float
result = calc.add(50)

print("✅ Test autocomplete by typing 'calc.' or 'analyzer.' on new lines")
print("✅ If NO ghost text appears, Copilot is successfully disabled!")
