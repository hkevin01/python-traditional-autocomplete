"""
Example Python file to test traditional auto-complete features.

This file demonstrates various Python features that should trigger
intelligent auto-completion without AI assistance.
"""

import os
import sys
from typing import List, Dict, Optional


class DataProcessor:
    """Example class for data processing."""
    
    def __init__(self, name: str, data: List[int]):
        """
        Initialize the data processor.
        
        Args:
            name: The name of the processor
            data: List of integers to process
        """
        self.name = name
        self.data = data
        self.results: Dict[str, float] = {}
    
    def calculate_statistics(self) -> Dict[str, float]:
        """
        Calculate basic statistics on the data.
        
        Returns:
            Dictionary containing mean, median, and sum
        """
        if not self.data:
            return {"mean": 0.0, "median": 0.0, "sum": 0.0}
        
        sorted_data = sorted(self.data)
        n = len(sorted_data)
        
        # Calculate mean
        mean = sum(sorted_data) / n
        
        # Calculate median
        if n % 2 == 0:
            median = (sorted_data[n//2 - 1] + sorted_data[n//2]) / 2
        else:
            median = sorted_data[n//2]
        
        self.results = {
            "mean": mean,
            "median": median,
            "sum": sum(sorted_data)
        }
        
        return self.results
    
    def process_file(self, filepath: str) -> Optional[List[str]]:
        """
        Read and process a file.
        
        Args:
            filepath: Path to the file to process
            
        Returns:
            List of lines from the file, or None if file not found
        """
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            return [line.strip() for line in lines]
        except FileNotFoundError:
            print(f"File not found: {filepath}")
            return None


def main():
    """Main function demonstrating traditional auto-complete features."""
    # When typing this, you should see:
    # - Class suggestions
    # - Method parameter hints
    # - Type information
    # - Import completions
    
    processor = DataProcessor("test", [1, 2, 3, 4, 5])
    stats = processor.calculate_statistics()
    
    print(f"Statistics: {stats}")
    
    # Path autocomplete should work here
    result = processor.process_file("example.txt")
    
    # Auto-complete should suggest os.path methods
    current_dir = os.path.dirname(__file__)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
