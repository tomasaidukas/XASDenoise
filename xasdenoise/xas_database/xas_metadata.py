"""
Simplified XAS metadata system using dataclass dynamic attributes.

This design leverages Python's dataclass ability to accept dynamic attributes,
providing a much simpler and more maintainable solution.
"""

from dataclasses import dataclass, fields, asdict
from typing import Optional, Dict, Any, Union, List
import json
import csv
from pathlib import Path
from webbrowser import get

@dataclass
class XASMetadata:
    """
    XAS metadata dataclass with dynamic attribute support.
    
    Predefined fields cover the most common XAS parameters, but any additional
    attributes can be added dynamically at runtime.
    """
    
    # === Core Sample Information ===
    element: str = "" 
    compound: Optional[str] = None # chemical formula "LiMnO4"
    name: Optional[str] = None # filename of the spectrum e.g., "LiMnO4_2025_04_01_superxas"
    edge: Optional[float] = None
    edge_type: Optional[str] = None  # K, L1, L2, L3, M1, M2, M3, etc.
    oxidation_state: Optional[float] = None
    coordination_number: Optional[float] = None
    chemical_name: Optional[str] = None # e.g., "Manganese ..."
    chemical_formula: Optional[str] = None # e.g., "LiMnO4"
    phase: Optional[str] = None # e.g., "Phase 1", "Phase 2" for polymorphs
    structure: Optional[str] = None
    
    # === File Paths ===
    path_spectrum: Optional[str] = None
    path_I0: Optional[str] = None
    path_I1: Optional[str] = None
    
    # === Normalization Parameters ===
    pre_edge_min_E: Optional[float] = -50.0
    pre_edge_max_E: Optional[float] = -10.0
    post_edge_min_E: Optional[float] = 50.0
    post_edge_max_E: Optional[float] = 200.0
    pre_edge_fit_func: Optional[str] = "V"  # Victoreen, linear, etc.
    post_edge_fit_func: Optional[str] = "V"
    
    # === Instrument/Experimental ===
    beamline: Optional[str] = None    
    measurement_type: Optional[str] = None # e.g., "XAS", "XES", "XFS"
    monochromator: Optional[str] = None
    # temperature: Optional[float] = None
    # measurement_date: Optional[str] = None
    
    # === Processing Status ===
    processed: Optional[bool] = False
    resampled: Optional[bool] = False

    # === Additional Metadata ===
    glitches: Optional[List[tuple]] = None  # List of glitches where each entry is a (start_energy, end_energy) tuple.

    def __repr__(self) -> str:
        """String representation showing key information."""
        element = self.element or "Unknown"
        compound = self.compound or "Unknown"
        edge_info = f" ({self.edge} eV)" if self.edge else ""
        return f"XASMetadata({element} {compound}{edge_info})"
        
    def get(self, attribute: str) -> Any:
        """
        Get an attribute value, similar to dict.get().
        
        Args:
            attribute: Name of the attribute to get
            
        Returns:
            Attribute value

        Example:
            meta.get('element')  # Returns element value
        """
        return getattr(self, attribute, None)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary including both predefined and dynamic attributes.
        
        Args:
            include_dynamic: If True, include dynamically added attributes
            
        Returns:
            Dictionary with all metadata
        """
        # Get predefined fields from dataclass
        result = asdict(self)
        
        # Add any dynamically assigned attributes
        predefined_fields = {f.name for f in fields(self)}
        for key, value in self.__dict__.items():
            if key not in predefined_fields:
                result[key] = value
        
        # Remove None values and empty strings for cleaner output
        # Handle numpy arrays and other iterables separately to avoid ambiguous truth value errors
        def should_keep(v):
            if v is None:
                return False
            # Check if it's a numpy array or similar iterable (but not a string)
            if hasattr(v, '__len__') and not isinstance(v, (str, dict)):
                return True  # Keep arrays/lists even if empty
            if v == "":
                return False
            return True
        
        return {k: v for k, v in result.items() if should_keep(v)}
    
    def from_dict(self, data: Dict[str, Any]) -> 'XASMetadata':
        """
        Load from dictionary, setting both predefined and dynamic attributes.
        
        Args:
            data: Dictionary with metadata
            
        Returns:
            Self for method chaining
        """
        for key, value in data.items():
            setattr(self, key, value)
        return self
    
    def update(self, **kwargs) -> 'XASMetadata':
        """
        Update metadata with new values.
        
        Args:
            **kwargs: Key-value pairs to update
            
        Returns:
            Self for method chaining
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
        return self
    
    def to_csv(self, filepath: str) -> None:
        """
        Export to CSV format (key-value pairs).
        
        Args:
            filepath: Path to save CSV file
        """
        data = self.to_dict()
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['key', 'value'])
            for key, value in data.items():
                writer.writerow([key, value])
    
    def from_csv(self, filepath: str) -> 'XASMetadata':
        """
        Load from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            Self for method chaining
        """
        data = {}
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = row['key']
                value = row['value']
                # Basic type conversion
                data[key] = self._convert_value(value)
        
        return self.from_dict(data)
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate Python type."""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            return value
    
    
    def copy(self) -> 'XASMetadata':
        """Create a deep copy of the metadata."""
        return XASMetadata().from_dict(self.to_dict())
    