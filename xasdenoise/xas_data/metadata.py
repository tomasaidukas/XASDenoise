"""
Helper functions used to process/create metadata.
"""

import pandas as pd
from typing import Optional, List, Union


def load_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load metadata from an HDF5 file.

    Args:
        metadata_path (str): Path to the HDF5 file containing metadata.

    Returns:
        pd.DataFrame: Metadata as a pandas DataFrame.
    """
    return pd.read_hdf(metadata_path, key='metadata')

def extract_paths_from_metadata(metadata: pd.DataFrame) -> List[str]:
    """
    Extract all file paths from metadata columns containing 'path'.

    Args:
        metadata (pd.DataFrame): Metadata DataFrame.

    Returns:
        List[str]: List of file paths.
    """
    path_columns = [col for col in metadata.columns if 'path' in col]
    paths = metadata[path_columns].values.flatten()
    return [path for path in paths if isinstance(path, str)]

def filter_metadata(
    metadata: pd.DataFrame, key: str, value: Optional[Union[str, List[str]]] = None
) -> Union[pd.Series, pd.DataFrame]:
    """
    Filter metadata by a specific key and optional values.

    Args:
        metadata (pd.DataFrame): Metadata DataFrame.
        key (str): Column name to filter by.
        value (str or list, optional): Value(s) to filter. If None, returns the column.

    Returns:
        pd.Series or pd.DataFrame: Filtered metadata.
    """
    if value is None:
        # Return the column as a Series
        return metadata[key]
    else:
        # Filter rows based on key-value pairs
        value = [value] if isinstance(value, str) else value
        return metadata[metadata[key].isin(value)]
