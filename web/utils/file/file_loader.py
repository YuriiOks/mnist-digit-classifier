# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/file/file_loader.py
# Description: Utilities for loading various file types
# Created: 2024-05-01

import json
import logging
import yaml
import csv
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

from utils.file.path_utils import resolve_path

logger = logging.getLogger(__name__)

class FileLoadError(Exception):
    """Exception raised when file loading fails."""
    pass


def load_json_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a JSON file.
    
    Args:
        file_path: Path to the JSON file.
    
    Returns:
        Dict[str, Any]: Parsed JSON data.
        
    Raises:
        FileLoadError: If the file cannot be loaded or parsed.
    """
    logger.debug(f"Loading JSON file: {file_path}")
    try:
        # Resolve the path
        path = resolve_path(file_path)
        
        # Read and parse the file
        with open(path, 'r') as f:
            data = json.load(f)
        
        logger.debug(f"Successfully loaded JSON file: {file_path}")
        return data
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error in file {file_path}: {str(e)}")
        raise FileLoadError(f"Failed to parse JSON in {file_path}: {str(e)}") from e
    except Exception as e:
        logger.error(f"Failed to load JSON file {file_path}: {str(e)}", exc_info=True)
        raise FileLoadError(f"Failed to load JSON file {file_path}: {str(e)}") from e


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load a YAML file.
    
    Args:
        file_path: Path to the YAML file.
    
    Returns:
        Dict[str, Any]: Parsed YAML data.
        
    Raises:
        FileLoadError: If the file cannot be loaded or parsed.
    """
    logger.debug(f"Loading YAML file: {file_path}")
    try:
        # Resolve the path
        path = resolve_path(file_path)
        
        # Read and parse the file
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        logger.debug(f"Successfully loaded YAML file: {file_path}")
        return data
    except yaml.YAMLError as e:
        logger.error(f"YAML parse error in file {file_path}: {str(e)}")
        raise FileLoadError(f"Failed to parse YAML in {file_path}: {str(e)}") from e
    except Exception as e:
        logger.error(f"Failed to load YAML file {file_path}: {str(e)}", exc_info=True)
        raise FileLoadError(f"Failed to load YAML file {file_path}: {str(e)}") from e


def load_csv_file(file_path: Union[str, Path], has_header: bool = True) -> List[Dict[str, str]]:
    """Load a CSV file.
    
    Args:
        file_path: Path to the CSV file.
        has_header: Whether the CSV file has a header row.
    
    Returns:
        List[Dict[str, str]]: List of dictionaries, one per row.
        
    Raises:
        FileLoadError: If the file cannot be loaded or parsed.
    """
    logger.debug(f"Loading CSV file: {file_path}, has_header: {has_header}")
    try:
        # Resolve the path
        path = resolve_path(file_path)
        
        # Read and parse the file
        with open(path, 'r', newline='') as f:
            if has_header:
                reader = csv.DictReader(f)
                data = list(reader)
            else:
                reader = csv.reader(f)
                data = []
                for row in reader:
                    data.append({str(i): cell for i, cell in enumerate(row)})
        
        logger.debug(f"Successfully loaded CSV file: {file_path}, {len(data)} rows")
        return data
    except csv.Error as e:
        logger.error(f"CSV parse error in file {file_path}: {str(e)}")
        raise FileLoadError(f"Failed to parse CSV in {file_path}: {str(e)}") from e
    except Exception as e:
        logger.error(f"Failed to load CSV file {file_path}: {str(e)}", exc_info=True)
        raise FileLoadError(f"Failed to load CSV file {file_path}: {str(e)}") from e


def load_text_file(file_path: Union[str, Path], encoding: str = "utf-8") -> str:
    """Load a text file and return its contents as a string.
    
    Args:
        file_path: Path to the text file.
    
    Returns:
        str: File contents.
        
    Raises:
        FileLoadError: If the file cannot be loaded.
    """
    logger.debug(f"Loading text file: {file_path}")
    try:
        path = Path(file_path)
        with open(path, 'r', encoding=encoding) as file:
            content = file.read()  # Store file content in variable
        logger.debug(f"Successfully loaded text file: {file_path}, {len(content)} bytes")
        return content
    except (FileNotFoundError, PermissionError, UnicodeDecodeError) as e:
        logger.error(f"Failed to load text file {file_path}: {str(e)}", exc_info=True)
        raise FileLoadError(f"Failed to load text file {file_path}: {str(e)}")

def save_json_file(data: Dict[str, Any], file_path: Union[str, Path], pretty: bool = True) -> None:
    """Save data to a JSON file.
    
    Args:
        data: Data to save.
        file_path: Path to save the file to.
        pretty: Whether to format the JSON for readability.
        
    Raises:
        FileLoadError: If the file cannot be saved.
    """
    logger.debug(f"Saving JSON file: {file_path}, pretty: {pretty}")
    try:
        # Resolve the path
        path = resolve_path(file_path)
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        with open(path, 'w') as f:
            if pretty:
                json.dump(data, f, indent=2)
            else:
                json.dump(data, f)
        
        logger.debug(f"Successfully saved JSON file: {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON file {file_path}: {str(e)}", exc_info=True)
        raise FileLoadError(f"Failed to save JSON file {file_path}: {str(e)}") from e


def load_binary_file(file_path: Union[str, Path], *, from_root: bool = True) -> bytes:
    """Load a binary file and return its contents as bytes.
    
    Args:
        file_path: Path to the file to load.
        from_root: Whether to resolve the path from the project root.
    
    Returns:
        bytes: The contents of the file.
    
    Raises:
        FileLoadError: If the file cannot be loaded.
    """
    try:
        path = resolve_path(file_path, from_root=from_root)
        with open(path, 'rb') as file:
            return file.read()
    except (FileNotFoundError, PermissionError) as e:
        raise FileLoadError(f"Failed to load binary file {file_path}: {str(e)}") from e


def save_text_file(content: str, file_path: Union[str, Path], *, 
                   from_root: bool = True, encoding: str = "utf-8") -> None:
    """Save text content to a file.
    
    Args:
        content: The text content to save.
        file_path: Path where the file should be saved.
        from_root: Whether to resolve the path from the project root.
        encoding: The encoding to use when writing the file.
    
    Raises:
        FileLoadError: If the file cannot be saved.
    """
    try:
        path = resolve_path(file_path, from_root=from_root)
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding=encoding) as file:
            file.write(content)
    except (FileNotFoundError, PermissionError, UnicodeEncodeError) as e:
        raise FileLoadError(f"Failed to save text file {file_path}: {str(e)}") from e


def file_exists(file_path: Union[str, Path], *, from_root: bool = True) -> bool:
    """Check if a file exists.
    
    Args:
        file_path: Path to the file to check.
        from_root: Whether to resolve the path from the project root.
    
    Returns:
        bool: True if the file exists, False otherwise.
    """
    path = resolve_path(file_path, from_root=from_root)
    return path.is_file() 