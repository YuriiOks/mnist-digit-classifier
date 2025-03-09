# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/file/path_utils.py
# Description: Path-related utilities
# Created: 2024-05-01

import os
import logging
from pathlib import Path
from typing import Union, Optional

logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """Get the absolute path to the project root directory.
    
    Returns:
        Path: Path to the project root directory.
    """
    logger.debug("Getting project root directory")
    try:
        # Find the project root by looking for key files or directories
        # The current file is in utils/file, so go up two levels
        current_file = Path(__file__).resolve()
        root = current_file.parent.parent.parent
        
        # Verify it's the project root by checking for key files/directories
        if (root / "app.py").exists() and (root / "utils").is_dir() and (root / "core").is_dir():
            logger.debug(f"Project root determined to be: {root}")
            return root
        
        # Alternative method: look for .git directory
        if (root / ".git").is_dir():
            logger.debug(f"Project root determined by .git to be: {root}")
            return root
        
        # If we can't find a good indicator, use parent of the utils directory
        logger.warning("Could not confidently determine project root, using best guess")
        return root
    except Exception as e:
        logger.error(f"Error determining project root: {str(e)}", exc_info=True)
        # Fall back to the directory containing utils
        current_file = Path(__file__).resolve()
        fallback_root = current_file.parent.parent.parent
        logger.warning(f"Using fallback project root: {fallback_root}")
        return fallback_root


def resolve_path(path: Union[str, Path], relative_to: Optional[Path] = None) -> Path:
    """Resolve a path, handling relative paths appropriately.
    
    Args:
        path: Path to resolve, can be absolute or relative.
        relative_to: Base path for relative paths. If None, uses project root.
    
    Returns:
        Path: Resolved absolute path.
    """
    logger.debug(f"Resolving path: {path}, relative_to: {relative_to}")
    try:
        # If path is already a Path object, make a copy to avoid modifying the original
        if isinstance(path, Path):
            p = Path(path)
        else:
            p = Path(path)
        
        # If path is absolute, return it directly
        if p.is_absolute():
            logger.debug(f"Path is absolute: {p}")
            return p
        
        # If no relative_to is provided, use project root
        if relative_to is None:
            relative_to = get_project_root()
            logger.debug(f"Using project root as base: {relative_to}")
        
        # Resolve the path relative to the base
        resolved = (relative_to / p).resolve()
        logger.debug(f"Resolved path: {resolved}")
        return resolved
    except Exception as e:
        logger.error(f"Error resolving path {path}: {str(e)}", exc_info=True)
        # Return the path as-is in case of error
        if isinstance(path, Path):
            return path
        return Path(path)


def normalize_path(path: Union[str, Path]) -> str:
    """Normalize a path for consistent representation.
    
    Args:
        path: Path to normalize.
        
    Returns:
        str: Normalized path string.
    """
    logger.debug(f"Normalizing path: {path}")
    try:
        # Convert to Path object if it's a string
        if isinstance(path, str):
            p = Path(path)
        else:
            p = path
        
        # Normalize the path
        normalized = str(p.resolve())
        logger.debug(f"Normalized path: {normalized}")
        return normalized
    except Exception as e:
        logger.error(f"Error normalizing path {path}: {str(e)}", exc_info=True)
        # Return the original path as a string
        return str(path)


def get_relative_path(path: Union[str, Path], base: Union[str, Path]) -> str:
    """Get the relative path from a base path.
    
    Args:
        path: Path to get relative representation of.
        base: Base path to get relative to.
        
    Returns:
        str: Relative path as a string.
    """
    logger.debug(f"Getting relative path: {path} relative to {base}")
    try:
        # Convert to Path objects if they're strings
        if isinstance(path, str):
            p = Path(path)
        else:
            p = path
            
        if isinstance(base, str):
            b = Path(base)
        else:
            b = base
        
        # Resolve both paths to absolute paths
        p = p.resolve()
        b = b.resolve()
        
        # Get the relative path
        relative = p.relative_to(b)
        logger.debug(f"Relative path: {relative}")
        return str(relative)
    except Exception as e:
        logger.error(f"Error getting relative path from {base} to {path}: {str(e)}", exc_info=True)
        # Return the original path as a string
        return str(path)


def ensure_directory_exists(directory_path: Union[str, Path]) -> Path:
    """Ensure that a directory exists, creating it if necessary.
    
    Args:
        directory_path: The path to the directory to ensure.
    
    Returns:
        Path: The absolute path to the directory.
    """
    resolved_path = Path(directory_path).resolve()
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def get_asset_path(asset_type: str, relative_path: Union[str, Path]) -> Path:
    """Get the absolute path to an asset file.
    
    Args:
        asset_type: Type of asset (css, js, images, etc.)
        relative_path: Path relative to the asset type directory.
    
    Returns:
        Path: The absolute path to the asset.
    
    Raises:
        ValueError: If the asset type is invalid.
    """
    valid_asset_types = ["css", "js", "images"]
    if asset_type not in valid_asset_types:
        raise ValueError(f"Invalid asset type: {asset_type}. Must be one of {valid_asset_types}")
    
    # Convert to string to manipulate the path
    path_str = str(relative_path)
    
    # Check if the path already includes the asset type prefix
    if path_str.startswith(f"assets/{asset_type}/") or path_str.startswith(f"{asset_type}/"):
        # The path already includes the prefix, use it directly
        return resolve_path(path_str)
    else:
        # Add the prefix
        return resolve_path(f"assets/{asset_type}/{path_str}")

def get_template_path(template_path: Union[str, Path]) -> Path:
    """Get the absolute path to a template file.
    
    Args:
        template_path: Path relative to the templates directory.
    
    Returns:
        Path: The absolute path to the template.
    """
    return resolve_path(f"templates/{template_path}") 