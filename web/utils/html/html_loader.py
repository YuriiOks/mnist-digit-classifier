# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/html/html_loader.py
# Description: HTML loading utilities
# Created: 2024-05-01

import logging
from typing import Union
from pathlib import Path
import os

from utils.file.path_utils import get_project_root, resolve_path
from core.errors.ui_errors import TemplateError

logger = logging.getLogger(__name__)


class HTMLLoadError(Exception):
    """Exception raised for errors in HTML template loading operations."""
    def __init__(self, message, template_file=None):
        self.message = message
        self.template_file = template_file
        super().__init__(self.message)


def load_html_template(template_path: Union[str, Path]) -> str:
    """Load an HTML template file.
    
    Args:
        template_path: Path to the template file, relative to the templates
            directory.
    
    Returns:
        str: The contents of the template file.
    """
    logger.debug(f"Loading HTML template: {template_path}")
    try:
        # If template_path is a relative path, resolve it relative to templates directory
        if (isinstance(template_path, str) 
                and not template_path.startswith('/')
                and not template_path.startswith('templates/')):
            template_path = f"templates/{template_path}"
            logger.debug(f"Resolved relative path to: {template_path}")
        
        # Get the absolute path
        abs_path = resolve_path(template_path)
        logger.debug(f"Resolved absolute path: {abs_path}")
        
        # Read the file
        with open(abs_path, 'r') as f:
            template_content = f.read()
            
        logger.debug(f"Successfully loaded HTML template: {template_path} ({len(template_content)} bytes)")
        return template_content
    except Exception as e:
        logger.error(f"Failed to load HTML template {template_path}: {str(e)}", exc_info=True)
        raise TemplateError(f"Failed to load HTML template {template_path}: {str(e)}", template_file=str(template_path)) from e


def load_view_template(view_id: str) -> str:
    """Load a template for a specific view.
    
    Args:
        view_id: ID of the view (e.g., "home", "settings").
    
    Returns:
        str: The contents of the view template.
        
    Raises:
        TemplateError: If the view template cannot be loaded.
    """
    logger.debug(f"Loading view template for: {view_id}")
    try:
        # Attempt to load view-specific template
        template_path = f"views/{view_id}/main.html"
        return load_html_template(template_path)
    except Exception as e:
        logger.error(f"Failed to load view template for {view_id}: {str(e)}", exc_info=True)
        raise TemplateError(f"Failed to load view template for {view_id}: {str(e)}", template_file=f"views/{view_id}/main.html") from e


def load_component_template(component_type: str, component_name: str) -> str:
    """Load a component template.
    
    Args:
        component_type: Type of component (e.g., "cards", "controls").
        component_name: Name of the component (e.g., "card", "button").
        
    Returns:
        str: Template content.
        
    Raises:
        FileNotFoundError: If the template file doesn't exist.
    """
    logger.debug(f"Loading template for {component_type}/{component_name}")
    template_path = os.path.join(
        get_project_root(),
        "templates", 
        "components",
        component_type,
        f"{component_name}.html"
    )
    
    try:
        with open(template_path, "r") as f:
            content = f.read()
        logger.debug(f"Template loaded successfully: {len(content)} bytes")
        return content
    except FileNotFoundError:
        logger.error(f"Template not found: {template_path}")
        # Return a minimal template as fallback
        return f"<div>Component {component_type}/{component_name} template not found</div>"
    except Exception as e:
        logger.error(f"Error loading template: {str(e)}")
        raise


def has_template(template_path: Union[str, Path]) -> bool:
    """Check if a template file exists.
    
    Args:
        template_path: Path to the template file, relative to the templates directory.
    
    Returns:
        bool: True if the template exists, False otherwise.
    """
    logger.debug(f"Checking if template exists: {template_path}")
    try:
        # If template_path is a relative path, resolve it relative to templates directory
        if (isinstance(template_path, str) 
                and not template_path.startswith('/')
                and not template_path.startswith('templates/')):
            template_path = f"templates/{template_path}"
            logger.debug(f"Resolved relative path to: {template_path}")
        
        # Get the absolute path
        abs_path = resolve_path(template_path)
        
        # Check if the file exists
        exists = abs_path.exists()
        logger.debug(f"Template existence check: {template_path} - {'Exists' if exists else 'Not found'}")
        return exists
    except Exception as e:
        logger.error(f"Error checking template existence {template_path}: {str(e)}")
        return False 
    
