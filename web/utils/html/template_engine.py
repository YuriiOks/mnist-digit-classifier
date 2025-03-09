# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/html/template_engine.py
# Description: Simple HTML template engine
# Created: 2024-05-01

import re
import logging
from typing import Dict, Any, Union
from pathlib import Path

from utils.html.html_loader import load_html_template, HTMLLoadError
from core.errors.ui_errors import TemplateError

logger = logging.getLogger(__name__)


class TemplateEngine:
    """Simple template engine for HTML rendering."""
    
    @staticmethod
    def render(template: str, variables: Dict[str, Any]) -> str:
        """Render a template with variables.
        
        Args:
            template: HTML template string with ${VARIABLE} placeholders.
            variables: Dictionary of variables to replace in the template.
            
        Returns:
            str: Rendered HTML with variables replaced.
        """
        logger.debug(f"Rendering template with {len(variables)} variables")
        rendered = template
        
        # Replace each variable in the template
        for key, value in variables.items():
            placeholder = f"${{{key}}}"
            rendered = rendered.replace(placeholder, str(value))
        
        # Look for any unreplaced variables
        unreplaced_vars = re.findall(r'\${([A-Z_]+)}', rendered)
        if unreplaced_vars:
            logger.warning(
                f"Unreplaced variables in template: {unreplaced_vars}"
            )
            # Replace them with empty strings
            for var in unreplaced_vars:
                rendered = rendered.replace(f"${{{var}}}", "")
        
        return rendered
    
    @classmethod
    def render_from_file(
        cls, template_path: Union[str, Path], variables: Dict[str, Any]
    ) -> str:
        """Load a template from a file and render it by substituting variables.
        
        Args:
            template_path: Path to the template file, relative to the templates 
                directory.
            variables: Dictionary of variables to substitute in the template.
        
        Returns:
            str: The rendered template with variables substituted.
        
        Raises:
            TemplateError: If the template file cannot be loaded or a required 
                variable is missing.
        """
        logger.debug(f"Rendering template file: {template_path}")
        try:
            template_content = load_html_template(template_path)
            return cls.render(template_content, variables)
        except HTMLLoadError as e:
            logger.error(
                f"Failed to load template {template_path}: {str(e)}", 
                exc_info=True
            )
            raise TemplateError(
                f"Failed to load template {template_path}: {str(e)}",
                template_file=str(template_path)
            ) from e
    
    @classmethod
    def render_component(
        cls, 
        component_type: str, 
        template_name: str, 
        variables: Dict[str, Any]
    ) -> str:
        """Render a component template with variable substitution.
        
        Args:
            component_type: Type of component (layout, cards, inputs, etc.).
            template_name: Name of the template file (without extension).
            variables: Dictionary of variables to substitute in the template.
        
        Returns:
            str: The rendered template with variables substituted.
        
        Raises:
            TemplateError: If the template file cannot be loaded or a required variable is missing.
        """
        template_path = f"components/{component_type}/{template_name}.html"
        return cls.render_from_file(template_path, variables)
    
    @classmethod
    def render_view(cls, view_type: str, template_name: str, variables: Dict[str, Any]) -> str:
        """Render a view template with variable substitution.
        
        Args:
            view_type: Type of view (home, drawing, history, etc.).
            template_name: Name of the template file (without extension).
            variables: Dictionary of variables to substitute in the template.
        
        Returns:
            str: The rendered template with variables substituted.
        
        Raises:
            TemplateError: If the template file cannot be loaded or a required variable is missing.
        """
        template_path = f"views/{view_type}/{template_name}.html"
        return cls.render_from_file(template_path, variables)