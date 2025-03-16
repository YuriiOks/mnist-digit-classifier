# MNIST Digit Classifier
# File: utils/resource_manager.py
# Complete rewrite for path resolution

import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional, Union, List

import streamlit as st

class ResourceType(Enum):
    """Types of resources that can be loaded."""
    CSS = "css"
    JS = "javascript"
    TEMPLATE = "template"
    DATA = "data"
    IMAGE = "image"
    CONFIG = "config"


class ResourceManager:
    """Unified manager for loading various resource types."""
    
    # Singleton instance
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern for ResourceManager."""
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the resource manager."""
        # Skip initialization if already done (singleton pattern)
        if self._initialized:
            return
            
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._project_root = project_root or self._detect_project_root()
        

        # Base directories for different resource types - relative to project root
        self._BASE_DIRS = {
            ResourceType.CSS: ["assets/css"],
            ResourceType.JS: ["assets/js"],
            ResourceType.TEMPLATE: ["assets/templates"],
            ResourceType.DATA: ["assets/data"],
            ResourceType.IMAGE: ["assets/images"],
            ResourceType.CONFIG: ["assets/config"],
        }
        
        # Log the actual paths we'll be checking
        for res_type, dirs in self._BASE_DIRS.items():
            paths = [str(self._project_root / d) for d in dirs]
            self._logger.debug(f"Resource paths for {res_type.value}: {paths}")
        
        # Cache for loaded resources
        self._cache = {res_type: {} for res_type in ResourceType}
        
        self._initialized = True
        self._logger.info(f"ResourceManager initialized with root: {self._project_root}")

    def _detect_project_root(self) -> Path:
        """Detect the project root directory."""
        # In Docker, the project root is always /app
        docker_path = Path("/app")
        if docker_path.exists():
            self._logger.debug(f"Docker environment detected, using root: {docker_path}")
            return docker_path
            
        # Otherwise use current directory
        cwd = Path.cwd()
        self._logger.debug(f"Using current directory as root: {cwd}")

        return cwd
        
    def get_resource_path(self, resource_type: ResourceType, resource_path: str) -> Optional[Path]:
        """
        Get the full path to a resource.
        
        Args:
            resource_type: Type of resource (CSS, JS, etc.)
            resource_path: Relative path to the resource
            
        Returns:
            Full path to the resource, or None if not found
        """
        # Handle absolute paths
        if Path(resource_path).is_absolute():
            return Path(resource_path) if Path(resource_path).exists() else None
        
        # Strip leading slash if present
        resource_path = resource_path.lstrip("/")
        
        # Paths to check (direct and with components/ prefix)
        paths_to_check = [resource_path]
        
        # Add a components/ prefix if not already there
        if not resource_path.startswith("components/"):
            paths_to_check.append(f"components/{resource_path}")
        
        # Check each base directory with each path variant
        checked_paths = []
        for base_dir in self._BASE_DIRS[resource_type]:
            for path in paths_to_check:
                full_path = self._project_root / base_dir / path
                checked_paths.append(str(full_path))
                
                if full_path.exists():
                    self._logger.debug(f"Found resource at: {full_path}")
                    return full_path
        
        # If resource not found, log what was checked
        self._logger.warning(f"Resource not found for {resource_type.value}: {resource_path}")
        self._logger.debug(f"Checked paths: {checked_paths}")
        return None

    def load_text_resource(self, resource_type: ResourceType, resource_path: str) -> Optional[str]:
        """Load a text-based resource."""
        # Get the full path to the resource
        full_path = self.get_resource_path(resource_type, resource_path)
        if not full_path:
            return None
            
        try:
            # Read the file
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self._logger.debug(f"Loaded {resource_type.value}: {resource_path}")
                return content
        except Exception as e:
            self._logger.error(f"Error loading {resource_type.value} {resource_path}: {str(e)}")
            return None
    
    def load_json_resource(self, resource_path: str) -> Optional[Any]:
        """Load a JSON resource."""
        # Handle theme files
        if resource_path in ["light.json", "dark.json", "default.json"]:
            resource_path = f"themes/{resource_path}"
        
        # Try to load from CONFIG for theme files
        content = self.load_text_resource(ResourceType.CONFIG, resource_path)
        
        # If not found and it's not a theme file, try DATA
        if content is None and "themes/" not in resource_path:
            content = self.load_text_resource(ResourceType.DATA, resource_path)
            
        if content is None:
            return None
            
        try:
            # Parse JSON
            data = json.loads(content)
            return data
        except json.JSONDecodeError as e:
            self._logger.error(f"Invalid JSON in {resource_path}: {str(e)}")
            return None

    def load_css(self, css_path: str) -> Optional[str]:
        """Load a CSS file."""
        # First try direct path
        content = self.load_text_resource(ResourceType.CSS, css_path)
        if content:
            return content
        
        # Try with plural/singular form correction (common issue in logs)
        if css_path.endswith('s.css'):
            alt_path = css_path[:-5] + '.css'  # cards.css -> card.css
        elif css_path.endswith('.css') and not css_path.endswith('s.css'):
            alt_path = css_path[:-4] + 's.css'  # card.css -> cards.css
        else:
            alt_path = None

        if alt_path:
            content = self.load_text_resource(ResourceType.CSS, alt_path)
            if content:
                return content

        # Try in components directory if not already there
        if not css_path.startswith("components/"):
            components_path = f"components/{css_path}"
            content = self.load_text_resource(ResourceType.CSS, components_path)
            if content:
                return content
            
        # Log what we tried with full paths
        self._logger.warning(f"Failed to load CSS file: {css_path} (checked alternatives)")
        return None

    def load_template(self, template_path: str) -> Optional[str]:
        """Load a template file."""
        # First try direct path
        content = self.load_text_resource(ResourceType.TEMPLATE, template_path)
        if content:
            return content
        
        # Try with components prefix if not already there
        if not template_path.startswith("components/"):
            components_path = f"components/{template_path}"
            content = self.load_text_resource(ResourceType.TEMPLATE, components_path)
            if content:
                return content
        
        # Try with alternate card names based on context
        if "card.html" in template_path:
            # Try specific card templates
            for card_type in ["feature_card.html", "welcome_card.html"]:
                alt_path = template_path.replace("card.html", card_type)
                content = self.load_text_resource(ResourceType.TEMPLATE, alt_path)
                if content:
                    return content

        self._logger.warning(f"Failed to load template: {template_path} (checked alternatives)")
        return None

    def render_template(self, template_path: str, context: Dict[str, Any]) -> Optional[str]:
        """Load and render a template with variable substitution."""
        template_content = self.load_template(template_path)
        if not template_content:
            self._logger.warning(f"Template not found: {template_path}")
            return None
            
        rendered = template_content
        
        try:
            # Replace template variables
            for key, value in context.items():
                placeholder = f"${{{key.upper()}}}"
                rendered = rendered.replace(placeholder, str(value))
                
            return rendered
        except Exception as e:
            self._logger.error(f"Error rendering template {template_path}: {str(e)}")
            return None
    
    def inject_css(self, css_content: str) -> None:
        """Inject CSS content into Streamlit."""
        if not css_content:
            return
            
        # Inject with proper style tags
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
    
    def load_and_inject_css(self, css_paths: List[str]) -> None:
        """Load and inject multiple CSS files."""
        combined_css = ""
        for css_path in css_paths:
            css_content = self.load_css(css_path)
            if css_content:
                combined_css += f"\n/* {css_path} */\n{css_content}"
            else:
                self._logger.warning(f"CSS file not found: {css_path}")
                
        if combined_css:
            self.inject_css(combined_css)
            
    def load_theme_json(self, theme_filename: str) -> Optional[Any]:
        """
        Load a theme JSON file from the correct location.
        
        Args:
            theme_filename: Name of the theme file (e.g., "light.json")
                
        Returns:
            Parsed JSON data, or None if loading failed
        """
        # Define the exact path where theme files should be found
        exact_path = self._project_root / "assets" / "config" / "themes" / theme_filename
        
        # Log the exact path we're checking
        self._logger.debug(f"Looking for theme file at: {exact_path}")
        
        if not exact_path.exists():
            self._logger.warning(f"Theme file not found: {exact_path}")
            return None
            
        try:
            # Read and parse the JSON file
            with open(exact_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse the JSON content
            data = json.loads(content)
            self._logger.debug(f"Successfully loaded theme file: {exact_path}")
            return data
        except Exception as e:
            self._logger.error(f"Error loading theme file {exact_path}: {str(e)}")
            return None
        
resource_manager = ResourceManager()