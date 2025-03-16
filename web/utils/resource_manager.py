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
        
        # Print debugging info if DEBUG_RESOURCES is set
        if os.environ.get("DEBUG_RESOURCES") == "true":
            self._logger.info(f"ResourceManager initialized with root: {self._project_root}")
            self._logger.info(f"Checking for assets directory: {self._project_root / 'assets'}")
            
            # List CSS directories
            css_dir = self._project_root / "assets" / "css"
            if css_dir.exists():
                self._logger.info(f"CSS directory exists: {css_dir}")
                for subdir in ["components", "global", "themes", "views"]:
                    path = css_dir / subdir
                    if path.exists():
                        self._logger.info(f"  - {subdir}/ ✓")
                        # If components, check component directories
                        if subdir == "components":
                            for comp_dir in ["cards", "controls", "layout"]:
                                comp_path = path / comp_dir
                                if comp_path.exists():
                                    self._logger.info(f"    - {comp_dir}/ ✓")
                                    # List CSS files in each directory
                                    for css_file in comp_path.glob("*.css"):
                                        self._logger.info(f"      * {css_file.name}")
                                else:
                                    self._logger.warning(f"    - {comp_dir}/ ✗ MISSING")
                    else:
                        self._logger.warning(f"  - {subdir}/ ✗ MISSING")
            else:
                self._logger.warning(f"CSS directory NOT FOUND: {css_dir}")
        

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
        """Get the full path to a resource with improved path resolution."""
        # Log requested path for debugging
        self._logger.debug(f"Resolving path for {resource_type.value}: {resource_path}")
        
        # Handle absolute paths
        if Path(resource_path).is_absolute():
            return Path(resource_path) if Path(resource_path).exists() else None
        
        # Strip leading slash if present
        resource_path = resource_path.lstrip("/")
        
        # Base directories to check based on resource type
        base_dirs = self._BASE_DIRS[resource_type]
        
        # Iterate through base directories
        for base_dir in base_dirs:
            full_path = self._project_root / base_dir / resource_path
            if full_path.exists():
                return full_path
                
            # If resource_type is CONFIG, also check DATA directories
            if resource_type == ResourceType.CONFIG:
                for data_dir in self._BASE_DIRS[ResourceType.DATA]:
                    data_path = self._project_root / data_dir / resource_path
                    if data_path.exists():
                        return data_path
        
        # Check the direct path under project root as a last resort
        direct_path = self._project_root / resource_path
        if direct_path.exists():
            return direct_path
            
        # Log the failure
        self._logger.debug(f"Resource not found: {resource_type.value}/{resource_path}")
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
        """Load a JSON resource with improved path resolution."""
        # Check if it's in config first
        config_content = self.load_text_resource(ResourceType.CONFIG, resource_path)
        
        # If not found in config, try data directory
        if config_content is None:
            data_content = self.load_text_resource(ResourceType.DATA, resource_path)
            if data_content:
                try:
                    return json.loads(data_content)
                except json.JSONDecodeError as e:
                    self._logger.error(f"Invalid JSON in data/{resource_path}: {str(e)}")
                    return None
        else:
            try:
                return json.loads(config_content)
            except json.JSONDecodeError as e:
                self._logger.error(f"Invalid JSON in config/{resource_path}: {str(e)}")
                return None
                
        # If still not found, try with home/ prefix
        if not resource_path.startswith("home/"):
            home_path = f"home/{resource_path}"
            return self.load_json_resource(home_path)
        
        # Log failure and return None
        self._logger.warning(f"JSON resource not found: {resource_path}")
        return None

    def load_css(self, css_path: str) -> Optional[str]:
        """Load a CSS file with robust path resolution for singular/plural variants."""
        # Log requested path
        self._logger.debug(f"Attempting to load CSS: {css_path}")
        
        # Try the original path first
        content = self.load_text_resource(ResourceType.CSS, css_path)
        if content:
            return content
        
        # Try plural/singular variants
        if css_path.endswith('.css'):
            base_path = css_path[:-4]  # Remove .css extension
            
            # Try adding/removing 's' at the end
            if base_path.endswith('s'):
                # Try singular form
                singular_path = f"{base_path[:-1]}.css"
                content = self.load_text_resource(ResourceType.CSS, singular_path)
                if content:
                    self._logger.info(f"Found CSS using singular form: {singular_path}")
                    return content
            else:
                # Try plural form
                plural_path = f"{base_path}s.css"
                content = self.load_text_resource(ResourceType.CSS, plural_path)
                if content:
                    self._logger.info(f"Found CSS using plural form: {plural_path}")
                    return content
        
        # Log what we tried
        self._logger.warning(f"Failed to load CSS file: {css_path} (tried singular/plural forms)")
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

    def load_data_json(self, json_path: str) -> Optional[Dict[str, Any]]:
        """Load a JSON file from the data directory with path variants."""
        # First try as config file
        data = self.load_json_resource(json_path)
        if data:
            self._logger.info(f"✅ Successfully loaded JSON from config: {json_path}")
            return data
        
        # Try data directory with "data/" prefix
        if not json_path.startswith("data/"):
            data_path = f"data/{json_path}"
            data = self.load_json_resource(data_path)
            if data:
                self._logger.info(f"✅ Successfully loaded JSON from data: {data_path}")
                return data
        
        # List available JSON files to help troubleshooting
        data_dir = self._project_root / "assets/data"
        if data_dir.exists():
            self._logger.info("Available JSON files in data directory:")
            for dirpath, _, filenames in os.walk(data_dir):
                for filename in filenames:
                    if filename.endswith('.json'):
                        rel_path = os.path.relpath(os.path.join(dirpath, filename), data_dir)
                        self._logger.info(f"  - {rel_path}")
        
        # If all attempts fail, create an empty fallback
        self._logger.warning(f"Unable to load JSON: {json_path} - using fallback")
        if "welcome_card" in json_path:
            return {
                "title": "MNIST Digit Classifier",
                "icon": "👋",
                "content": "Welcome to the MNIST Digit Classifier. Draw or upload a digit to get started."
            }
        elif "feature_cards" in json_path:
            return [
                {"title": "Draw", "icon": "✏️", "content": "Draw a digit from 0-9"},
                {"title": "Upload", "icon": "📤", "content": "Upload an image"},
                {"title": "Predict", "icon": "🔮", "content": "Get AI predictions"}
            ]
        
        return {}


resource_manager = ResourceManager()