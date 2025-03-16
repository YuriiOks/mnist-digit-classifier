# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/resource_manager.py
# Description: Unified resource loading and management
# Created: 2025-03-16

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
    """
    Unified manager for loading various resource types.
    
    This class provides centralized access to application resources including
    CSS files, JavaScript, HTML templates, JSON data, and images. It handles
    path resolution, caching, and content injection.
    """
    
    # Singleton instance
    _instance = None
    
    # Base directories for different resource types
    _BASE_DIRS = {
        ResourceType.CSS: ["assets/css", "css", "assets/styles"],
        ResourceType.JS: ["assets/js", "js", "assets/javascript"],
        ResourceType.TEMPLATE: ["assets/templates", "templates", "views"],
        ResourceType.DATA: ["assets/data", "data"],
        ResourceType.IMAGE: ["assets/images", "images", "assets/img"],
        ResourceType.CONFIG: ["assets/config", "config"],
    }
    
    # Cache for loaded resources to prevent repeated disk reads
    _cache = {
        ResourceType.CSS: {},
        ResourceType.JS: {},
        ResourceType.TEMPLATE: {},
        ResourceType.DATA: {},
        ResourceType.CONFIG: {},
    }
    
    def __new__(cls, *args, **kwargs):
        """Ensure singleton pattern for ResourceManager."""
        if cls._instance is None:
            cls._instance = super(ResourceManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, project_root: Optional[Path] = None):
        """
        Initialize the resource manager.
        
        Args:
            project_root: Optional path to project root directory.
                          If None, will be auto-detected.
        """
        # Skip initialization if already done (singleton pattern)
        if self._initialized:
            return
            
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._project_root = project_root or self._detect_project_root()
        self._initialized = True
        self._logger.info(f"ResourceManager initialized with root: {self._project_root}")
    
    def _detect_project_root(self) -> Path:
        """
        Detect the project root directory.
        
        Returns:
            Path to the project root directory.
        """
        # Try to find app.py as a marker for project root
        cwd = Path.cwd()
        
        # Check current directory for app.py
        if (cwd / "app.py").exists():
            return cwd
            
        # Check for Docker-style paths
        for docker_path in ["/app", "/usr/src/app"]:
            path = Path(docker_path)
            if path.exists() and (path / "app.py").exists():
                return path
                
        # Fall back to current directory with a warning
        self._logger.warning(f"Could not detect project root, using {cwd}")
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
        
        # Try each base directory for this resource type
        for base_dir in self._BASE_DIRS[resource_type]:
            # Try direct path
            full_path = self._project_root / base_dir / resource_path
            if full_path.exists():
                return full_path
            
            # If dealing with a component, try different component paths
            if resource_type in [ResourceType.CSS, ResourceType.TEMPLATE, ResourceType.JS]:
                # Try with 'components/' prefix if not already present
                if not resource_path.startswith("components/"):
                    alt_path = self._project_root / base_dir / "components" / resource_path
                    if alt_path.exists():
                        return alt_path
                
                # Try without 'components/' prefix if already present
                if resource_path.startswith("components/"):
                    alt_path = self._project_root / base_dir / resource_path[11:]
                    if alt_path.exists():
                        return alt_path
        
        # If resource not found, log a warning and return None
        self._logger.warning(f"Resource not found: {resource_type.value}/{resource_path}")
        return None
    
    def load_text_resource(self, resource_type: ResourceType, resource_path: str) -> Optional[str]:
        """
        Load a text-based resource.
        
        Args:
            resource_type: Type of resource
            resource_path: Path to the resource
            
        Returns:
            Content of the resource as string, or None if not found
        """
        # Check cache first
        cache_key = f"{resource_type.value}:{resource_path}"
        if cache_key in self._cache[resource_type]:
            return self._cache[resource_type][cache_key]
        
        # Get the full path to the resource
        full_path = self.get_resource_path(resource_type, resource_path)
        if not full_path:
            return None
            
        try:
            # Read the file and add to cache
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
                self._cache[resource_type][cache_key] = content
                self._logger.debug(f"Loaded {resource_type.value}: {resource_path}")
                return content
        except Exception as e:
            self._logger.error(f"Error loading {resource_type.value} {resource_path}: {str(e)}")
            return None
    
    def load_binary_resource(self, resource_type: ResourceType, resource_path: str) -> Optional[bytes]:
        """
        Load a binary resource.
        
        Args:
            resource_type: Type of resource
            resource_path: Path to the resource
            
        Returns:
            Content of the resource as bytes, or None if not found
        """
        # Get the full path to the resource
        full_path = self.get_resource_path(resource_type, resource_path)
        if not full_path:
            return None
            
        try:
            # Read the file
            with open(full_path, 'rb') as f:
                content = f.read()
                self._logger.debug(f"Loaded binary {resource_type.value}: {resource_path}")
                return content
        except Exception as e:
            self._logger.error(f"Error loading binary {resource_type.value} {resource_path}: {str(e)}")
            return None
    
    def load_json_resource(self, resource_path: str) -> Optional[Any]:
        """
        Load a JSON resource.
        
        Args:
            resource_path: Path to the JSON resource
            
        Returns:
            Parsed JSON data, or None if loading failed
        """
        # Check cache first
        cache_key = f"json:{resource_path}"
        if cache_key in self._cache[ResourceType.DATA]:
            return self._cache[ResourceType.DATA][cache_key]
        
        # Load the text content
        content = self.load_text_resource(ResourceType.DATA, resource_path)
        if not content:
            return None
            
        try:
            # Parse JSON and cache
            data = json.loads(content)
            self._cache[ResourceType.DATA][cache_key] = data
            return data
        except json.JSONDecodeError as e:
            self._logger.error(f"Invalid JSON in {resource_path}: {str(e)}")
            return None
    
    def load_css(self, css_path: str) -> Optional[str]:
        """
        Load a CSS file.
        
        Args:
            css_path: Path to the CSS file
            
        Returns:
            CSS content as string, or None if loading failed
        """
        return self.load_text_resource(ResourceType.CSS, css_path)
    
    def load_js(self, js_path: str) -> Optional[str]:
        """
        Load a JavaScript file.
        
        Args:
            js_path: Path to the JavaScript file
            
        Returns:
            JavaScript content as string, or None if loading failed
        """
        return self.load_text_resource(ResourceType.JS, js_path)
    
    def load_template(self, template_path: str) -> Optional[str]:
        """
        Load a template file.
        
        Args:
            template_path: Path to the template file
            
        Returns:
            Template content as string, or None if loading failed
        """
        return self.load_text_resource(ResourceType.TEMPLATE, template_path)
    
    def render_template(self, template_path: str, context: Dict[str, Any]) -> Optional[str]:
        """
        Load and render a template with variable substitution.
        
        Args:
            template_path: Path to the template
            context: Dictionary of variables to substitute
            
        Returns:
            Rendered template as string, or None if rendering failed
        """
        template_content = self.load_template(template_path)
        if not template_content:
            return None
            
        rendered = template_content
        
        try:
            # First try modern {{variable}} syntax
            for key, value in context.items():
                placeholder = f"{{{{{key}}}}}"
                rendered = rendered.replace(placeholder, str(value))
            
            # Then try legacy ${VARIABLE} syntax
            for key, value in context.items():
                placeholder = f"${{{key.upper()}}}"
                rendered = rendered.replace(placeholder, str(value))
                
            return rendered
        except Exception as e:
            self._logger.error(f"Error rendering template {template_path}: {str(e)}")
            return None
    
    def inject_css(self, css_content: str) -> None:
        """
        Inject CSS content into Streamlit.
        
        Args:
            css_content: CSS content to inject
        """
        if not css_content:
            return
            
        # Remove any existing style tags
        clean_css = css_content
        if "<style>" in css_content.lower():
            clean_css = css_content.replace("<style>", "").replace("</style>", "")
        
        # Inject with proper style tags
        st.markdown(f"<style>{clean_css}</style>", unsafe_allow_html=True)
        self._logger.debug("Injected CSS content")
    
    def inject_js(self, js_content: str) -> None:
        """
        Inject JavaScript content into Streamlit.
        
        Args:
            js_content: JavaScript content to inject
        """
        if not js_content:
            return
            
        # Wrap with script tags and inject
        html = f"""
        <script type="text/javascript">
            (function() {{
                {js_content}
            }})();
        </script>
        """
        st.components.v1.html(html, height=0)
        self._logger.debug("Injected JavaScript content")
    
    def load_and_inject_css(self, css_paths: List[str]) -> None:
        """
        Load and inject multiple CSS files.
        
        Args:
            css_paths: List of paths to CSS files
        """
        combined_css = ""
        for css_path in css_paths:
            css_content = self.load_css(css_path)
            if css_content:
                combined_css += f"\n/* {css_path} */\n{css_content}"
                
        if combined_css:
            self.inject_css(combined_css)
    
    def load_and_inject_js(self, js_paths: List[str]) -> None:
        """
        Load and inject multiple JavaScript files.
        
        Args:
            js_paths: List of paths to JavaScript files
        """
        combined_js = ""
        for js_path in js_paths:
            js_content = self.load_js(js_path)
            if js_content:
                combined_js += f"\n// {js_path}\n{js_content}\n"
                
        if combined_js:
            self.inject_js(combined_js)
    
    def clear_cache(self, resource_type: Optional[ResourceType] = None) -> None:
        """
        Clear the resource cache.
        
        Args:
            resource_type: Specific resource type to clear, or None for all
        """
        if resource_type:
            self._cache[resource_type] = {}
            self._logger.debug(f"Cleared {resource_type.value} cache")
        else:
            for res_type in ResourceType:
                self._cache[res_type] = {}
            self._logger.debug("Cleared all resource caches")


# Create a singleton instance for easy importing
resource_manager = ResourceManager()