# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/base/component.py
# Description: Base component class for UI components
# Created: 2024-05-01

import streamlit as st
from typing import Dict, Any, Optional, List
import uuid
import logging

from utils.html.template_engine import TemplateEngine
from utils.html.html_loader import load_component_template
from utils.css.css_loader import load_component_css, inject_css
from core.errors.ui_errors import UIError, ComponentError, TemplateError


logger = logging.getLogger(__name__)


class Component:
    """Base class for UI components.
    
    This class provides common functionality for all UI components,
    including template rendering, CSS loading, and HTML output.
    """
    
    def __init__(
        self,
        component_type: str,
        component_name: str,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None,
        css_path: Optional[str] = None
    ):
        """Initialize a component.
        
        Args:
            component_type: Type of component (e.g., "layout", "card").
            component_name: Name of component (e.g., "header", "footer").
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the
                component.
            css_path: Custom CSS path to load instead of default component CSS.
        """
        logger.debug(
            f"Initializing component: {component_type}/{component_name}"
        )
        self.component_type = component_type
        self.component_name = component_name
        component_uuid = str(uuid.uuid4())[:8]
        self.id = id or f"{component_type}-{component_name}-{component_uuid}"
        self.classes = classes or []
        self.attributes = attributes or {}
        self.css_path = css_path
        
        # Create instance-specific logger
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        
        # Ensure component CSS is loaded
        self._load_css()
    
    def _load_css(self) -> None:
        """Load component CSS if it exists."""
        try:
            # First check for custom CSS path
            if self.css_path:
                try:
                    from utils.css.css_loader import load_css_file
                    css_content = load_css_file(self.css_path)
                    inject_css(css_content)
                except Exception as e:
                    self.logger.warning(f"Failed to load custom CSS: {str(e)}")
            else:
                # Try to load component-specific CSS
                try:
                    css_content = load_component_css(
                        self.component_type, 
                        self.component_name
                    )
                    inject_css(css_content)
                except Exception:
                    # Not all components have CSS, so silently ignore this
                    # error
                    pass
        except Exception as e:
            # Log but don't raise since CSS is not critical
            error = UIError(
                f"Failed to load CSS for {self.component_type}/"
                f"{self.component_name}: {str(e)}",
                component_type=self.component_type,
                component_name=self.component_name,
            )
            self.logger.warning(str(error))
    
    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for rendering.
        
        Override this method in subclasses to provide component-specific
        variables.
        
        Returns:
            Dict[str, Any]: Dictionary of variables for template rendering.
        """
        # Convert list of classes to space-separated string
        classes_str = " ".join(self.classes)
        
        # Convert attributes dictionary to HTML attribute string
        attrs_items = [(k, v) for k, v in self.attributes.items()]
        attrs_str = " ".join([f'{k}="{v}"' for k, v in attrs_items])
        
        return {
            "ID": self.id,
            "CLASSES": classes_str,
            "ATTRIBUTES": attrs_str
        }
    
    def render(self) -> str:
        """Render the component as HTML.
        
        Returns:
            str: HTML representation of the component.
            
        Raises:
            ComponentError: If rendering fails.
        """
        logger.debug(
            f"Rendering component: {self.component_type}/{self.component_name}"
        )
        try:
            # Get template and variables
            template_content = load_component_template(
                self.component_type, 
                self.component_name
            )
            template_vars = self.get_template_variables()
            
            # Render the template
            return TemplateEngine.render(template_content, template_vars)
        except (TemplateError, UIError) as e:
            logger.error(
                f"Error rendering component {self.component_type}/"
                f"{self.component_name}: {str(e)}", 
                exc_info=True
            )
            # Re-raise as ComponentError
            raise ComponentError(
                f"Failed to render component {self.component_type}/"
                f"{self.component_name}: {str(e)}",
                component_type=self.component_type,
                component_name=self.component_name,
                original_exception=e
            )
    
    def safe_render(self) -> str:
        """Safely render the component with error handling.
        
        This method catches exceptions during rendering and returns a
        fallback UI.
        
        Returns:
            str: HTML representation of the component or error message.
        """
        try:
            return self.render()
        except Exception as e:
            self.logger.error(
                f"Error rendering component: {str(e)}", 
                exc_info=True
            )
            
            # Return error UI
            return f"""
            <div style="
                border: 2px solid #f44336;
                border-radius: 4px;
                padding: 10px;
                background-color: #ffebee;
                color: #b71c1c;
                margin: 10px 0;
            ">
                <div><strong>Component Error:</strong> {self.component_type}/
                {self.component_name}</div>
                <div>{str(e)}</div>
            </div>
            """
    
    def display(self) -> None:
        """Display the component in Streamlit.
        
        Injects component-specific CSS before rendering the HTML.
        """
        self.logger.debug(
            f"Displaying {self.component_type}/{self.component_name} component"
        )
        try:
            # Load component-specific CSS
            component_css = ""
            try:
                from utils.css.css_loader import load_css_file
                css_path = (
                    f"components/{self.component_type}/"
                    f"{self.component_name}.css"
                )
                component_css = f"<style>{load_css_file(css_path)}</style>"
                self.logger.debug(f"Loaded component CSS from {css_path}")
            except Exception:
                # This is optional, so ignore errors
                pass
                
            # Render the component
            html = self.safe_render()
            
            # Display in Streamlit
            st.markdown(component_css + html, unsafe_allow_html=True)
            
        except Exception as e:
            self.logger.error(
                f"Error displaying component: {str(e)}", 
                exc_info=True
            )
            st.error(
                f"Error displaying component: "
                f"{self.component_type}/{self.component_name}"
            ) 