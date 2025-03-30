# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: ui/components/base/component.py
# Description: Base component class for all UI components
# Created: 2025-03-16

import streamlit as st
import logging
from abc import ABC, abstractmethod
from typing import (
    Dict,
    Any,
    Optional,
    List,
    Union,
    Callable,
    TypeVar,
    Generic,
)

from utils.resource_manager import resource_manager, ResourceType
from utils.aspects import AspectUtils
from core.errors import UIError, TemplateError, ComponentError

# Type variable for component return values
T = TypeVar("T")


class Component(ABC, Generic[T]):
    """
    Base component class for all UI components.

    Provides common functionality for UI components including resource loading,
    state access, error handling, and a standardized rendering interface.
    """

    def __init__(
        self,
        component_type: str = "component",
        component_name: Optional[str] = None,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None,
        key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize the component.

        Args:
            component_type: Type of component (e.g., "input", "layout").
            component_name: Optional name for the component.
            id: Optional HTML ID for the component.
            classes: Optional list of CSS classes.
            attributes: Optional dict of HTML attributes.
            key: Optional unique key for the component.
            **kwargs: Additional keyword arguments.
        """
        # Set up component properties
        self.__key = key
        self.__kwargs = kwargs
        self.__component_type = component_type
        self.__component_name = component_name or self.__class__.__name__
        self.__id = id
        self.__classes = classes or []
        self.__attributes = attributes or {}

        # Set up logger
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._logger.debug(f"Initializing {self.__class__.__name__} component")

        # Automatically load CSS for this component if available
        self._load_component_css()

    @AspectUtils.catch_errors
    def _load_component_css(self) -> None:
        """Load CSS specific to this component."""
        css_path = f"{self.__component_type}/{self.__component_name.lower()}.css"
        css = resource_manager.load_css(css_path)
        if css:
            resource_manager.inject_css(css)
            self._logger.debug(f"Loaded CSS for {self.__component_name}")
        else:
            # Fallback to component type CSS
            type_css = resource_manager.load_css(f"{self.__component_type}.css")
            if type_css:
                resource_manager.inject_css(type_css)
                self._logger.debug(f"Loaded CSS for {self.__component_type}")

    @property
    def component_name(self) -> str:
        """Return the component's name."""
        return self.__component_name

    @property
    def component_type(self) -> str:
        """Return the Component type (input, card, etc.)"""
        return self.__component_type

    @property
    def classes(self) -> List[str]:
        """Get the list of CSS classes."""
        return self.__classes

    @classes.setter
    def classes(self, value: List[str]) -> None:
        """Set the CSS classes, ensuring value is a list."""
        if not isinstance(value, list):
            raise ValueError("Classes must be provided as a list")
        self.__classes = value

    @property
    def attributes(self) -> Dict[str, str]:
        """Get the HTML attributes."""
        return self.__attributes

    @attributes.setter
    def attributes(self, value: Dict[str, str]) -> None:
        """Set the HTML attributes, ensuring value is a dict."""
        if not isinstance(value, dict):
            raise ValueError("Attributes must be provided as a dict")
        self.__attributes = value

    @property
    def component_id(self) -> str:
        """Compute and return the component's unique HTML ID."""
        return (
            self.__id or f"{self.__component_type}-" f"{self.__component_name}"
        ).lower()

    @property
    def template_variables(self) -> Dict[str, Any]:
        """Return a dict of template variables for rendering."""
        return {
            "COMPONENT_TYPE": self.__component_type,
            "COMPONENT_NAME": self.__component_name,
            "COMPONENT_ID": self.component_id,
            "COMPONENT_CLASSES": " ".join(self.__classes),
            "COMPONENT_ATTRIBUTES": " ".join(
                [f'{k}="{v}"' for k, v in self.__attributes.items()]
            ),
        }

    def load_template(self, template_path: str) -> Optional[str]:
        """
        Load a template for this component.

        Args:
            template_path: Relative path to the template file.

        Returns:
            Template content as string, or None if loading failed.
        """
        return resource_manager.load_template(template_path)

    def render_template(
        self, template_path: str, context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Load and render a template with variable substitution.

        Args:
            template_path: Relative path to the template file.
            context: Dictionary of variables to substitute.

        Returns:
            Rendered template as string, or None if rendering failed.
        """
        try:
            # Combine base template variables with provided context
            full_context = self.template_variables.copy()
            if context:
                full_context.update(context)

            # Try to load the template directly
            template_content = resource_manager.load_template(template_path)

            if not template_content:
                # Try to infer template path variants
                alt_paths = []

                # If not already prefixed with components/
                if not template_path.startswith("components/"):
                    alt_paths.append(f"components/{template_path}")

                # If it might be a component type path
                parts = template_path.split("/")
                if len(parts) > 1:
                    # Try variations
                    alt_paths.append(f"{parts[0]}/{parts[-1]}")

                # Try all alternative paths
                for alt_path in alt_paths:
                    template_content = resource_manager.load_template(alt_path)
                    if template_content:
                        self._logger.debug(
                            f"Found template at alternative path: {alt_path}"
                        )
                        break

            if not template_content:
                # No template found - log warning but don't raise error
                self._logger.warning(f"Template not found: {template_path}")
                return None

            # Process template variables
            rendered = template_content

            # Replace variables
            for key, value in full_context.items():
                key = key.lower()
                key = "{" + key + "}"
                # if class name is "Sidebar"
                # if self.__component_name == "sidebar":
                # print(f"key: {key}, value: {value}")
                rendered = rendered.replace(key, str(value))  # double curly braces
                rendered = rendered.replace(f"${{{key.upper()}}}", str(value))

            return rendered
        except Exception as e:
            self._logger.error(f"Template rendering error: {str(e)}")
            # Don't raise immediately to allow fallback rendering
            return None

    @abstractmethod
    def render(self) -> str:
        """
        Render the component to HTML.

        Returns:
            HTML representation of the component.
        """
        pass

    @abstractmethod
    def display(self) -> T:
        """
        Display the component in the Streamlit app.

        Returns:
            Component-specific return value.
        """
        pass

    def handle_error(self, error: Exception) -> None:
        """
        Handle and log errors during component rendering.

        Args:
            error: The exception that occurred.
        """
        self._logger.error(
            f"Error in {self.__class__.__name__} component: {str(error)}",
            exc_info=True,
        )

        # Show error to user in development mode
        if st.session_state.get("debug_mode", False):
            st.error(f"Component Error: {str(error)}")
        else:
            st.error("An error occurred while rendering this component.")

    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Retrieve a value from Streamlit's session state.

        Args:
            key: The key to retrieve.
            default: The default value if key is not found.

        Returns:
            The value from session state or default.
        """
        return st.session_state.get(key, default)

    def set_state(self, key: str, value: Any) -> None:
        """
        Set a value in Streamlit's session state.

        Args:
            key: The key to set.
            value: The value to assign.
        """
        st.session_state[key] = value

    def has_state(self, key: str) -> bool:
        """
        Check if a key exists in Streamlit's session state.

        Args:
            key: The key to check.

        Returns:
            True if key exists, else False.
        """
        return key in st.session_state

    def add_class(self, class_name: str) -> None:
        """
        Add a CSS class to the component.

        Args:
            class_name: The CSS class name to add.
        """
        if class_name not in self.__classes:
            self.__classes.append(class_name)

    def remove_class(self, class_name: str) -> None:
        """
        Remove a CSS class from the component.

        Args:
            class_name: The CSS class name to remove.
        """
        if class_name in self.__classes:
            self.__classes.remove(class_name)

    def set_attribute(self, name: str, value: str) -> None:
        """
        Set an HTML attribute on the component.

        Args:
            name: Attribute name.
            value: Attribute value.
        """
        self.__attributes[name] = value

    def get_attribute(self, name: str, default: str = "") -> str:
        """
        Retrieve an HTML attribute's value.

        Args:
            name: Attribute name.
            default: Default value if not found.

        Returns:
            The attribute value or default.
        """
        return self.__attributes.get(name, default)

    def safe_render(self) -> str:
        """
        Safely render the component with error handling.

        Returns:
            HTML representation of the component or error message.
        """
        try:
            return self.render()
        except Exception as e:
            self.handle_error(e)
            return f'<div class="component-error">Error rendering {self.__component_name}</div>'

    def safe_display(self) -> Optional[T]:
        """
        Safely display the component with error handling.

        Returns:
            Component return value or None if an error occurred.
        """
        try:
            return self.display()
        except Exception as e:
            self.handle_error(e)
            return None

    def __str__(self) -> str:
        """String representation of the component."""
        return f"{self.__class__.__name__}(id={self.component_id})"

    def __repr__(self) -> str:
        """Detailed representation of the component."""
        return f"{self.__class__.__name__}(type={self.__component_type}, name={self.__component_name}, id={self.component_id})"
