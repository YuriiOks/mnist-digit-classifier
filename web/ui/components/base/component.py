# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/base/component.py
# Description: Base component class for UI components
# Created: 2024-05-01

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)


class Component(ABC):
    """Base class for all UI components.
    
    This abstract base class provides common functionality for all UI components,
    including error handling and a standard rendering interface.
    """
    
    def __init__(
        self,
        component_type: str = "component",
        component_name: Optional[str] = None,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize the base component.
        
        Args:
            component_type: Type of component (e.g., "input", "display", "layout")
            component_name: Optional name for the component
            id: Optional HTML ID attribute for the component
            classes: Optional list of CSS classes to apply to the component
            attributes: Optional dictionary of HTML attributes to apply to the 
                component
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(
            f"Initializing {component_type} component: "
            f"{component_name or self.__class__.__name__}"
        )
        
        self.component_type = component_type
        self.component_name = component_name or self.__class__.__name__
        self.id = id
        self.classes = classes or []
        self.attributes = attributes or {}
    
    @abstractmethod
    def display(self) -> None:
        """Display the component.
        
        This abstract method must be implemented by all component subclasses.
        """
        pass
    
    def handle_error(self, error: Exception) -> None:
        """Handle errors that occur during component rendering.
        
        Args:
            error: The exception that was raised
        """
        error_message = f"Error in {self.component_name} component: {str(error)}"
        self.logger.error(error_message, exc_info=True)
        
        # Log the error but don't display anything to the user
        # Components should handle their own error display if needed
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get information about this component.
        
        Returns:
            Dict[str, Any]: Component information
        """
        return {
            "type": self.component_type,
            "name": self.component_name,
            "class": self.__class__.__name__,
            "id": self.id,
            "classes": self.classes,
            "attributes": self.attributes
        }
    
    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for component rendering.
        
        Returns:
            Dict[str, Any]: Template variables
        """
        return {
            "COMPONENT_TYPE": self.component_type,
            "COMPONENT_NAME": self.component_name,
            "COMPONENT_ID": self.id or f"{self.component_type}-{self.component_name}".lower(),
            "COMPONENT_CLASSES": " ".join(self.classes),
            "COMPONENT_ATTRIBUTES": " ".join(
                [f'{k}="{v}"' for k, v in self.attributes.items()]
            )
        } 