# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/base_component.py
# Description: Base component class for all UI components
# Created: 2024-05-01

import streamlit as st
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class BaseComponent(ABC):
    """Base component class for all UI components.
    
    This abstract base class provides common functionality for all UI components,
    including logging, state access, and a standard rendering interface.
    """
    
    def __init__(self, key: Optional[str] = None, **kwargs):
        """Initialize a new component.
        
        Args:
            key: Optional unique key for the component
            **kwargs: Additional keyword arguments for the component
        """
        self.key = key
        self.kwargs = kwargs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Initializing {self.__class__.__name__} component")
    
    @abstractmethod
    def display(self) -> None:
        """Display the component in the Streamlit app."""
        pass
    
    def get_html(self) -> str:
        """Get the HTML representation of the component.
        
        Returns:
            str: HTML code for the component
        """
        return ""
    
    @staticmethod
    def sanitize_html_content(content: str) -> str:
        """Sanitize HTML content for safe rendering.
        
        Args:
            content: HTML content to sanitize
            
        Returns:
            str: Sanitized HTML content
        """
        # Basic sanitization - in a real app, use a proper HTML sanitizer
        content = str(content).strip()
        return content

    def handle_error(self, error: Exception) -> None:
        """Handle errors that occur during component rendering.
        
        Args:
            error: The exception that was raised
        """
        error_message = f"Error in {self.__class__.__name__} component: {str(error)}"
        self.logger.error(error_message, exc_info=True)
        
        # Log the error but don't display anything to the user
        # Components should handle their own error display if needed
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get information about this component.
        
        Returns:
            Dict[str, Any]: Component information
        """
        return {
            "type": self.__class__.__name__,
            "name": self.key,
            "class": self.__class__.__name__
        } 