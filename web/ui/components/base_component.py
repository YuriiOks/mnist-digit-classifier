# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/ui/components/base_component.py
# Description: Base component class for all UI components
# Created: 2024-05-01
# Updated: 2025-03-30

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
            "class": self.__class__.__name__,
        }
