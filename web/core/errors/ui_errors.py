# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: core/errors/ui_errors.py
# Description: UI-specific error classes
# Created: 2025-03-16

import logging
from typing import Optional, Any, Dict

from core.errors.error_handler import ErrorHandler

logger = logging.getLogger(__name__)


class UIError(Exception):
    """Base exception for UI-related errors."""

    def __init__(
        self,
        message: str,
        *,
        component: Optional[str] = None,
        view: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        error_code: Optional[str] = None,
    ):
        """
        Initialize UI error.

        Args:
            message: Error message
            component: Name of the UI component that raised the error
            view: Name of the view that raised the error
            details: Additional error details
            original_exception: Original exception if this is a wrapper
            error_code: Error code for more specific identification
        """
        self.component = component
        self.view = view
        self.details = details or {}
        self.original_exception = original_exception
        self.error_code = error_code or "UI_ERROR"

        # Format message with component and view information
        location_info = []
        if view:
            location_info.append(f"view='{view}'")
        if component:
            location_info.append(f"component='{component}'")

        location_str = ", ".join(location_info)
        full_message = (
            f"{message}"
            if not location_str
            else f"{message} [{location_str}]"
        )

        super().__init__(full_message)

    def log_error(self, level: str = ErrorHandler.LEVEL_ERROR) -> None:
        """
        Log the error with appropriate context.

        Args:
            level: Error level to log at
        """
        try:
            # Prepare context with UI-specific information
            context = {
                "component": self.component,
                "view": self.view,
                "error_code": self.error_code,
            }
            if self.details:
                context.update(self.details)

            # Use error handler to log consistently
            ErrorHandler.handle_error(
                self.original_exception or self,
                level=level,
                message=str(self),
                context=context,
                show_user_message=False,
            )
        except Exception as e:
            # Fallback if error handler fails
            logger.error(f"Failed to log UI error: {str(e)}", exc_info=True)
            logger.error(f"Original error: {str(self)}")


class TemplateError(UIError):
    """Exception for template-related UI errors."""

    def __init__(
        self,
        message: str,
        *,
        template_name: Optional[str] = None,
        template_path: Optional[str] = None,
        component: Optional[str] = None,
        view: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        error_code: Optional[str] = None,
    ):
        """
        Initialize template error.

        Args:
            message: Error message
            template_name: Name of the template
            template_path: Path to the template file
            component: Name of the UI component that raised the error
            view: Name of the view that raised the error
            details: Additional error details
            original_exception: Original exception if this is a wrapper
            error_code: Error code for more specific identification
        """
        # Add template-specific details
        template_details = {
            "template_name": template_name,
            "template_path": template_path,
        }

        # Filter out None values
        template_details = {
            k: v for k, v in template_details.items() if v is not None
        }

        # Combine with provided details
        combined_details = details or {}
        combined_details.update(template_details)

        error_code = error_code or "TEMPLATE_ERROR"
        super().__init__(
            message,
            component=component,
            view=view,
            details=combined_details,
            original_exception=original_exception,
            error_code=error_code,
        )

    def get_user_message(self) -> str:
        """
        Get a user-friendly error message.

        Returns:
            str: User-friendly error message
        """
        template_name = self.details.get("template_name", "Unknown")
        return f"Failed to render template: {template_name}"


class ComponentError(UIError):
    """Exception for component-related UI errors."""

    def __init__(
        self,
        message: str,
        *,
        component_type: Optional[str] = None,
        component_name: Optional[str] = None,
        view: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        error_code: Optional[str] = None,
    ):
        """
        Initialize component error.

        Args:
            message: Error message
            component_type: Type of component
            component_name: Name of the component
            view: Name of the view that raised the error
            details: Additional error details
            original_exception: Original exception if this is a wrapper
            error_code: Error code for more specific identification
        """
        # Create full component name
        full_component = None
        if component_type or component_name:
            parts = []
            if component_type:
                parts.append(component_type)
            if component_name:
                parts.append(component_name)
            full_component = "/".join(parts)

        # Add component-specific details
        component_details = {
            "component_type": component_type,
            "component_name": component_name,
        }

        # Filter out None values
        component_details = {
            k: v for k, v in component_details.items() if v is not None
        }

        # Combine with provided details
        combined_details = details or {}
        combined_details.update(component_details)

        error_code = error_code or "COMPONENT_ERROR"
        super().__init__(
            message,
            component=full_component,
            view=view,
            details=combined_details,
            original_exception=original_exception,
            error_code=error_code,
        )

    def get_user_message(self) -> str:
        """
        Get a user-friendly error message.

        Returns:
            str: User-friendly error message
        """
        component_type = self.details.get("component_type", "Unknown")
        component_name = self.details.get("component_name", "component")
        return f"Failed to render {component_type} {component_name}"
