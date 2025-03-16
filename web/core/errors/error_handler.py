# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/errors/error_handler.py
# Description: Base error handling classes and utilities
# Created: 2025-03-16

import logging
import traceback
import sys
from typing import Optional, Any, Dict, Type, Callable
import streamlit as st

logger = logging.getLogger(__name__)

class ErrorHandler:
    """
    Base error handler class.
    
    Provides common error handling functionality for the application.
    """
    
    # Error levels
    LEVEL_INFO = "info"
    LEVEL_WARNING = "warning"
    LEVEL_ERROR = "error"
    LEVEL_CRITICAL = "critical"
    
    # Default error messages
    DEFAULT_ERROR_MESSAGE = "An unexpected error occurred."
    DEFAULT_USER_MESSAGE = "Something went wrong. Please try again later."
    
    @classmethod
    def handle_error(
        cls,
        error: Exception,
        *,
        level: str = LEVEL_ERROR,
        message: Optional[str] = None,
        user_message: Optional[str] = None,
        log_exception: bool = True,
        show_user_message: bool = True,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Handle an error with appropriate logging and user feedback.
        
        Args:
            error: The exception to handle
            level: Error level (info, warning, error, critical)
            message: Message to log (defaults to error string)
            user_message: Message to show to the user
            log_exception: Whether to log the exception traceback
            show_user_message: Whether to show a message to the user
            context: Additional context for the error
        """
        try:
            # Prepare messages
            log_message = message or str(error) or cls.DEFAULT_ERROR_MESSAGE
            display_message = user_message or str(error) or cls.DEFAULT_USER_MESSAGE
            
            # Add context to log message if provided
            if context:
                context_str = ", ".join(f"{k}={v}" for k, v in context.items())
                log_message = f"{log_message} [Context: {context_str}]"
            
            # Log according to level
            log_function = cls._get_log_function(level)
            exc_info = sys.exc_info() if log_exception else None
            
            log_function(log_message, exc_info=exc_info)
            
            # Show user message if requested
            if show_user_message:
                cls._show_user_message(level, display_message)
        except Exception as e:
            # Emergency fallback if error handler itself fails
            fallback_message = f"Error handler failed: {str(e)}"
            logger.error(fallback_message, exc_info=True)
            try:
                st.error(fallback_message)
            except:
                # Last resort if st is not available
                print(fallback_message, file=sys.stderr)
    
    @staticmethod
    def _get_log_function(level: str) -> Callable:
        """
        Get the appropriate logging function based on level.
        
        Args:
            level: Error level
            
        Returns:
            Logging function to use
        """
        if level == ErrorHandler.LEVEL_INFO:
            return logger.info
        elif level == ErrorHandler.LEVEL_WARNING:
            return logger.warning
        elif level == ErrorHandler.LEVEL_CRITICAL:
            return logger.critical
        else:
            # Default to error level
            return logger.error
    
    @staticmethod
    def _show_user_message(level: str, message: str) -> None:
        """
        Show a message to the user via Streamlit.
        
        Args:
            level: Error level
            message: Message to display
        """
        try:
            if level == ErrorHandler.LEVEL_INFO:
                st.info(message)
            elif level == ErrorHandler.LEVEL_WARNING:
                st.warning(message)
            elif level == ErrorHandler.LEVEL_CRITICAL:
                st.error(f"Critical Error: {message}")
            else:
                # Default to error level
                st.error(message)
        except Exception as e:
            # If Streamlit is not available, fallback to print
            logger.error(f"Could not display message to user: {str(e)}")
            print(f"User Message ({level}): {message}", file=sys.stderr)
    
    @classmethod
    def format_exception(cls, exc: Exception) -> str:
        """
        Format an exception into a readable string with traceback.
        
        Args:
            exc: Exception to format
            
        Returns:
            str: Formatted exception string
        """
        try:
            exc_type = type(exc).__name__
            exc_msg = str(exc)
            tb_str = "".join(traceback.format_tb(exc.__traceback__))
            return f"{exc_type}: {exc_msg}\nTraceback:\n{tb_str}"
        except Exception as e:
            logger.error(f"Error formatting exception: {str(e)}")
            return f"Error: {str(exc)}"