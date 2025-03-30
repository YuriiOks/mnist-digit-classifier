# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: utils/aspects.py
# Description: Decorator utilities for cross-cutting concerns
# Created: 2025-03-16

import logging
from functools import wraps
from types import FunctionType
import streamlit as st


class AspectUtils:
    """
    Utility class providing decorators for method logging, error handling,
    and applying cross-cutting concerns uniformly across components.
    """

    @staticmethod
    def log_method(method):
        """
        Decorator for logging method entry and exit points.

        Args:
            method: The method to decorate

        Returns:
            Decorated method with logging
        """

        @wraps(method)
        def wrapper(*args, **kwargs):
            # For class methods (instance methods)
            if args and hasattr(args[0], "_logger"):
                logger = args[0]._logger
                logger.debug(f"🔹 Entering method: {method.__name__}")
                result = method(*args, **kwargs)
                logger.debug(f"✅ Exiting method: {method.__name__}")
                return result
            else:
                # For static methods or functions
                module_logger = logging.getLogger(method.__module__)
                module_logger.debug(
                    f"🔹 Entering function: {method.__name__}"
                )
                result = method(*args, **kwargs)
                module_logger.debug(f"✅ Exiting function: {method.__name__}")
                return result

        return wrapper

    @staticmethod
    def catch_errors(method):
        """
        Decorator for robust error handling.

        Args:
            method: The method to decorate

        Returns:
            Decorated method with error handling
        """

        @wraps(method)
        def wrapper(*args, **kwargs):
            try:
                return method(*args, **kwargs)
            except Exception as e:
                # For methods with handle_error
                if args and hasattr(args[0], "handle_error"):
                    args[0].handle_error(e)
                    method_name = method.__name__
                    if method_name == "render":
                        return f"<div class='error'>🚨 Error rendering component: {str(e)}</div>"
                    elif method_name == "display":
                        st.error(f"🚨 Error displaying component")
                # For methods with logger
                elif args and hasattr(args[0], "_logger"):
                    args[0]._logger.error(
                        f"Error in {method.__name__}: {str(e)}", exc_info=True
                    )
                    st.error(f"🚨 Error: {str(e)}")
                # Fallback
                else:
                    logging.error(
                        f"Error in {method.__name__}: {str(e)}", exc_info=True
                    )
                    st.error(f"🚨 Error: {str(e)}")
                return None

        return wrapper

    @classmethod
    def log_all_methods(cls, target_cls):
        """
        Class decorator to apply method logging to all methods in a class.

        Args:
            target_cls: The class whose methods will be decorated

        Returns:
            Class with all methods wrapped in log_method decorator
        """
        for attr_name in dir(target_cls):
            if attr_name.startswith("__"):
                continue
            attr = getattr(target_cls, attr_name)
            if isinstance(attr, FunctionType):
                setattr(target_cls, attr_name, cls.log_method(attr))
        return target_cls
