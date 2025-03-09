# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/errors/service_errors.py
# Description: Service-related error classes
# Created: 2024-05-01

import logging
from typing import Optional, Any, Dict, Type, List

from core.errors.error_handler import ErrorHandler

logger = logging.getLogger(__name__)

class ServiceError(Exception):
    """Base exception for service-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        error_code: Optional[str] = None
    ):
        """Initialize service error.
        
        Args:
            message: Error message
            service_name: Name of the service that raised the error
            details: Additional error details
            original_exception: Original exception if this is a wrapper
            error_code: Error code for more specific identification
        """
        logger.debug(f"Creating ServiceError: {message}")
        self.service_name = service_name
        self.details = details or {}
        self.original_exception = original_exception
        self.error_code = error_code or "SERVICE_ERROR"
        
        # Format the message with service name if provided
        full_message = f"[{service_name}] {message}" if service_name else message
        super().__init__(full_message)
    
    def log_error(self, level: str = ErrorHandler.LEVEL_ERROR) -> None:
        """Log the error with appropriate context.
        
        Args:
            level: Error level to log at
        """
        logger.debug(f"Logging ServiceError at level {level}")
        try:
            # Prepare context with service-specific information
            context = {
                "service": self.service_name,
                "error_code": self.error_code
            }
            if self.details:
                context.update(self.details)
                
            # Use error handler to log consistently
            ErrorHandler.handle_error(
                self.original_exception or self,
                level=level,
                message=str(self),
                context=context,
                show_user_message=False
            )
            logger.debug("ServiceError logged successfully")
        except Exception as e:
            # Fallback if error handler fails
            logger.error(f"Failed to log service error: {str(e)}", exc_info=True)
            logger.error(f"Original error: {str(self)}")


class DataError(ServiceError):
    """Exception for data-related service errors."""
    
    def __init__(
        self,
        message: str,
        *,
        data_source: Optional[str] = None,
        data_action: Optional[str] = None,
        data_type: Optional[str] = None,
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        error_code: Optional[str] = None
    ):
        """Initialize data error.
        
        Args:
            message: Error message
            data_source: Source of the data (file, API, etc.)
            data_action: Action being performed (read, write, etc.)
            data_type: Type of data being processed
            service_name: Name of the service that raised the error
            details: Additional error details
            original_exception: Original exception if this is a wrapper
            error_code: Error code for more specific identification
        """
        logger.debug(f"Creating DataError: {message}")
        # Add data-specific details
        data_details = {
            "data_source": data_source,
            "data_action": data_action,
            "data_type": data_type
        }
        
        # Filter out None values
        data_details = {k: v for k, v in data_details.items() if v is not None}
        
        # Combine with provided details
        combined_details = details or {}
        combined_details.update(data_details)
        
        error_code = error_code or "DATA_ERROR"
        super().__init__(
            message,
            service_name=service_name,
            details=combined_details,
            original_exception=original_exception,
            error_code=error_code
        )


class ModelError(ServiceError):
    """Exception for machine learning model-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        model_name: Optional[str] = None,
        operation: Optional[str] = None,
        input_shape: Optional[Any] = None,
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        error_code: Optional[str] = None
    ):
        """Initialize model error.
        
        Args:
            message: Error message
            model_name: Name of the model
            operation: Operation being performed (predict, train, etc.)
            input_shape: Shape of the input data
            service_name: Name of the service that raised the error
            details: Additional error details
            original_exception: Original exception if this is a wrapper
            error_code: Error code for more specific identification
        """
        logger.debug(f"Creating ModelError: {message}")
        # Add model-specific details
        model_details = {
            "model_name": model_name,
            "operation": operation,
            "input_shape": str(input_shape) if input_shape is not None else None
        }
        
        # Filter out None values
        model_details = {k: v for k, v in model_details.items() if v is not None}
        
        # Combine with provided details
        combined_details = details or {}
        combined_details.update(model_details)
        
        error_code = error_code or "MODEL_ERROR"
        super().__init__(
            message,
            service_name=service_name,
            details=combined_details,
            original_exception=original_exception,
            error_code=error_code
        )


class ConfigError(ServiceError):
    """Exception for configuration-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        config_source: Optional[str] = None,
        config_key: Optional[str] = None,
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        error_code: Optional[str] = None
    ):
        """Initialize configuration error.
        
        Args:
            message: Error message
            config_source: Source of the configuration
            config_key: Specific configuration key that caused the error
            service_name: Name of the service that raised the error
            details: Additional error details
            original_exception: Original exception if this is a wrapper
            error_code: Error code for more specific identification
        """
        logger.debug(f"Creating ConfigError: {message}")
        # Add config-specific details
        config_details = {
            "config_source": config_source,
            "config_key": config_key
        }
        
        # Filter out None values
        config_details = {k: v for k, v in config_details.items() if v is not None}
        
        # Combine with provided details
        combined_details = details or {}
        combined_details.update(config_details)
        
        error_code = error_code or "CONFIG_ERROR"
        super().__init__(
            message,
            service_name=service_name,
            details=combined_details,
            original_exception=original_exception,
            error_code=error_code
        ) 

class PredictionError(ServiceError):
    """Exception for prediction-related errors."""
    
    def __init__(
        self,
        message: str,
        *,
        prediction_type: Optional[str] = None,
        input_data: Optional[Any] = None,
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
        error_code: Optional[str] = None
    ):
        """Initialize prediction error.
        
        Args:
            message: Error message
            prediction_type: Type of prediction being attempted
            input_data: Description of the input data that caused the error
            service_name: Name of the service that raised the error
            details: Additional error details
            original_exception: Original exception if this is a wrapper
            error_code: Error code for more specific identification
        """
        # Add prediction-specific details
        prediction_details = {
            "prediction_type": prediction_type,
            "input_data": str(input_data) if input_data is not None else None
        }
        
        # Filter out None values
        prediction_details = {k: v for k, v in prediction_details.items() if v is not None}
        
        # Combine with provided details
        combined_details = details or {}
        combined_details.update(prediction_details)
        
        error_code = error_code or "PREDICTION_ERROR"
        super().__init__(
            message,
            service_name=service_name,
            details=combined_details,
            original_exception=original_exception,
            error_code=error_code
        )