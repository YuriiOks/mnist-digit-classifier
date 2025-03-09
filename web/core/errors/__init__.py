# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/errors/__init__.py
# Description: Error handling initialization for the MNIST Digit Classifier
# Created: 2024-05-01

# Import key classes to make them available at the package level
from core.errors.error_handler import ErrorHandler
from core.errors.ui_errors import UIError, TemplateError, ComponentError
from core.errors.service_errors import ServiceError, DataError, PredictionError

# Make errors package a proper Python package