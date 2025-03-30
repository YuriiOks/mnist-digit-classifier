# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/core/errors/__init__.py
# Description: Error handling initialization
# Created: 2025-03-16
# Updated: 2025-03-30

"""Error handling for the MNIST Digit Classifier."""

from core.errors.error_handler import ErrorHandler
from core.errors.ui_errors import UIError, TemplateError, ComponentError

__all__ = ["ErrorHandler", "UIError", "TemplateError", "ComponentError"]
