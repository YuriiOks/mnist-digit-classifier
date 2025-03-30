# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: ui/components/inputs/text_input.py
# Description: Text input component for user data entry
# Created: 2024-05-01

import streamlit as st
import re
from typing import Dict, Any, Optional, List, Union, Callable, Pattern
import uuid
import logging

from ui.components.base.component import Component

logger = logging.getLogger(__name__)


class TextInput(Component):
    """Text input component for user data entry.

    This component provides a flexible text input with styling, validation, and labels.
    """

    def __init__(
        self,
        name: str,
        *,
        label: Optional[str] = None,
        value: str = "",
        placeholder: str = "",
        help_text: Optional[str] = None,
        required: bool = False,
        disabled: bool = False,
        readonly: bool = False,
        type: str = "text",
        pattern: Optional[Union[str, Pattern]] = None,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        validation_message: Optional[str] = None,
        on_change: Optional[Callable[[str], Any]] = None,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None,
    ):
        """Initialize a text input component.

        Args:
            name: Name attribute for the input field (used for form submission).
            label: Label text for the input.
            value: Initial value of the input.
            placeholder: Placeholder text to display when the input is empty.
            help_text: Additional information to display below the input.
            required: Whether the input is required.
            disabled: Whether the input is disabled.
            readonly: Whether the input is read-only.
            type: Input type (e.g., 'text', 'email', 'password').
            pattern: Regex pattern for validation.
            min_length: Minimum length for text input.
            max_length: Maximum length for text input.
            validation_message: Custom validation message to display.
            on_change: Function to call when the input value changes.
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
        """
        logger.debug(f"Initializing TextInput component with name: {name}")
        # Generate a key for Streamlit input
        self.key = f"input_{uuid.uuid4().hex[:8]}"

        # Prepare classes
        input_classes = ["input-container"]
        if classes:
            input_classes.extend(classes)

        # Prepare attributes
        input_attributes = attributes or {}

        super().__init__(
            "inputs",
            "text_input",
            id=id or self.key,
            classes=input_classes,
            attributes=input_attributes,
        )

        self.name = name
        self.label = label
        self.value = value
        self.placeholder = placeholder
        self.help_text = help_text
        self.required = required
        self.disabled = disabled
        self.readonly = readonly
        self.type = type
        self.pattern = pattern
        self.min_length = min_length
        self.max_length = max_length
        self.validation_message = validation_message
        self.on_change = on_change

        # Initialize state
        self.error = None
        self.is_valid = True
        self.logger = logging.getLogger(
            f"{__name__}.{self.__class__.__name__}"
        )
        self.logger.debug(f"TextInput initialized with value: {self.value}")

    def validate(self, value: str) -> bool:
        """Validate the input value.

        Args:
            value: The value to validate.

        Returns:
            bool: Whether the value is valid.
        """
        # Required validation
        if self.required and not value:
            self.error = "This field is required"
            return False

        # Skip other validations if the field is empty and not required
        if not value:
            self.error = None
            return True

        # Length validation
        if self.min_length is not None and len(value) < self.min_length:
            self.error = f"Minimum length is {self.min_length} characters"
            return False

        if self.max_length is not None and len(value) > self.max_length:
            self.error = f"Maximum length is {self.max_length} characters"
            return False

        # Pattern validation
        if self.pattern:
            pattern_obj = (
                self.pattern
                if hasattr(self.pattern, "match")
                else re.compile(self.pattern)
            )
            if not pattern_obj.match(value):
                self.error = self.validation_message or "Invalid format"
                return False

        # Type-specific validation
        if self.type == "email" and not re.match(
            r"[^@]+@[^@]+\.[^@]+", value
        ):
            self.error = "Invalid email address"
            return False

        # Validation passed
        self.error = None
        return True

    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for rendering.

        Returns:
            Dict[str, Any]: Dictionary of variables for template rendering.
        """
        self.logger.debug("Getting template variables for text input")
        variables = super().get_template_variables()

        # Prepare input attributes
        input_attrs = {}
        input_attrs["name"] = self.name
        input_attrs["value"] = self.value
        input_attrs["placeholder"] = self.placeholder
        input_attrs["type"] = self.type

        if self.required:
            input_attrs["required"] = "required"
        if self.disabled:
            input_attrs["disabled"] = "disabled"
        if self.readonly:
            input_attrs["readonly"] = "readonly"
        if self.min_length is not None:
            input_attrs["minlength"] = str(self.min_length)
        if self.max_length is not None:
            input_attrs["maxlength"] = str(self.max_length)
        if isinstance(self.pattern, str):
            input_attrs["pattern"] = self.pattern

        # Convert attributes to HTML attribute string
        input_attrs_str = " ".join(
            [f'{k}="{v}"' for k, v in input_attrs.items()]
        )

        variables.update(
            {
                "INPUT_NAME": self.name,
                "INPUT_LABEL": self.label or "",
                "INPUT_ATTRIBUTES": input_attrs_str,
                "INPUT_HELP_TEXT": self.help_text or "",
                "INPUT_ERROR": self.error or "",
                "HAS_LABEL": "has-label" if self.label else "",
                "HAS_ERROR": "has-error" if self.error else "",
                "IS_REQUIRED": "required" if self.required else "",
                "IS_DISABLED": "disabled" if self.disabled else "",
            }
        )

        self.logger.debug("Template variables prepared successfully")
        return variables

    def display(self) -> str:
        """Display the input component and handle its state.

        Returns:
            str: The current value of the input.
        """
        self.logger.debug("Displaying text input component")
        try:
            # Render the HTML
            html = self.safe_render()
            st.markdown(html, unsafe_allow_html=True)

            # Create the Streamlit input to handle the actual interaction
            # This will be hidden with CSS, but will handle the state
            input_kwargs = {
                "label": "",
                "value": self.value,
                "key": self.key,
                "disabled": self.disabled,
                "label_visibility": "collapsed",
            }

            # Use the appropriate Streamlit input type
            if self.type == "password":
                new_value = st.text_input(**input_kwargs, type="password")
            elif self.type == "number":
                new_value = st.number_input(**input_kwargs, format="%d")
            else:
                new_value = st.text_input(**input_kwargs)

            # Check if value changed
            if new_value != self.value:
                self.value = new_value
                self.is_valid = self.validate(new_value)

                # Call on_change callback if provided
                if self.on_change and callable(self.on_change):
                    self.on_change(new_value)

            self.logger.debug(f"Text input value: {self.value}")
            self.logger.debug("Text input component displayed successfully")
            return self.value
        except Exception as e:
            self.logger.error(
                f"Error displaying text input component: {str(e)}",
                exc_info=True,
            )
            st.error("Error displaying text input")
            return self.value
