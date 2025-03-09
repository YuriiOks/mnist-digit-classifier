# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/controls/toggles.py
# Description: Toggle switch components for boolean settings
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Optional, List, Union, Callable
import uuid

from ui.components.base.component import Component
from ui.theme.theme_manager import ThemeManager

logger = logging.getLogger(__name__)


class Toggle(Component):
    """Toggle switch component for boolean settings.
    
    This component provides a toggle switch with on/off states and styling.
    """
    
    def __init__(
        self,
        label: str,
        *,
        value: bool = False,
        on_change: Optional[Callable[[bool], Any]] = None,
        on_text: str = "On",
        off_text: str = "Off",
        disabled: bool = False,
        show_labels: bool = True,
        size: str = "medium",
        color: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize a toggle component.
        
        Args:
            label: Accessible label for the toggle.
            value: Initial state of the toggle (True=on, False=off).
            on_change: Function to call when the toggle state changes.
            on_text: Text to display when toggle is on.
            off_text: Text to display when toggle is off.
            disabled: Whether the toggle is disabled.
            show_labels: Whether to show the on/off text labels.
            size: Toggle size ('small', 'medium', 'large').
            color: Custom color for the toggle (CSS color value).
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
        """
        logger.debug(f"Initializing Toggle component with label: {label}, value: {value}")
        # Generate a component ID if not provided
        component_id = id or f"toggle_{uuid.uuid4().hex[:8]}"
        
        # Generate a unique key for Streamlit state
        self.key = f"toggle_state_{component_id}"
        
        # Prepare classes
        toggle_classes = ["toggle-component"]
        toggle_classes.append(f"toggle-{size}")
        if disabled:
            toggle_classes.append("disabled")
        if classes:
            toggle_classes.extend(classes)
        
        super().__init__(
            "controls",
            "toggle",
            id=component_id,
            classes=toggle_classes,
            attributes=attributes
        )
        
        self.label = label
        self.value = value
        self.on_change = on_change
        self.on_text = on_text
        self.off_text = off_text
        self.disabled = disabled
        self.show_labels = show_labels
        self.size = size
        self.color = color
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Toggle initialized with value: {self.value}")
    
    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for rendering.
        
        Returns:
            Dict[str, Any]: Dictionary of variables for template rendering.
        """
        variables = super().get_template_variables()
        
        # Inline style for custom color if provided
        toggle_style = f"--toggle-active-color: {self.color};" if self.color else ""
        
        variables.update({
            "TOGGLE_LABEL": self.label,
            "TOGGLE_IS_ON": "toggle-on" if self.value else "",
            "TOGGLE_DISABLED": "true" if self.disabled else "false",
            "TOGGLE_ON_TEXT": self.on_text,
            "TOGGLE_OFF_TEXT": self.off_text,
            "TOGGLE_SHOW_LABELS": "true" if self.show_labels else "false",
            "TOGGLE_STYLE": toggle_style,
            "TOGGLE_SIZE": self.size
        })
        
        self.logger.debug("Template variables prepared successfully")
        return variables
    
    def _create_click_handler_js(self) -> str:
        """Create JavaScript for handling toggle clicks.
        
        Returns:
            str: JavaScript code for the click handler.
        """
        return f"""
        <script>
        (function() {{
            // Get elements
            const toggle = document.querySelector('#{self.id}');
            const toggleInput = toggle.querySelector('.toggle-input');
            
            // Toggle click handler
            function handleToggleClick(event) {{
                if (toggle.getAttribute('data-disabled') === 'true') {{
                    return; // Don't toggle if disabled
                }}
                
                // Toggle the state
                const isOn = toggle.classList.contains('toggle-on');
                const newState = !isOn;
                
                if (newState) {{
                    toggle.classList.add('toggle-on');
                }} else {{
                    toggle.classList.remove('toggle-on');
                }}
                
                // Update Streamlit state
                const stateValue = {{value: newState, id: "{self.key}"}};
                window.parent.postMessage(
                    {{type: "streamlit:setComponentValue", value: stateValue}}, "*"
                );
            }}
            
            // Add click handler to toggle
            toggle.addEventListener('click', handleToggleClick);
            
            // Handle keyboard events for accessibility
            toggle.addEventListener('keydown', function(e) {{
                if (e.key === ' ' || e.key === 'Enter') {{
                    e.preventDefault();
                    handleToggleClick(e);
                }}
            }});
        }})();
        </script>
        """
    
    def display(self) -> bool:
        """Display the toggle component and handle its state.
        
        Returns:
            bool: The current state of the toggle (True=on, False=off).
        """
        # Render the HTML
        html = self.safe_render()
        
        # Add JavaScript for handling toggle interaction
        js = self._create_click_handler_js()
        html += js
        
        # Display the component
        st.markdown(html, unsafe_allow_html=True)
        
        # Get and update state
        new_value = st.session_state.get(self.key, self.value)
        
        # Check if value changed
        if new_value != self.value:
            self.value = new_value
            
            # Call on_change callback if provided
            if self.on_change and callable(self.on_change):
                self.on_change(new_value)
        
        self.logger.debug(f"Toggle value: {self.value}")
        return self.value


class ThemeToggle(Toggle):
    """Theme toggle component for switching between light and dark themes."""
    
    def __init__(
        self,
        label: str = "Theme",
        *,
        value: Optional[bool] = None,
        on_change: Optional[Callable[[bool], Any]] = None,
        disabled: bool = False,
        show_mode_text: bool = True,
        light_text: str = "Light",
        dark_text: str = "Dark",
        size: str = "medium",
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize a theme toggle component.
        
        Args:
            label: Accessible label for the toggle.
            value: Initial state of the toggle (True=dark, False=light).
                  Defaults to current theme setting if None.
            on_change: Function to call when the theme changes.
            disabled: Whether the toggle is disabled.
            show_mode_text: Whether to show the light/dark mode text.
            light_text: Text to display for light mode.
            dark_text: Text to display for dark mode.
            size: Toggle size ('small', 'medium', 'large').
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
        """
        logger.debug(f"Initializing ThemeToggle component with label: {label}")
        from ui.theme.theme_manager import ThemeManager
        
        # Determine initial value based on current theme if not specified
        if value is None:
            value = ThemeManager.is_dark_mode()
        
        toggle_classes = ["theme-toggle"]
        if classes:
            toggle_classes.extend(classes)
        
        # Call parent constructor
        super().__init__(
            label,
            value=value,
            on_change=on_change,
            on_text=dark_text,
            off_text=light_text,
            disabled=disabled,
            show_labels=show_mode_text,
            size=size,
            id=id or "theme_toggle",
            classes=toggle_classes,
            attributes=attributes
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"ThemeToggle initialized with dark mode: {value}")
    
    def display(self) -> bool:
        """Display the theme toggle and handle theme changes.
        
        Returns:
            bool: The current theme state (True=dark, False=light).
        """
        from ui.theme.theme_manager import ThemeManager
        
        # Use the toggle display method
        new_state = super().display()
        
        # Update actual theme if value changed
        if new_state != ThemeManager.is_dark_mode():
            ThemeManager.toggle_theme()
            # Don't call st.rerun() here as it should be done by the parent component
        
        self.logger.debug(f"Theme toggle value: {new_state}")
        return new_state