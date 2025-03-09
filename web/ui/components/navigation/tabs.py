# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/navigation/tabs.py
# Description: Tab navigation component
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Optional, List, Union, Callable, NamedTuple
import uuid

from ui.components.base.component import Component

logger = logging.getLogger(__name__)


class Tab(NamedTuple):
    """Tab data for tabs navigation."""
    id: str
    label: str
    icon: Optional[str] = None
    content: Optional[str] = None
    disabled: bool = False


class Tabs(Component):
    """Tab navigation component.
    
    This component provides a tabbed interface for navigating between different content sections.
    """
    
    def __init__(
        self,
        tabs: List[Tab],
        *,
        active_tab: Optional[str] = None,
        on_change: Optional[Callable[[str], Any]] = None,
        vertical: bool = False,
        pill_style: bool = False,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize a tabs component.
        
        Args:
            tabs: List of Tab objects defining the tabs.
            active_tab: ID of the initially active tab. Defaults to the first tab if None.
            on_change: Function to call when the active tab changes.
            vertical: Whether to display tabs vertically instead of horizontally.
            pill_style: Whether to use pill-style tabs instead of traditional tabs.
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
        """
        logger.debug(f"Initializing Tabs component with {len(tabs)} tabs")
        # Generate a component ID if not provided
        component_id = id or f"tabs_{uuid.uuid4().hex[:8]}"
        
        # Prepare classes
        tabs_classes = ["tabs"]
        if vertical:
            tabs_classes.append("tabs-vertical")
        if pill_style:
            tabs_classes.append("tabs-pill")
        if classes:
            tabs_classes.extend(classes)
        
        super().__init__(
            "navigation",
            "tabs",
            id=component_id,
            classes=tabs_classes,
            attributes=attributes
        )
        
        self.tabs = tabs
        self.vertical = vertical
        self.pill_style = pill_style
        self.on_change = on_change
        
        # Set the active tab (default to first tab if not specified)
        if active_tab and any(tab.id == active_tab for tab in tabs):
            self.active_tab = active_tab
        elif tabs:
            self.active_tab = tabs[0].id
        else:
            self.active_tab = None
        
        # Store the session state key for this tabs instance
        self.state_key = f"tabs_state_{component_id}"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Tabs initialized with active tab: {self.active_tab}")
    
    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for rendering.
        
        Returns:
            Dict[str, Any]: Dictionary of variables for template rendering.
        """
        self.logger.debug("Getting template variables for tabs")
        variables = super().get_template_variables()
        
        # Generate HTML for tabs
        tabs_html = ""
        content_html = ""
        
        for tab in self.tabs:
            # Skip rendering disabled tabs content
            if tab.disabled:
                continue
                
            # Determine if this tab is active
            is_active = tab.id == self.active_tab
            active_class = "active" if is_active else ""
            disabled_class = "disabled" if tab.disabled else ""
            
            # Add icon if provided
            icon_html = f'<span class="tab-icon">{tab.icon}</span>' if tab.icon else ""
            
            # Create a tab
            tabs_html += f"""
            <li class="tab {active_class} {disabled_class}" data-tab-id="{tab.id}">
                <button class="tab-button" {disabled_class and 'disabled'}>
                    {icon_html}
                    <span class="tab-label">{tab.label}</span>
                </button>
            </li>
            """
            
            # Create corresponding content
            display_style = "block" if is_active else "none"
            content_html += f"""
            <div class="tab-content {active_class}" data-tab-id="{tab.id}" style="display: {display_style}">
                {tab.content or ''}
            </div>
            """
        
        variables.update({
            "TABS_LIST": tabs_html,
            "TABS_CONTENT": content_html,
            "ACTIVE_TAB": self.active_tab or "",
            "STATE_KEY": self.state_key
        })
        
        self.logger.debug("Template variables prepared successfully")
        return variables
    
    def _create_click_handler_js(self) -> str:
        """Create JavaScript for handling tab clicks.
        
        Returns:
            str: JavaScript code for the click handler.
        """
        return f"""
        <script>
        (function() {{
            // Tab click handler
            function handleTabClick(tabId) {{
                // Find all tabs and content
                const tabs = document.querySelectorAll('#{self.id} .tab');
                const contents = document.querySelectorAll('#{self.id} .tab-content');
                
                // Update active state
                tabs.forEach(tab => {{
                    if (tab.dataset.tabId === tabId) {{
                        tab.classList.add('active');
                    }} else {{
                        tab.classList.remove('active');
                    }}
                }});
                
                // Show/hide content
                contents.forEach(content => {{
                    if (content.dataset.tabId === tabId) {{
                        content.style.display = 'block';
                        content.classList.add('active');
                    }} else {{
                        content.style.display = 'none';
                        content.classList.remove('active');
                    }}
                }});
                
                // Update Streamlit state
                const widgetValue = {{value: tabId, id: "{self.state_key}"}};
                window.parent.postMessage(
                    {{type: "streamlit:setComponentValue", value: widgetValue}}, "*"
                );
            }}
            
            // Add click handlers to all tabs
            const tabs = document.querySelectorAll('#{self.id} .tab:not(.disabled)');
            tabs.forEach(tab => {{
                tab.addEventListener('click', () => handleTabClick(tab.dataset.tabId));
            }});
        }})();
        </script>
        """
    
    def display(self) -> str:
        """Display the tabs component and handle tab changes.
        
        Returns:
            str: The ID of the active tab.
        """
        self.logger.debug("Displaying tabs component")
        try:
            # Render the HTML
            html = self.safe_render()
            
            # Add JavaScript for handling tab clicks
            js = self._create_click_handler_js()
            html += js
            
            # Display the component
            st.markdown(html, unsafe_allow_html=True)
            
            # Create a hidden widget to track state
            new_active_tab = st.session_state.get(self.state_key, self.active_tab)
            
            # Check if tab changed
            if new_active_tab != self.active_tab:
                self.active_tab = new_active_tab
                
                # Call on_change callback if provided
                if self.on_change and callable(self.on_change):
                    self.on_change(new_active_tab)
                self.logger.debug(f"Tab changed to: {new_active_tab}")
                st.rerun()
            
            self.logger.debug("Tabs component displayed successfully")
            return self.active_tab
        except Exception as e:
            self.logger.error(f"Error displaying tabs component: {str(e)}", exc_info=True)
            st.error("Error displaying tabs")
            return self.active_tab