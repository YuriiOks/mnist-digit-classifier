# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/navigation/tabs.py
# Description: Tab navigation component
# Created: 2024-05-01

import streamlit as st
import logging
import uuid
from typing import Dict, Any, Optional, List, Tuple

from ui.components.base.component import Component
from utils.aspects import AspectUtils

logger = logging.getLogger(__name__)


class Tab:
    """Tab data for tabs navigation."""

    def __init__(
        self,
        id: str,
        label: str,
        icon: Optional[str] = None,
        content: Optional[str] = None,
        disabled: bool = False
    ):
        self.__id = id
        self.__label = label
        self.__icon = icon
        self.__content = content
        self.__disabled = disabled


class Tabs(Component):
    """Tab navigation component.

    Provides a tabbed interface for navigating between
    different contents.
    """

    def __init__(
        self,
        tabs: List[Tab],
        active_tab: Optional[str] = None,
        on_change: Optional[callable] = None,
        *,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None,
        vertical: bool = False,
        pill_style: bool = False,
        template_loader: Optional[Any] = None
    ):
        """
        Initialize the tabs component.

        Args:
          tabs: List of Tab objects.
          active_tab: ID of the initially active tab.
          on_change: Callback when active tab changes.
          id: HTML ID.
          classes: CSS classes.
          attributes: HTML attributes.
          vertical: Display tabs vertically.
          pill_style: Use pill-style tabs.
          template_loader: Optional TemplateLoader.
        """
        self.vertical = vertical
        self.pill_style = pill_style

        comp_id = id or f"tabs_{uuid.uuid4()}"
        # Prepare classes list.
        tabs_cls = classes or []
        if vertical:
            tabs_cls.append("vertical-tabs")
        else:
            tabs_cls.append("horizontal-tabs")

        super().__init__(
            component_type="navigation",
            component_name="tabs",
            id=comp_id,
            classes=tabs_cls,
            attributes=attributes
        )

        self.__tabs = tabs
        self.__active_tab = active_tab or tabs[0].id
        self.__on_change = on_change
        self.__state_key = f"tabs_state_{comp_id}"

        self.logger.debug(
            f"Tabs initialized with active tab: {self.__active_tab}"
        )

    @property
    def active_tab(self) -> str:
        """Return the current active tab ID."""
        return self.__active_tab
    
    @property
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def template_variables(self) -> Dict[str, Any]:
        """
        Generate template variables for rendering.
        
        Returns:
            Dict[str, Any]: Dictionary of variables for template rendering.
        """

        base_vars = super().template_variables.copy()
        tabs_html = ""
        contents_html = ""
        for tab in self.__tabs:
            is_active = tab.id == self.__active_tab
            active_cls = "active" if is_active else ""
            disabled_attr = "disabled" if tab.disabled else ""
            tabs_html += (
                f'<li class="tab {active_cls}" data-tab-id="{tab.id}">'
                f'<button class="tab-button" {disabled_attr}>'
                f'{tab.icon or ""}<span>{tab.label}</span>'
                f'</button></li>'
            )
            contents_html += (
                f'<div class="tab-content '
                f'{"active" if is_active else "hidden"}" '
                f'data-tab-id="{tab.id}">'
                f'{tab.content or ""}</div>'
            )
        base_vars.update({
            "TABS_HTML": tabs_html,
            "CONTENTS_HTML": contents_html,
            "ACTIVE_TAB": self.__active_tab,
            "STATE_KEY": self.__state_key
        })
        return base_vars

    def _create_click_handler_js(self) -> str:
        """
        Create and return JS code for handling tab clicks.

        This function returns a script that:
        - Finds all tabs and content areas using the component_id.
        - Updates the active state and toggles content display.
        - Sends the selected tab ID to Streamlit via postMessage.
        """
        return (
            f"<script>\n"
            f"(function() {{\n"
            f"  function handleTabClick(tabId) {{\n"
            f"    const tabs = document.querySelectorAll('#{self.component_id} "
            f".tab');\n"
            f"    const contents = document.querySelectorAll('#{self.component_id} "
            f".tab-content');\n"
            f"    tabs.forEach(tab => {{\n"
            f"      if (tab.dataset.tabId === tabId) {{\n"
            f"        tab.classList.add('active');\n"
            f"      }} else {{\n"
            f"        tab.classList.remove('active');\n"
            f"      }}\n"
            f"    }});\n"
            f"    contents.forEach(content => {{\n"
            f"      if (content.dataset.tabId === tabId) {{\n"
            f"        content.style.display = 'block';\n"
            f"        content.classList.add('active');\n"
            f"      }} else {{\n"
            f"        content.style.display = 'none';\n"
            f"        content.classList.remove('active');\n"
            f"      }}\n"
            f"    }});\n"
            f"    const widgetValue = {{value: tabId, id: \"{self.__state_key}\"}};\n"
            f"    window.parent.postMessage({{type: "
            f"\"streamlit:setComponentValue\", value: widgetValue}}, \"*\");\n"
            f"  }}\n"
            f"  const tabs = document.querySelectorAll('#{self.component_id} "
            f".tab:not(.disabled)');\n"
            f"  tabs.forEach(tab => {{\n"
            f"    tab.addEventListener('click', () => \n"
            f"      handleTabClick(tab.dataset.tabId));\n"
            f"  }});\n"
            f"}})();\n"
            f"</script>"
        )

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def display(self) -> str:
        """
        Render and display the tabs component.
        """
        html = self.template_loader.render_template(
            "components/navigation/tabs.html",
            self.template_variables
        )
        st.components.v1.html(html, height=300)

        # Update active tab from session state.
        curr = st.session_state.get(
            self.component_id, self.__active_tab
        )
        if curr:
            if curr != self.__active_tab:
                self.__active_tab = curr
                if self.__on_change:
                    self.__on_change(self.__active_tab)
        return self.__active_tab



class InputTabs(Tabs):
    """Specialized tabs for input selection: Draw, Upload, URL."""

    def __init__(
        self,
        default_tab: Optional[str] = None,
        on_change: Optional[callable] = None,
        *,
        key: str = "input_tabs"
    ):
        """
        Initialize input selection tabs.

        Args:
          default_tab: The default tab ID.
          on_change: Callback when tab selection changes.
          key: Unique key for the component.
        """
        input_tabs = [
            Tab("draw", "Draw"),
            Tab("upload", "Upload"),
            Tab("url", "URL")
        ]
        super().__init__(
            tabs=input_tabs,
            active_tab=default_tab or input_tabs[0].id,
            on_change=on_change,
            id=key
        )

    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def render(self) -> Tuple[str, List[st._DeltaGenerator]]:
        """
        Render the specialized input tabs.

        Returns:
          Tuple of selected tab ID and list of containers.
        """
        selected = st.radio(
            "Select Input Method",
            options=[tab.label for tab in self.__tabs],
            key=self.component_id,
            index=next(
                (i for i, tab in enumerate(self.__tabs)
                 if tab.id == self.__active_tab), 0
            ),
            horizontal=True
        )
        self.__active_tab = next(
            (tab.id for tab in self.__tabs 
             if tab.label == selected),
            self.__tabs[0].id
        )
        if self.__on_change:
            self.__on_change(self.__active_tab)
        conts = [st.container() for _ in self.__tabs]
        return self.__active_tab, conts