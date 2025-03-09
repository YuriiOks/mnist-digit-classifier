# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/components/navigation/menu.py
# Description: Menu navigation component
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Optional, List, Union, Callable, NamedTuple
import uuid

from ui.components.base.component import Component

logger = logging.getLogger(__name__)

class MenuItem(NamedTuple):
    """Menu item data for menu navigation."""
    id: str
    label: str
    icon: Optional[str] = None
    url: Optional[str] = None
    disabled: bool = False
    children: Optional[List['MenuItem']] = None


class Menu(Component):
    """Menu navigation component.
    
    This component provides a menu interface for navigating between different sections.
    """
    
    def __init__(
        self,
        items: List[MenuItem],
        *,
        active_item: Optional[str] = None,
        on_change: Optional[Callable[[str], Any]] = None,
        orientation: str = "vertical",
        compact: bool = False,
        collapsible: bool = False,
        collapsed: bool = False,
        id: Optional[str] = None,
        classes: Optional[List[str]] = None,
        attributes: Optional[Dict[str, str]] = None
    ):
        """Initialize a menu component.
        
        Args:
            items: List of MenuItem objects defining the menu.
            active_item: ID of the initially active item.
            on_change: Function to call when the active item changes.
            orientation: Menu orientation ('vertical' or 'horizontal').
            compact: Whether to use a compact display style.
            collapsible: Whether the menu can be collapsed (vertical only).
            collapsed: Whether the menu starts collapsed (if collapsible).
            id: HTML ID attribute for the component.
            classes: List of CSS classes to apply to the component.
            attributes: Dictionary of HTML attributes to apply to the component.
        """
        logger.debug(f"Initializing Menu component with {len(items)} items")
        # Generate a component ID if not provided
        component_id = id or f"menu_{uuid.uuid4().hex[:8]}"
        
        # Prepare classes
        menu_classes = ["menu", f"menu-{orientation}"]
        if compact:
            menu_classes.append("menu-compact")
        if collapsible:
            menu_classes.append("menu-collapsible")
        if collapsed and collapsible:
            menu_classes.append("menu-collapsed")
        if classes:
            menu_classes.extend(classes)
        
        super().__init__(
            "navigation",
            "menu",
            id=component_id,
            classes=menu_classes,
            attributes=attributes
        )
        
        self.items = items
        self.active_item = active_item
        self.orientation = orientation
        self.on_change = on_change
        self.collapsible = collapsible
        self.collapsed = collapsed and collapsible
        
        # Store the session state key for this menu instance
        self.state_key = f"menu_state_{component_id}"
        self.collapse_state_key = f"menu_collapse_{component_id}"
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"Menu initialized with active item: {self.active_item}")
    
    def _generate_menu_items_html(self, items: List[MenuItem], level: int = 0) -> str:
        """Generate HTML for menu items.
        
        Args:
            items: List of MenuItem objects.
            level: Current nesting level.
            
        Returns:
            str: HTML for the menu items.
        """
        if not items:
            return ""
            
        items_html = ""
        
        for item in items:
            # Determine if this item is active
            is_active = item.id == self.active_item
            active_class = "active" if is_active else ""
            disabled_class = "disabled" if item.disabled else ""
            has_children = item.children and len(item.children) > 0
            has_children_class = "has-children" if has_children else ""
            
            # Add icon if provided
            icon_html = f'<span class="menu-item-icon">{item.icon}</span>' if item.icon else ""
            
            # Determine if submenu should be expanded
            is_expanded = is_active or self._has_active_child(item)
            expanded_class = "expanded" if is_expanded else ""
            
            # Create menu item
            items_html += f"""
            <li class="menu-item level-{level} {active_class} {disabled_class} {has_children_class} {expanded_class}" 
                data-item-id="{item.id}">
                <div class="menu-item-content">
                    <a class="menu-item-link" href="{item.url or '#'}" 
                       data-item-id="{item.id}" {disabled_class and 'tabindex="-1"'}>
                        {icon_html}
                        <span class="menu-item-label">{item.label}</span>
                    </a>
                    {has_children and '<button class="menu-toggle-btn">▾</button>' or ''}
                </div>
            """
            
            # Add submenu if item has children
            if has_children:
                submenu_style = "display: block;" if is_expanded else "display: none;"
                items_html += f"""
                <ul class="submenu level-{level + 1}" style="{submenu_style}">
                    {self._generate_menu_items_html(item.children, level + 1)}
                </ul>
                """
            
            items_html += "</li>"
        
        return items_html
    
    def _has_active_child(self, item: MenuItem) -> bool:
        """Check if a menu item has an active child.
        
        Args:
            item: MenuItem to check.
            
        Returns:
            bool: True if the item has an active child, False otherwise.
        """
        if not item.children:
            return False
            
        for child in item.children:
            if child.id == self.active_item:
                return True
            if self._has_active_child(child):
                return True
                
        return False
    
    def get_template_variables(self) -> Dict[str, Any]:
        """Get template variables for rendering.
        
        Returns:
            Dict[str, Any]: Dictionary of variables for template rendering.
        """
        self.logger.debug("Getting template variables for menu")
        variables = super().get_template_variables()
        
        # Generate HTML for menu items
        items_html = self._generate_menu_items_html(self.items)
        
        # Generate toggle button for collapsible menu
        toggle_html = ""
        if self.collapsible:
            toggle_icon = "◀" if self.collapsed else "▶"
            toggle_html = f"""
            <button class="menu-collapse-toggle" title="Toggle menu">
                <span class="toggle-icon">{toggle_icon}</span>
            </button>
            """
        
        variables.update({
            "MENU_ITEMS": items_html,
            "MENU_TOGGLE": toggle_html,
            "ACTIVE_ITEM": self.active_item or "",
            "STATE_KEY": self.state_key,
            "COLLAPSE_STATE_KEY": self.collapse_state_key,
            "IS_COLLAPSED": "true" if self.collapsed else "false"
        })
        
        self.logger.debug("Template variables prepared successfully")
        return variables
    
    def _create_click_handler_js(self) -> str:
        """Create JavaScript for handling menu interactions.
        
        Returns:
            str: JavaScript code for the interaction handlers.
        """
        return f"""
        <script>
        (function() {{
            // Item click handler
            function handleItemClick(itemId, event) {{
                // Prevent default for hash links
                if (event.currentTarget.getAttribute('href') === '#') {{
                    event.preventDefault();
                }}
                
                // Find all menu items
                const items = document.querySelectorAll('#{self.id} .menu-item');
                
                // Update active state
                items.forEach(item => {{
                    if (item.dataset.itemId === itemId) {{
                        item.classList.add('active');
                    }} else {{
                        item.classList.remove('active');
                    }}
                }});
                
                // Update Streamlit state
                const widgetValue = {{value: itemId, id: "{self.state_key}"}};
                window.parent.postMessage(
                    {{type: "streamlit:setComponentValue", value: widgetValue}}, "*"
                );
            }}
            
            // Toggle submenu visibility
            function toggleSubmenu(item, event) {{
                event.stopPropagation();
                const submenu = item.querySelector('.submenu');
                const isExpanded = item.classList.contains('expanded');
                
                if (submenu) {{
                    if (isExpanded) {{
                        item.classList.remove('expanded');
                        submenu.style.display = 'none';
                    }} else {{
                        item.classList.add('expanded');
                        submenu.style.display = 'block';
                    }}
                }}
            }}
            
            // Toggle menu collapse state
            function toggleMenuCollapse() {{
                const menu = document.getElementById('{self.id}');
                const isCollapsed = menu.classList.contains('menu-collapsed');
                
                if (isCollapsed) {{
                    menu.classList.remove('menu-collapsed');
                }} else {{
                    menu.classList.add('menu-collapsed');
                }}
                
                // Update Streamlit state
                const collapseValue = {{value: !isCollapsed, id: "{self.collapse_state_key}"}};
                window.parent.postMessage(
                    {{type: "streamlit:setComponentValue", value: collapseValue}}, "*"
                );
            }}
            
            // Add click handlers to all menu items
            const links = document.querySelectorAll('#{self.id} .menu-item-link:not(.disabled)');
            links.forEach(link => {{
                link.addEventListener('click', (e) => handleItemClick(link.dataset.itemId, e));
            }});
            
            // Add click handlers to toggle buttons
            const toggleBtns = document.querySelectorAll('#{self.id} .menu-toggle-btn');
            toggleBtns.forEach(btn => {{
                btn.addEventListener('click', (e) => {{
                    const item = btn.closest('.menu-item');
                    toggleSubmenu(item, e);
                }});
            }});
            
            // Add click handler to collapse toggle
            const collapseToggle = document.querySelector('#{self.id} .menu-collapse-toggle');
            if (collapseToggle) {{
                collapseToggle.addEventListener('click', toggleMenuCollapse);
            }}
        }})();
        </script>
        """
    
    def display(self) -> str:
        """Display the menu component and handle navigation.
        
        Returns:
            str: The ID of the active menu item.
        """
        self.logger.debug("Displaying menu component")
        try:
            # Render the HTML
            html = self.safe_render()
            
            # Add JavaScript for handling menu interactions
            js = self._create_click_handler_js()
            html += js
            
            # Display the component
            st.markdown(html, unsafe_allow_html=True)
            
            # Update state based on interaction
            new_active_item = st.session_state.get(self.state_key, self.active_item)
            new_collapsed_state = st.session_state.get(self.collapse_state_key, self.collapsed)
            
            # Check if item changed
            if new_active_item != self.active_item:
                self.active_item = new_active_item
                
                # Call on_change callback if provided
                if self.on_change and callable(self.on_change):
                    self.on_change(new_active_item)
            
            # Check if collapse state changed
            if new_collapsed_state != self.collapsed:
                self.collapsed = new_collapsed_state
            
            self.logger.debug("Menu component displayed successfully")
            return self.active_item
        except Exception as e:
            self.logger.error(f"Error displaying menu component: {str(e)}", exc_info=True)
            st.error("Error displaying navigation menu")
            return self.active_item