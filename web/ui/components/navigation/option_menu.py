# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/ui/components/navigation/option_menu.py
# Description: [Brief description of the file's purpose]
# Created: 2025-03-10
# Updated: 2025-03-30

import streamlit as st
from streamlit_option_menu import option_menu
from typing import List, Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def create_option_menu(
    menu_title: str,
    options: List[str],
    icons: Optional[List[str]] = None,
    default_index: int = 0,
    orientation: str = "vertical",
    on_change: Optional[Callable] = None,
    key: Optional[str] = None,
):
    """Create a nicer option menu with streamlit-option-menu.

    Args:
        menu_title: Title of the menu
        options: List of menu options
        icons: List of icons for each option (must match length of options)
        default_index: Default selected index
        orientation: "vertical" or "horizontal"
        on_change: Optional callback when selection changes
        key: Unique key for the component

    Returns:
        str: The selected option
    """
    try:
        logger.debug(f"Creating option menu with {len(options)} options")

        # Default key if not provided
        if key is None:
            key = f"option_menu_{menu_title.lower().replace(' ', '_')}"

        # Use the option_menu component
        selected = option_menu(
            menu_title=menu_title,
            options=options,
            icons=icons,
            default_index=default_index,
            orientation=orientation,
            key=key,
        )

        # Call on_change if provided and selection changed
        if on_change and "previous_selection" in st.session_state:
            if st.session_state["previous_selection"] != selected:
                on_change(selected)

        # Store current selection for next comparison
        st.session_state["previous_selection"] = selected

        logger.debug(f"Selected option: {selected}")
        return selected

    except Exception as e:
        logger.error(f"Error creating option menu: {str(e)}")
        # Fallback to basic selectbox
        return st.selectbox(
            menu_title, options, index=default_index, key=f"fallback_{key}"
        )
