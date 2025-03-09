# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/js/js_loader.py
# Description: Utilities for loading JavaScript files
# Created: 2024-05-01

import streamlit as st
from pathlib import Path
from typing import Union, Dict, Any, Optional, List
import logging
import uuid

from utils.file.path_utils import get_asset_path
from utils.file.file_loader import load_text_file, FileLoadError

logger = logging.getLogger(__name__)


class JSLoadError(Exception):
    """Exception raised for errors in JavaScript loading operations."""
    pass


def load_js_file(js_path: Union[str, Path]) -> str:
    """Load a JavaScript file from the assets/js directory.
    
    Args:
        js_path: Path to the JavaScript file, relative to the assets/js directory.
    
    Returns:
        str: The contents of the JavaScript file.
    
    Raises:
        JSLoadError: If the JavaScript file cannot be loaded.
    """
    try:
        full_path = get_asset_path("js", js_path)
        logger.debug(f"Loading JavaScript file: {full_path}")
        return load_text_file(full_path)
    except (FileLoadError, ValueError) as e:
        logger.error(f"Failed to load JavaScript file {js_path}: {str(e)}", exc_info=True)
        raise JSLoadError(f"Failed to load JavaScript file {js_path}: {str(e)}") from e

def inject_js(js_content: str) -> None:
    """Inject JavaScript content into the Streamlit app.
    
    Uses the recommended Streamlit approach for JavaScript injection.
    
    Args:
        js_content: JavaScript content to inject.
    """
    # Create a unique component key
    component_key = f"js_inject_{uuid.uuid4().hex}"
    
    # Use Streamlit's component iframe or HTML approach for better isolation
    html = f"""
    <div id="{component_key}"></div>
    <script type="text/javascript">
        (function() {{
            {js_content}
        }})();
    </script>
    """
    st.components.v1.html(html, height=0)


def load_and_inject_js(js_path: Union[str, Path]) -> None:
    """Load a JavaScript file and inject it into the Streamlit app.
    
    Args:
        js_path: Path to the JavaScript file, relative to the assets/js directory.
    
    Raises:
        JSLoadError: If the JavaScript file cannot be loaded.
    """
    try:
        js_content = load_js_file(js_path)
        inject_js(js_content)
    except JSLoadError as e:
        raise JSLoadError(f"Failed to load and inject JavaScript {js_path}: {str(e)}") from e


def create_callback_js(callback_name: str, callback_logic: str) -> str:
    """Create JavaScript code for a callback function.
    
    Args:
        callback_name: Name of the callback function.
        callback_logic: JavaScript code for the callback function body.
    
    Returns:
        str: JavaScript code defining the callback function.
    """
    return f"""
    function {callback_name}() {{
        {callback_logic}
    }}
    """


def load_js(js_path: Union[str, Path]) -> None:
    """Load and inject a JavaScript file.
    
    This is a convenience function that combines load_js_file and inject_js.
    
    Args:
        js_path: Path to the JavaScript file, relative to the assets/js directory.
    """
    try:
        js_content = load_js_file(js_path)
        inject_js(js_content)
    except JSLoadError as e:
        logging.warning(f"Failed to load JavaScript {js_path}: {str(e)}")