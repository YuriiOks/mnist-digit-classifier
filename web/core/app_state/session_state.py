# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/session_state.py
# Description: Session state management for Streamlit
# Created: 2024-05-01

import logging
import streamlit as st
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SessionState:
    """Manage session state in Streamlit.
    
    This class provides a unified interface for accessing and modifying
    Streamlit's session state, enabling consistent state management across
    the application.
    """
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize session state with defaults if needed."""
        logger.debug("Initializing session state")
        
        # Check if any required default keys need to be initialized
        required_keys = [
            # Add any required default keys here
        ]
        
        for key in required_keys:
            if key not in st.session_state:
                st.session_state[key] = None
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get value from session state with optional default.
        
        Args:
            key: State key to retrieve
            default: Default value if key doesn't exist
            
        Returns:
            The value from session state or default
        """
        return st.session_state.get(key, default)
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set value in session state.
        
        Args:
            key: State key to set
            value: Value to store
        """
        st.session_state[key] = value
    
    @classmethod
    def delete(cls, key: str) -> None:
        """Delete key from session state.
        
        Args:
            key: State key to delete
        """
        if key in st.session_state:
            del st.session_state[key]
    
    @classmethod
    def has_key(cls, key: str) -> bool:
        """Check if key exists in session state.
        
        Args:
            key: State key to check
            
        Returns:
            True if key exists, False otherwise
        """
        return key in st.session_state
    
    @classmethod
    def clear_all(cls) -> None:
        """Clear all session state data."""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        logger.debug("Cleared all session state data")
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all session state data.
        
        Returns:
            Dictionary of all session state data
        """
        return dict(st.session_state)