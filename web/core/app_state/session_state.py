# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/session_state.py
# Description: Session state management
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

logger = logging.getLogger(__name__)

class SessionState:
    """Provides a consistent interface for accessing Streamlit session state."""
    
    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """Get a value from session state.
        
        Args:
            key: Key to get
            default: Default value if key doesn't exist
            
        Returns:
            Any: Value from session state or default
        """
        logger.debug(f"Getting session state key: {key}")
        try:
            if key in st.session_state:
                value = st.session_state[key]
                logger.debug(f"Retrieved value for key '{key}': {type(value).__name__}")
                return value
            
            logger.debug(f"Key '{key}' not found in session state, returning default")
            return default
        except Exception as e:
            logger.error(f"Error getting session state key '{key}': {str(e)}", exc_info=True)
            return default
    
    @classmethod
    def set(cls, key: str, value: Any) -> None:
        """Set a value in session state.
        
        Args:
            key: Key to set
            value: Value to store
        """
        logger.debug(f"Setting session state key: {key}")
        try:
            st.session_state[key] = value
            logger.debug(f"Set value for key '{key}': {type(value).__name__}")
        except Exception as e:
            logger.error(f"Error setting session state key '{key}': {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def delete(cls, key: str) -> bool:
        """Delete a key from session state.
        
        Args:
            key: Key to delete
            
        Returns:
            bool: True if key was deleted, False if key didn't exist
        """
        logger.debug(f"Deleting session state key: {key}")
        try:
            if key in st.session_state:
                del st.session_state[key]
                logger.debug(f"Deleted key '{key}' from session state")
                return True
            
            logger.debug(f"Key '{key}' not found in session state, nothing to delete")
            return False
        except Exception as e:
            logger.error(f"Error deleting session state key '{key}': {str(e)}", exc_info=True)
            return False
    
    @classmethod
    def has_key(cls, key: str) -> bool:
        """Check if a key exists in session state.
        
        Args:
            key: Key to check
            
        Returns:
            bool: True if key exists, False otherwise
        """
        logger.debug(f"Checking if session state has key: {key}")
        try:
            exists = key in st.session_state
            logger.debug(f"Key '{key}' exists in session state: {exists}")
            return exists
        except Exception as e:
            logger.error(f"Error checking if session state has key '{key}': {str(e)}", exc_info=True)
            return False
    
    @classmethod
    def get_all(cls) -> Dict[str, Any]:
        """Get all session state key-value pairs.
        
        Returns:
            Dict[str, Any]: All session state values
        """
        logger.debug("Getting all session state values")
        try:
            # Convert to dict to avoid modifying the original during iteration
            state_dict = dict(st.session_state)
            logger.debug(f"Retrieved {len(state_dict)} session state values")
            return state_dict
        except Exception as e:
            logger.error(f"Error getting all session state values: {str(e)}", exc_info=True)
            return {}
    
    @classmethod
    def clear(cls) -> None:
        """Clear all session state values."""
        logger.debug("Clearing all session state values")
        try:
            # Get keys to avoid modifying during iteration
            keys = list(st.session_state.keys())
            
            for key in keys:
                del st.session_state[key]
                
            logger.info(f"Cleared {len(keys)} keys from session state")
        except Exception as e:
            logger.error(f"Error clearing session state: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize the session state.
        
        This method follows Streamlit's best practices for session state
        initialization.
        """
        logger.debug("Initializing session state")
        try:
            # Initialize default values if they don't exist
            if "initialized" not in st.session_state:
                st.session_state.initialized = True
                # Set other default values here
                
            logger.info("Session state initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing session state: {str(e)}", exc_info=True)