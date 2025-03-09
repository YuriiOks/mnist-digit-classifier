# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/history_state.py
# Description: State management for prediction history
# Created: 2024-05-01

import logging
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
import base64

from core.app_state.session_state import SessionState

logger = logging.getLogger(__name__)

class HistoryState:
    """Manage prediction history state."""
    
    HISTORY_KEY = "prediction_history"
    CURRENT_PREDICTION_KEY = "current_prediction"
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize history state if not already present."""
        if not SessionState.has_key(cls.HISTORY_KEY):
            logger.debug("Initializing prediction history")
            # Initialize with an empty list
            SessionState.set(cls.HISTORY_KEY, [])
        
        if not SessionState.has_key(cls.CURRENT_PREDICTION_KEY):
            SessionState.set(cls.CURRENT_PREDICTION_KEY, None)
    
    @classmethod
    def get_predictions(cls) -> List[Dict[str, Any]]:
        """Get full prediction history.
        
        Returns:
            List[Dict[str, Any]]: List of all stored predictions
        """
        cls.initialize()
        predictions = SessionState.get(cls.HISTORY_KEY, [])
        logger.debug(f"Retrieved {len(predictions)} predictions from history")
        return predictions
    
    @classmethod
    def get_latest_prediction(cls) -> Optional[Dict[str, Any]]:
        """Get the most recent prediction.
        
        Returns:
            Optional[Dict[str, Any]]: The most recent prediction or None
        """
        cls.initialize()
        history = SessionState.get(cls.HISTORY_KEY, [])
        if history:
            return history[-1]
        return None
    
    @classmethod
    def add_prediction(cls, digit: int, confidence: float, image_data: Optional[str] = None) -> Dict[str, Any]:
        """Add a new prediction to history.
        
        Args:
            digit: The predicted digit
            confidence: Confidence score (0-1)
            image_data: Optional base64 encoded image data
            
        Returns:
            Dict[str, Any]: The newly created prediction entry
        """
        cls.initialize()
        
        # Create prediction entry
        prediction = {
            "id": str(uuid.uuid4()),
            "digit": digit,
            "confidence": confidence,
            "timestamp": datetime.now(),
            "image": image_data
        }
        
        # Add to history
        history = SessionState.get(cls.HISTORY_KEY, [])
        history.append(prediction)
        SessionState.set(cls.HISTORY_KEY, history)
        
        # Update current prediction
        SessionState.set(cls.CURRENT_PREDICTION_KEY, prediction)
        
        logger.debug(f"Added prediction for digit {digit} to history")
        return prediction
    
    @classmethod
    def clear_history(cls) -> None:
        """Clear all prediction history."""
        cls.initialize()
        SessionState.set(cls.HISTORY_KEY, [])
        SessionState.set(cls.CURRENT_PREDICTION_KEY, None)
        logger.debug("Cleared prediction history")
    
    @classmethod
    def get_history(cls) -> List[Dict[str, Any]]:
        """Alias for get_predictions for backward compatibility."""
        return cls.get_predictions()

    @classmethod
    def get_paginated_history(cls, page: int = 1, page_size: int = 10) -> List[Dict[str, Any]]:
        """Get paginated prediction history.
        
        Args:
            page: Page number (1-based)
            page_size: Number of items per page
            
        Returns:
            List of prediction entries for the requested page
        """
        cls.initialize()
        history = SessionState.get(cls.HISTORY_KEY)
        
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        
        return history[start_idx:end_idx]
    
    @classmethod
    def get_history_size(cls) -> int:
        """Get total number of history entries.
        
        Returns:
            Number of entries in history
        """
        cls.initialize()
        return len(SessionState.get(cls.HISTORY_KEY))
    
    @classmethod
    def get_current_prediction(cls) -> Optional[Dict[str, Any]]:
        """Get current prediction result.
        
        Returns:
            Current prediction entry or None
        """
        cls.initialize()
        return SessionState.get(cls.CURRENT_PREDICTION_KEY)
    
    @classmethod
    def set_user_correction(cls, entry_id: str, correct_digit: int) -> None:
        """Set user correction for a prediction.
        
        Args:
            entry_id: ID of history entry to update
            correct_digit: The correct digit value (0-9)
        """
        cls.initialize()
        
        history = SessionState.get(cls.HISTORY_KEY)
        for entry in history:
            if entry["id"] == entry_id:
                entry["user_correction"] = correct_digit
                break
                
        SessionState.set(cls.HISTORY_KEY, history)
        
        # Update current prediction if it matches
        current = SessionState.get(cls.CURRENT_PREDICTION_KEY)
        if current and current["id"] == entry_id:
            current["user_correction"] = correct_digit
            SessionState.set(cls.CURRENT_PREDICTION_KEY, current)
            
        logger.debug(f"User correction set for entry {entry_id}: {correct_digit}") 