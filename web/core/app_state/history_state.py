# MNIST Digit Classifier
# Copyright (c) 2025
# File: core/app_state/history_state.py
# Description: Manages prediction history state
# Created: 2024-05-01

import logging
from typing import List, Dict, Any, Optional
import time
import uuid
import numpy as np
from datetime import datetime

from core.app_state.session_state import SessionState

logger = logging.getLogger(__name__)

class HistoryState:
    """Manages prediction history state."""
    
    # State keys
    HISTORY_KEY = "_prediction_history"
    
    # Default values
    DEFAULT_MAX_HISTORY = 50
    
    @classmethod
    def initialize(cls) -> None:
        """Initialize history state if not already initialized."""
        logger.debug("Initializing history state")
        try:
            if not SessionState.has_key(cls.HISTORY_KEY):
                logger.debug("Setting empty prediction history")
                SessionState.set(cls.HISTORY_KEY, [])
            logger.debug("History state initialization complete")
        except Exception as e:
            logger.error(f"Error initializing history state: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_history(cls) -> List[Dict[str, Any]]:
        """Get the prediction history.
        
        Returns:
            List[Dict[str, Any]]: List of prediction entries
        """
        logger.debug("Getting prediction history")
        try:
            cls.initialize()
            history = SessionState.get(cls.HISTORY_KEY, [])
            logger.debug(f"Retrieved prediction history with {len(history)} entries")
            return history
        except Exception as e:
            logger.error(f"Error getting prediction history: {str(e)}", exc_info=True)
            return []
    
    @classmethod
    def add_prediction(
        cls,
        image: np.ndarray,
        prediction: int,
        confidence: float,
        all_confidences: Optional[Dict[int, float]] = None,
        max_history: Optional[int] = None
    ) -> str:
        """Add a prediction to the history.
        
        Args:
            image: Image as numpy array
            prediction: Predicted digit
            confidence: Confidence score (0-1)
            all_confidences: Confidence scores for all classes
            max_history: Maximum number of history entries
            
        Returns:
            str: ID of the added prediction
        """
        logger.debug(f"Adding prediction: {prediction} with confidence {confidence:.4f}")
        try:
            cls.initialize()
            
            # Generate a unique ID for this prediction
            prediction_id = str(uuid.uuid4())
            
            # Create prediction entry
            entry = {
                "id": prediction_id,
                "timestamp": datetime.now().isoformat(),
                "unix_time": time.time(),
                "image": image.copy() if image is not None else None,
                "prediction": prediction,
                "confidence": confidence,
                "all_confidences": all_confidences or {},
            }
            
            # Get current history
            history = cls.get_history()
            
            # Add new entry
            history.append(entry)
            
            # Trim history if needed
            max_entries = max_history or cls.DEFAULT_MAX_HISTORY
            if len(history) > max_entries:
                logger.debug(f"Trimming history to {max_entries} entries")
                history = history[-max_entries:]
            
            # Save updated history
            SessionState.set(cls.HISTORY_KEY, history)
            
            logger.info(f"Added prediction {prediction_id} to history, now {len(history)} entries")
            return prediction_id
        except Exception as e:
            logger.error(f"Error adding prediction to history: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def get_prediction(cls, prediction_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific prediction by ID.
        
        Args:
            prediction_id: ID of the prediction to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: Prediction entry or None if not found
        """
        logger.debug(f"Getting prediction with ID: {prediction_id}")
        try:
            history = cls.get_history()
            
            # Find prediction by ID
            for entry in history:
                if entry.get("id") == prediction_id:
                    logger.debug(f"Found prediction {prediction_id}")
                    return entry
            
            logger.warning(f"Prediction {prediction_id} not found in history")
            return None
        except Exception as e:
            logger.error(f"Error getting prediction {prediction_id}: {str(e)}", exc_info=True)
            return None
    
    @classmethod
    def clear_history(cls) -> None:
        """Clear the prediction history."""
        logger.debug("Clearing prediction history")
        try:
            cls.initialize()
            SessionState.set(cls.HISTORY_KEY, [])
            logger.info("Prediction history cleared")
        except Exception as e:
            logger.error(f"Error clearing prediction history: {str(e)}", exc_info=True)
            raise
    
    @classmethod
    def delete_prediction(cls, prediction_id: str) -> bool:
        """Delete a specific prediction by ID.
        
        Args:
            prediction_id: ID of the prediction to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        logger.debug(f"Deleting prediction with ID: {prediction_id}")
        try:
            history = cls.get_history()
            
            # Find and remove prediction
            for i, entry in enumerate(history):
                if entry.get("id") == prediction_id:
                    del history[i]
                    
                    # Save updated history
                    SessionState.set(cls.HISTORY_KEY, history)
                    logger.info(f"Deleted prediction {prediction_id} from history")
                    return True
            
            logger.warning(f"Prediction {prediction_id} not found, nothing deleted")
            return False
        except Exception as e:
            logger.error(f"Error deleting prediction {prediction_id}: {str(e)}", exc_info=True)
            return False
    
    @classmethod
    def get_statistics(cls) -> Dict[str, Any]:
        """Get statistics about the prediction history.
        
        Returns:
            Dict[str, Any]: Dictionary of statistics
        """
        logger.debug("Getting prediction history statistics")
        try:
            history = cls.get_history()
            
            if not history:
                logger.debug("History is empty, returning empty statistics")
                return {
                    "count": 0,
                    "classes": {},
                    "avg_confidence": 0,
                    "max_confidence": 0,
                    "min_confidence": 0
                }
            
            # Count by class
            classes = {}
            confidences = []
            
            for entry in history:
                prediction = entry.get("prediction")
                confidence = entry.get("confidence", 0)
                
                # Count by class
                if prediction is not None:
                    classes[prediction] = classes.get(prediction, 0) + 1
                
                # Track confidences
                if confidence is not None:
                    confidences.append(confidence)
            
            # Calculate statistics
            stats = {
                "count": len(history),
                "classes": classes,
                "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
                "max_confidence": max(confidences) if confidences else 0,
                "min_confidence": min(confidences) if confidences else 0,
                "recent_predictions": [entry.get("prediction") for entry in history[-5:]]
            }
            
            logger.debug(f"Calculated history statistics: {len(history)} entries, {len(classes)} classes")
            return stats
        except Exception as e:
            logger.error(f"Error getting prediction history statistics: {str(e)}", exc_info=True)
            return {
                "count": 0,
                "classes": {},
                "avg_confidence": 0,
                "max_confidence": 0,
                "min_confidence": 0,
                "error": str(e)
            } 