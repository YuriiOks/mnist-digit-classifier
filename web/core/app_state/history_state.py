# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/core/app_state/history_state.py
# Description: State management for prediction history with database integration
# Created: 2024-05-01
# Updated: 2025-03-30

import logging
from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
import base64

from core.app_state.session_state import SessionState
from core.database.db_manager import db_manager
from utils.aspects import AspectUtils

logger = logging.getLogger(__name__)


class HistoryState:
    """Manage prediction history state with database integration."""

    HISTORY_KEY = "prediction_history"
    CURRENT_PREDICTION_KEY = "current_prediction"

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def initialize(cls) -> None:
        """Initialize history state if not already present."""
        if not SessionState.has_key(cls.HISTORY_KEY):
            # Initialize with an empty list
            SessionState.set(cls.HISTORY_KEY, [])

        if not SessionState.has_key(cls.CURRENT_PREDICTION_KEY):
            SessionState.set(cls.CURRENT_PREDICTION_KEY, None)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_predictions(
        cls,
        limit: int = 50,
        offset: int = 0,
        digit_filter: Optional[int] = None,
        min_confidence: float = 0.0,
        sort_by: str = "newest",
    ) -> List[Dict[str, Any]]:
        """
        Get prediction history from database.

        Args:
            limit: Maximum number of predictions to return
            offset: Number of predictions to skip (for pagination)
            digit_filter: Filter predictions by digit
            min_confidence: Minimum confidence threshold
            sort_by: Sorting method ("newest", "oldest", "highest_conf", "lowest_conf")

        Returns:
            List[Dict[str, Any]]: List of all stored predictions
        """
        cls.initialize()

        # Get predictions from database
        try:
            predictions = db_manager.get_predictions(
                limit=limit,
                offset=offset,
                digit_filter=digit_filter,
                min_confidence=min_confidence,
                sort_by=sort_by,
            )

            return predictions
        except Exception as e:
            logger.error(f"Error getting predictions from database: {e}")
            # Fallback to session state if database access fails
            return SessionState.get(cls.HISTORY_KEY, [])

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_latest_prediction(cls) -> Optional[Dict[str, Any]]:
        """
        Get the most recent prediction.

        Returns:
            Optional[Dict[str, Any]]: The most recent prediction or None
        """
        cls.initialize()

        # Try to get from database first (limit=1, newest first)
        try:
            predictions = db_manager.get_predictions(limit=1, sort_by="newest")
            if predictions:
                return predictions[0]
        except Exception as e:
            logger.error(f"Error getting latest prediction from database: {e}")

        # Fallback to session state
        history = SessionState.get(cls.HISTORY_KEY, [])
        if history:
            return history[-1]
        return None

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def add_prediction(
        cls,
        digit: int,
        confidence: float,
        input_type: str = "canvas",
        image_data: Optional[bytes] = None,
    ) -> Dict[str, Any]:
        """
        Add a new prediction to history and database.

        Args:
            digit: The predicted digit
            confidence: Confidence score (0-1)
            input_type: The input method used (canvas, upload, url)
            image_data: Optional image data

        Returns:
            Dict[str, Any]: The newly created prediction entry
        """
        cls.initialize()

        # Create prediction entry
        pred_id = str(uuid.uuid4())
        timestamp = datetime.now()

        prediction = {
            "id": pred_id,
            "digit": digit,
            "confidence": confidence,
            "timestamp": timestamp,
            "input_type": input_type,
            "image_data": image_data,
        }

        # Add to database
        try:
            db_manager.add_prediction(prediction)
            logger.info(f"Added prediction to database: {pred_id}")
        except Exception as e:
            logger.error(f"Error adding prediction to database: {e}")
            # Continue to add to session state even if DB fails

        # Add to history in session state
        history = SessionState.get(cls.HISTORY_KEY, [])
        history.append(prediction)
        SessionState.set(cls.HISTORY_KEY, history)

        # Update current prediction
        SessionState.set(cls.CURRENT_PREDICTION_KEY, prediction)

        return prediction

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def clear_history(cls) -> None:
        """Clear all prediction history from session and database."""
        cls.initialize()

        # Clear database
        try:
            db_manager.clear_predictions()
            logger.info("Cleared prediction history from database")
        except Exception as e:
            logger.error(f"Error clearing prediction history from database: {e}")

        # Clear session state
        SessionState.set(cls.HISTORY_KEY, [])
        SessionState.set(cls.CURRENT_PREDICTION_KEY, None)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_history(cls) -> List[Dict[str, Any]]:
        """Alias for get_predictions for backward compatibility."""
        return cls.get_predictions()

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_paginated_history(
        cls,
        page: int = 1,
        page_size: int = 10,
        digit_filter: Optional[int] = None,
        min_confidence: float = 0.0,
        sort_by: str = "newest",
    ) -> List[Dict[str, Any]]:
        """
        Get paginated prediction history.

        Args:
            page: Page number (1-based)
            page_size: Number of items per page
            digit_filter: Filter predictions by digit
            min_confidence: Minimum confidence threshold
            sort_by: Sorting method

        Returns:
            List of prediction entries for the requested page
        """
        cls.initialize()

        offset = (page - 1) * page_size

        return cls.get_predictions(
            limit=page_size,
            offset=offset,
            digit_filter=digit_filter,
            min_confidence=min_confidence,
            sort_by=sort_by,
        )

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_history_size(
        cls, digit_filter: Optional[int] = None, min_confidence: float = 0.0
    ) -> int:
        """
        Get total number of history entries.

        Args:
            digit_filter: Optional digit to filter by
            min_confidence: Minimum confidence threshold

        Returns:
            Number of entries in history
        """
        cls.initialize()

        try:
            # Get count from database
            return db_manager.count_predictions(
                digit_filter=digit_filter, min_confidence=min_confidence
            )
        except Exception as e:
            logger.error(f"Error getting prediction count from database: {e}")
            # Fallback to session state
            history = SessionState.get(cls.HISTORY_KEY, [])

            # Apply filters if needed
            if digit_filter is not None or min_confidence > 0:
                filtered_history = []
                for entry in history:
                    if digit_filter is not None and entry.get("digit") != digit_filter:
                        continue
                    if entry.get("confidence", 0) < min_confidence:
                        continue
                    filtered_history.append(entry)
                return len(filtered_history)

            return len(history)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_current_prediction(cls) -> Optional[Dict[str, Any]]:
        """
        Get current prediction result.

        Returns:
            Current prediction entry or None
        """
        cls.initialize()
        return SessionState.get(cls.CURRENT_PREDICTION_KEY)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def set_user_correction(cls, entry_id: str, correct_digit: int) -> None:
        """
        Set user correction for a prediction.

        Args:
            entry_id: ID of history entry to update
            correct_digit: The correct digit value (0-9)
        """
        cls.initialize()

        # Update in database
        try:
            db_manager.update_prediction(entry_id, {"user_correction": correct_digit})
            logger.info(
                f"Updated prediction {entry_id} with correction {correct_digit}"
            )
        except Exception as e:
            logger.error(f"Error updating prediction in database: {e}")

        # Update in session state as well
        history = SessionState.get(cls.HISTORY_KEY, [])
        for entry in history:
            if entry.get("id") == entry_id:
                entry["user_correction"] = correct_digit
                break

        SessionState.set(cls.HISTORY_KEY, history)

        # Update current prediction if it matches
        current = SessionState.get(cls.CURRENT_PREDICTION_KEY)
        if current and current.get("id") == entry_id:
            current["user_correction"] = correct_digit
            SessionState.set(cls.CURRENT_PREDICTION_KEY, current)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def delete_entry(cls, entry_id: str) -> bool:
        """
        Delete a specific history entry.

        Args:
            entry_id: ID of the entry to delete

        Returns:
            True if successful, False otherwise
        """
        cls.initialize()

        # Delete from database
        db_success = False
        try:
            db_success = db_manager.delete_prediction(entry_id)
            if db_success:
                logger.info(f"Deleted prediction {entry_id} from database")
        except Exception as e:
            logger.error(f"Error deleting prediction from database: {e}")

        # Delete from session state too
        history = SessionState.get(cls.HISTORY_KEY, [])
        updated_history = [entry for entry in history if entry.get("id") != entry_id]

        # Only update if something was removed
        if len(updated_history) < len(history):
            SessionState.set(cls.HISTORY_KEY, updated_history)
            logger.info(f"Deleted prediction {entry_id} from session state")

            # Clear current prediction if it matches
            current = SessionState.get(cls.CURRENT_PREDICTION_KEY)
            if current and current.get("id") == entry_id:
                SessionState.set(cls.CURRENT_PREDICTION_KEY, None)

            return True

        return db_success  # Return DB success if session state didn't change
