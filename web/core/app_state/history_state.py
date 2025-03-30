# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/core/app_state/history_state.py
# Description: State management for prediction history using db_manager
# Created: 2024-05-01
# Updated: 2025-03-30 (Refactored to use db_manager)

import logging
from typing import Dict, Any, List, Optional, Tuple
import uuid  # Keep for potential future use, though DB ID is primary now
from datetime import datetime
import base64

from core.app_state.session_state import SessionState
from core.database.db_manager import db_manager  # Use db_manager
from utils.aspects import AspectUtils

logger = logging.getLogger(__name__)


class HistoryState:
    """Manage prediction history state using db_manager."""

    # Session state keys are primarily for caching the *latest* prediction info
    # The database (via db_manager) is the main source of truth for history lists.
    CURRENT_PREDICTION_CACHE_KEY = (
        "current_prediction_cache"  # Cache for the prediction just made
    )

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def initialize(cls) -> None:
        """Initialize history state cache keys if not already present."""
        if not SessionState.has_key(cls.CURRENT_PREDICTION_CACHE_KEY):
            SessionState.set(cls.CURRENT_PREDICTION_CACHE_KEY, None)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def add_prediction(
        cls,
        digit: int,
        confidence: float,
        input_type: str = "canvas",
        image_data: Optional[str] = None,  # Expecting base64 string
    ) -> Optional[Dict[str, Any]]:
        """
        Add a new prediction via db_manager and update the current prediction cache.

        Args:
            digit: The predicted digit.
            confidence: Confidence score (0-1).
            input_type: The input method used ('canvas', 'upload', 'url').
            image_data: Optional base64 encoded image data.

        Returns:
            Dict[str, Any]: The prediction data including the new DB ID, or None on failure.
        """
        cls.initialize()

        prediction_to_log = {
            "digit": digit,
            "confidence": confidence,
            "input_type": input_type,
            "image_data": image_data,  # Pass base64 string directly
            "timestamp": datetime.now(),  # Get current time
            "true_label": None,  # Start with no correction
        }

        # Add to database via db_manager and get the integer ID
        db_id = db_manager.add_prediction(prediction_to_log)

        if db_id is not None:
            # Create the full prediction entry *with the DB ID* to cache
            full_prediction_cache = {
                "id": db_id,  # Use the integer DB ID
                "digit": digit,
                "confidence": confidence,
                "timestamp": prediction_to_log[
                    "timestamp"
                ].isoformat(),  # Store as ISO string
                "input_type": input_type,
                "image_data": image_data,  # Store base64 in cache? Optional, might be large.
                "true_label": None,  # Initial state in cache
                # Note: 'user_correction' concept replaced by 'true_label'
            }

            # Update current prediction cache in session state
            SessionState.set(cls.CURRENT_PREDICTION_CACHE_KEY, full_prediction_cache)
            logger.info(f"Prediction added with DB ID: {db_id} and cached.")
            return full_prediction_cache  # Return the cached entry with DB ID
        else:
            logger.error("Failed to add prediction via db_manager.")
            SessionState.set(
                cls.CURRENT_PREDICTION_CACHE_KEY, None
            )  # Clear cache if DB failed
            return None

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_predictions(  # Renamed from get_history to match db_manager more closely
        cls,
        limit: int = 50,
        offset: int = 0,
        digit_filter: Optional[int] = None,
        min_confidence: float = 0.0,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """
        Get prediction history directly from db_manager.

        Returns:
            List[Dict[str, Any]]: List of predictions from the database.
        """
        cls.initialize()  # Ensure keys exist, though not used for fetching list
        try:
            # Delegate directly to db_manager
            predictions = db_manager.get_predictions(
                limit=limit,
                offset=offset,
                digit_filter=digit_filter,
                min_confidence=min_confidence,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            return predictions
        except Exception as e:
            logger.error(
                f"Error getting predictions via db_manager: {e}", exc_info=True
            )
            return []  # Return empty list on failure

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_history(cls) -> List[Dict[str, Any]]:
        """Alias for get_predictions."""
        return cls.get_predictions()

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_paginated_history(
        cls,
        page: int = 1,
        page_size: int = 12,  # Default page size used in history_view
        digit_filter: Optional[int] = None,
        min_confidence: float = 0.0,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
    ) -> Tuple[List[Dict[str, Any]], int]:  # Return count as well
        """Get paginated prediction history and total count from db_manager."""
        cls.initialize()
        offset = (page - 1) * page_size
        try:
            # Fetch data and count in separate calls for simplicity here
            # Could be combined in db_manager later if performance needed it
            total_count = db_manager.count_predictions(
                digit_filter=digit_filter, min_confidence=min_confidence
            )
            predictions = db_manager.get_predictions(
                limit=page_size,
                offset=offset,
                digit_filter=digit_filter,
                min_confidence=min_confidence,
                sort_by=sort_by,
                sort_order=sort_order,
            )
            return predictions, total_count
        except Exception as e:
            logger.error(
                f"Error getting paginated history via db_manager: {e}", exc_info=True
            )
            return [], 0  # Return empty list and zero count on failure

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_history_size(
        cls, digit_filter: Optional[int] = None, min_confidence: float = 0.0
    ) -> int:
        """Get total number of history entries from db_manager."""
        cls.initialize()
        try:
            # Delegate count to db_manager
            return db_manager.count_predictions(
                digit_filter=digit_filter, min_confidence=min_confidence
            )
        except Exception as e:
            logger.error(
                f"Error getting prediction count via db_manager: {e}", exc_info=True
            )
            return 0  # Return 0 on failure

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def get_current_prediction(cls) -> Optional[Dict[str, Any]]:
        """Get the currently cached prediction result (the last one added)."""
        cls.initialize()
        return SessionState.get(cls.CURRENT_PREDICTION_CACHE_KEY)

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def set_user_correction(cls, entry_db_id: int, correct_digit: int) -> bool:
        """
        Set user correction for a prediction using db_manager.
        NOTE: This now directly updates the database via db_manager.

        Args:
            entry_db_id: The *integer* database ID of the entry to update.
            correct_digit: The correct digit value (0-9).

        Returns:
            True if update was successful, False otherwise.
        """
        cls.initialize()
        if not isinstance(entry_db_id, int):
            logger.error(
                f"Invalid entry_db_id type for correction: {type(entry_db_id)}. Expected int."
            )
            return False

        # Update in database via db_manager
        success = db_manager.update_prediction(
            entry_db_id, {"true_label": correct_digit}
        )

        if success:
            logger.info(
                f"Updated prediction DB ID {entry_db_id} with correction {correct_digit}"
            )
            # --- Optional Cache Update ---
            # If you want the *cached* current prediction to reflect the change immediately:
            current_cached = SessionState.get(cls.CURRENT_PREDICTION_CACHE_KEY)
            if current_cached and current_cached.get("id") == entry_db_id:
                current_cached["true_label"] = correct_digit
                SessionState.set(cls.CURRENT_PREDICTION_CACHE_KEY, current_cached)
            # --- End Optional Cache Update ---
        else:
            logger.error(
                f"Failed to update correction for DB ID {entry_db_id} via db_manager."
            )

        return success

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def delete_entry(cls, entry_db_id: int) -> bool:
        """
        Delete a specific history entry using db_manager.

        Args:
            entry_db_id: The *integer* database ID of the entry to delete.

        Returns:
            True if successful, False otherwise.
        """
        cls.initialize()
        if not isinstance(entry_db_id, int):
            logger.error(
                f"Invalid entry_db_id type for deletion: {type(entry_db_id)}. Expected int."
            )
            return False

        # Delete from database via db_manager
        success = db_manager.delete_prediction(entry_db_id)

        if success:
            logger.info(f"Deleted prediction DB ID {entry_db_id} via db_manager")
            # --- Optional Cache Update ---
            # Clear current prediction cache if it matches the deleted ID
            current_cached = SessionState.get(cls.CURRENT_PREDICTION_CACHE_KEY)
            if current_cached and current_cached.get("id") == entry_db_id:
                SessionState.set(cls.CURRENT_PREDICTION_CACHE_KEY, None)
            # --- End Optional Cache Update ---
        else:
            logger.error(f"Failed to delete DB ID {entry_db_id} via db_manager.")

        return success

    @classmethod
    @AspectUtils.catch_errors
    @AspectUtils.log_method
    def clear_history(cls) -> None:
        """Clear all prediction history using db_manager."""
        cls.initialize()
        success = db_manager.clear_predictions()
        if success:
            logger.info("Cleared prediction history via db_manager.")
            # Clear cache
            SessionState.set(cls.CURRENT_PREDICTION_CACHE_KEY, None)
        else:
            logger.error("Failed to clear prediction history via db_manager.")
