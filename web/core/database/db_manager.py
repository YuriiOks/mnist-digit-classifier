# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/core/database/db_manager.py
# Description: Thread-safe database manager for SQLite database operations
# Created: 2025-03-24
# Updated: 2025-03-30

import os
import sqlite3
import json
import logging
import uuid
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Thread-safe SQLite database manager for the MNIST Digit Classifier.
    Handles all database operations for the application.
    """

    # Singleton instance
    _instance = None

    # Thread-local storage for connections
    _local = threading.local()

    def __new__(cls):
        """Implement singleton pattern for DatabaseManager."""
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the database manager."""
        # Skip initialization if already done
        if getattr(self, "_initialized", False):
            return

        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Initialize database path
        self._db_dir = self._get_db_directory()
        self._db_path = os.path.join(self._db_dir, "mnist_app.db")

        # Ensure database directory exists
        os.makedirs(self._db_dir, exist_ok=True)

        # Initialize connection and tables
        self._connect()
        self._create_tables()

        self._initialized = True
        self._logger.info(f"DatabaseManager initialized with DB path: {self._db_path}")

    def _get_db_directory(self) -> str:
        """
        Get the directory where the database should be stored.

        Returns:
            str: Path to database directory
        """
        # In Docker, use /app/data
        docker_path = Path("/app/data")
        if docker_path.exists():
            return str(docker_path)

        # Otherwise use ~/.mnist_app
        home_dir = os.path.expanduser("~")
        data_dir = os.path.join(home_dir, ".mnist_app")
        return data_dir

    def _get_connection(self):
        """Get a thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            try:
                self._local.conn = sqlite3.connect(self._db_path)
                # Enable foreign keys
                self._local.conn.execute("PRAGMA foreign_keys = ON")
                # Enable extended result codes
                self._local.conn.execute("PRAGMA extra_result_codes = ON")
                # Allow JSON functions
                try:
                    self._local.conn.enable_load_extension(True)
                except:
                    pass  # Not all SQLite builds support extensions

                # Set row factory to return dictionaries
                self._local.conn.row_factory = sqlite3.Row

                self._logger.debug(
                    f"Created new connection for thread {threading.get_ident()}"
                )
            except sqlite3.Error as e:
                self._logger.error(f"Error creating connection: {e}")
                raise
        return self._local.conn

    def _get_cursor(self):
        """Get a cursor from the thread-local connection."""
        conn = self._get_connection()
        return conn.cursor()

    def _connect(self) -> None:
        """Connect to the SQLite database - just initialize the connection."""
        self._get_connection()

    def _create_tables(self) -> None:
        """Create necessary tables if they don't exist."""
        cursor = self._get_cursor()
        try:
            # Predictions table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                digit INTEGER NOT NULL,
                confidence REAL NOT NULL,
                timestamp DATETIME NOT NULL,
                input_type TEXT NOT NULL,
                image_data BLOB,
                metadata TEXT,
                user_correction INTEGER
            )
            """
            )

            # Settings table
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS settings (
                category TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                PRIMARY KEY (category, key)
            )
            """
            )

            self._get_connection().commit()
            self._logger.debug("Tables created successfully")
        except sqlite3.Error as e:
            self._logger.error(f"Error creating tables: {e}")
            raise

    def close(self) -> None:
        """Close the database connection for the current thread."""
        if hasattr(self._local, "conn") and self._local.conn:
            self._local.conn.close()
            self._local.conn = None
            self._logger.debug(f"Closed connection for thread {threading.get_ident()}")

    def add_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """
        Add a prediction to the database.

        Args:
            prediction_data: Dictionary containing prediction data

        Returns:
            str: ID of the inserted prediction
        """
        cursor = self._get_cursor()
        try:
            # Generate ID if not provided
            pred_id = prediction_data.get("id") or str(uuid.uuid4())

            # Prepare metadata as JSON if present
            metadata = prediction_data.get("metadata")
            if metadata and isinstance(metadata, dict):
                metadata = json.dumps(metadata)

            # Extract image data if present
            image_data = prediction_data.get("image_data")

            # Convert timestamp to string if it's a datetime object
            timestamp = prediction_data.get("timestamp")
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            elif not timestamp:
                timestamp = datetime.now().isoformat()

            cursor.execute(
                """
            INSERT INTO predictions (
                id, digit, confidence, timestamp, input_type, 
                image_data, metadata, user_correction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    pred_id,
                    prediction_data.get("digit", 0),
                    prediction_data.get("confidence", 0.0),
                    timestamp,
                    prediction_data.get("input_type", "unknown"),
                    image_data,
                    metadata,
                    prediction_data.get("user_correction"),
                ),
            )

            self._get_connection().commit()
            self._logger.debug(f"Added prediction with ID: {pred_id}")
            return pred_id
        except sqlite3.Error as e:
            self._logger.error(f"Error adding prediction: {e}")
            self._get_connection().rollback()
            raise

    def get_prediction(self, prediction_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a prediction by ID.

        Args:
            prediction_id: ID of the prediction to retrieve

        Returns:
            Dict containing prediction data or None if not found
        """
        cursor = self._get_cursor()
        try:
            cursor.execute(
                """
            SELECT id, digit, confidence, timestamp, input_type, 
                   image_data, metadata, user_correction
            FROM predictions
            WHERE id = ?
            """,
                (prediction_id,),
            )

            row = cursor.fetchone()
            if not row:
                return None

            prediction = dict(row)

            # Parse metadata JSON if present
            if prediction.get("metadata"):
                try:
                    prediction["metadata"] = json.loads(prediction["metadata"])
                except json.JSONDecodeError:
                    pass

            return prediction
        except sqlite3.Error as e:
            self._logger.error(f"Error getting prediction {prediction_id}: {e}")
            return None

    def get_predictions(
        self,
        limit: int = 50,
        offset: int = 0,
        digit_filter: Optional[int] = None,
        min_confidence: float = 0.0,
        sort_by: str = "newest",
    ) -> List[Dict[str, Any]]:
        """
        Get predictions with optional filtering and sorting.

        Args:
            limit: Maximum number of predictions to return
            offset: Number of predictions to skip (for pagination)
            digit_filter: Filter predictions by digit
            min_confidence: Minimum confidence threshold
            sort_by: Sorting method ("newest", "oldest", "highest_conf", "lowest_conf")

        Returns:
            List of prediction dictionaries
        """
        cursor = self._get_cursor()
        try:
            # Base query
            query = """
            SELECT id, digit, confidence, timestamp, input_type, 
                   metadata, user_correction
            FROM predictions
            WHERE confidence >= ?
            """

            params = [min_confidence]

            # Add digit filter if provided
            if digit_filter is not None:
                query += " AND digit = ?"
                params.append(digit_filter)

            # Add sorting
            if sort_by == "oldest":
                query += " ORDER BY timestamp ASC"
            elif sort_by == "highest_conf":
                query += " ORDER BY confidence DESC"
            elif sort_by == "lowest_conf":
                query += " ORDER BY confidence ASC"
            else:  # default to newest
                query += " ORDER BY timestamp DESC"

            # Add limit and offset
            query += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            cursor.execute(query, params)

            predictions = []
            for row in cursor.fetchall():
                prediction = dict(row)

                # Parse metadata JSON if present
                if prediction.get("metadata"):
                    try:
                        prediction["metadata"] = json.loads(prediction["metadata"])
                    except json.JSONDecodeError:
                        pass

                predictions.append(prediction)

            return predictions
        except sqlite3.Error as e:
            self._logger.error(f"Error getting predictions: {e}")
            return []

    def count_predictions(
        self, digit_filter: Optional[int] = None, min_confidence: float = 0.0
    ) -> int:
        """
        Count predictions with optional filtering.

        Args:
            digit_filter: Filter predictions by digit
            min_confidence: Minimum confidence threshold

        Returns:
            Number of matching predictions
        """
        cursor = self._get_cursor()
        try:
            # Base query
            query = "SELECT COUNT(*) FROM predictions WHERE confidence >= ?"
            params = [min_confidence]

            # Add digit filter if provided
            if digit_filter is not None:
                query += " AND digit = ?"
                params.append(digit_filter)

            cursor.execute(query, params)
            count = cursor.fetchone()[0]
            return count
        except sqlite3.Error as e:
            self._logger.error(f"Error counting predictions: {e}")
            return 0

    def update_prediction(self, prediction_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update a prediction by ID.

        Args:
            prediction_id: ID of the prediction to update
            updates: Dictionary of fields to update

        Returns:
            True if successful, False otherwise
        """
        cursor = self._get_cursor()
        try:
            # Start building the query
            query_parts = []
            params = []

            # Build SET part of the query
            for key, value in updates.items():
                if key in (
                    "digit",
                    "confidence",
                    "input_type",
                    "user_correction",
                ):
                    query_parts.append(f"{key} = ?")
                    params.append(value)
                elif key == "metadata" and isinstance(value, dict):
                    query_parts.append("metadata = ?")
                    params.append(json.dumps(value))

            if not query_parts:
                return False

            # Complete the query
            query = f"UPDATE predictions SET {', '.join(query_parts)} WHERE id = ?"
            params.append(prediction_id)

            cursor.execute(query, params)
            self._get_connection().commit()

            return cursor.rowcount > 0
        except sqlite3.Error as e:
            self._logger.error(f"Error updating prediction {prediction_id}: {e}")
            self._get_connection().rollback()
            return False

    def delete_prediction(self, prediction_id: str) -> bool:
        """
        Delete a prediction by ID.

        Args:
            prediction_id: ID of the prediction to delete

        Returns:
            True if successful, False otherwise
        """
        cursor = self._get_cursor()
        try:
            cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
            self._get_connection().commit()

            success = cursor.rowcount > 0
            if success:
                self._logger.debug(f"Deleted prediction with ID: {prediction_id}")
            else:
                self._logger.warning(f"No prediction found with ID: {prediction_id}")

            return success
        except sqlite3.Error as e:
            self._logger.error(f"Error deleting prediction {prediction_id}: {e}")
            self._get_connection().rollback()
            return False

    def clear_predictions(self) -> bool:
        """
        Delete all predictions from the database.

        Returns:
            True if successful, False otherwise
        """
        cursor = self._get_cursor()
        try:
            cursor.execute("DELETE FROM predictions")
            self._get_connection().commit()

            self._logger.info(
                f"Cleared all predictions, deleted {cursor.rowcount} records"
            )
            return True
        except sqlite3.Error as e:
            self._logger.error(f"Error clearing predictions: {e}")
            self._get_connection().rollback()
            return False

    def get_setting(self, category: str, key: str, default: Any = None) -> Any:
        """
        Get a setting value.

        Args:
            category: Setting category
            key: Setting key
            default: Default value if setting not found

        Returns:
            Setting value or default
        """
        cursor = self._get_cursor()
        try:
            cursor.execute(
                """
            SELECT value 
            FROM settings 
            WHERE category = ? AND key = ?
            """,
                (category, key),
            )

            row = cursor.fetchone()
            if not row:
                return default

            value = row[0]

            # Try to parse as JSON
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                # Return as string if not valid JSON
                return value
        except sqlite3.Error as e:
            self._logger.error(f"Error getting setting {category}.{key}: {e}")
            return default

    def set_setting(self, category: str, key: str, value: Any) -> bool:
        """
        Set a setting value.

        Args:
            category: Setting category
            key: Setting key
            value: Setting value (will be converted to JSON)

        Returns:
            True if successful, False otherwise
        """
        cursor = self._get_cursor()
        try:
            # Convert value to JSON string
            if not isinstance(value, str):
                value = json.dumps(value)

            cursor.execute(
                """
            INSERT OR REPLACE INTO settings (category, key, value)
            VALUES (?, ?, ?)
            """,
                (category, key, value),
            )

            self._get_connection().commit()
            self._logger.debug(f"Setting updated: {category}.{key}")
            return True
        except sqlite3.Error as e:
            self._logger.error(f"Error setting {category}.{key}: {e}")
            self._get_connection().rollback()
            return False

    def delete_setting(self, category: str, key: str) -> bool:
        """
        Delete a setting.

        Args:
            category: Setting category
            key: Setting key

        Returns:
            True if successful, False otherwise
        """
        cursor = self._get_cursor()
        try:
            cursor.execute(
                """
            DELETE FROM settings
            WHERE category = ? AND key = ?
            """,
                (category, key),
            )

            self._get_connection().commit()
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            self._logger.error(f"Error deleting setting {category}.{key}: {e}")
            self._get_connection().rollback()
            return False

    def get_db_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary with database statistics
        """
        stats = {
            "total_predictions": 0,
            "predictions_by_digit": {},
            "average_confidence": 0.0,
            "database_size_kb": 0,
        }

        try:
            cursor = self._get_cursor()

            # Get total predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            stats["total_predictions"] = cursor.fetchone()[0]

            # Get predictions by digit
            cursor.execute(
                """
            SELECT digit, COUNT(*) 
            FROM predictions 
            GROUP BY digit
            ORDER BY digit
            """
            )
            stats["predictions_by_digit"] = {
                str(row[0]): row[1] for row in cursor.fetchall()
            }

            # Get average confidence
            if stats["total_predictions"] > 0:
                cursor.execute("SELECT AVG(confidence) FROM predictions")
                stats["average_confidence"] = cursor.fetchone()[0]

            # Get database file size
            if os.path.exists(self._db_path):
                stats["database_size_kb"] = os.path.getsize(self._db_path) / 1024

            return stats
        except sqlite3.Error as e:
            self._logger.error(f"Error getting DB stats: {e}")
            return stats


# Create a singleton instance for import
db_manager = DatabaseManager()
