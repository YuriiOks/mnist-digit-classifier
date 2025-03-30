# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: services/prediction/prediction_service.py
# Description: Service for handling predictions and database logging
# Created: 2025-03-17

import logging
import os
import json
import requests
import time
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

# We'll conditionally import psycopg2 to handle the case where it's not installed
try:
    import psycopg2
    import psycopg2.extras

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logging.warning(
        "psycopg2 not available. Database functionality will be disabled."
    )


class PredictionService:
    """Service for handling digit predictions and logging to database."""

    _instance = None

    def __new__(cls):
        """Create a singleton instance."""
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the prediction service."""
        # Skip initialization if already done
        if getattr(self, "_initialized", False):
            return

        self.logger = logging.getLogger(__name__)

        # Database connection parameters from environment variables
        self.db_config = {
            "host": os.environ.get("DB_HOST", "db"),
            "database": os.environ.get("DB_NAME", "digit_predictions"),
            "user": os.environ.get("DB_USER", "mnist_app"),
            "password": os.environ.get("DB_PASSWORD", "secure_password"),
            "port": os.environ.get("DB_PORT", "5432"),
        }

        self.model_url = os.environ.get("MODEL_URL", "http://model:5000")
        self.log_to_db = os.environ.get("LOG_TO_DB", "true").lower() == "true"

        # Check if database is available
        if PSYCOPG2_AVAILABLE and self.log_to_db:
            self._check_and_setup_database()
        else:
            self.logger.warning(
                "Database logging is disabled or psycopg2 is not available"
            )

        self._initialized = True
        self.logger.info("PredictionService initialized")

    def _check_and_setup_database(self) -> None:
        """Check database connection and set up tables if needed."""
        if not PSYCOPG2_AVAILABLE:
            self.logger.warning(
                "psycopg2 not available. Cannot set up database."
            )
            return

        conn = None
        try:
            # Log connection attempt
            self.logger.info(
                f"Attempting to connect to database at {self.db_config['host']}:{self.db_config['port']}"
            )

            # Try to connect with a timeout
            conn = psycopg2.connect(**self.db_config, connect_timeout=5)

            # Create tables if they don't exist
            cursor = conn.cursor()

            # Create prediction_logs table if not exists
            cursor.execute(
                """
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                prediction INTEGER NOT NULL,
                true_label INTEGER,
                confidence FLOAT NOT NULL,
                input_type VARCHAR(20)
            )
            """
            )

            # Create index on timestamp for faster retrieval
            cursor.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON prediction_logs (timestamp DESC)
            """
            )

            conn.commit()
            cursor.close()

            self.logger.info(
                "Database connection successful and tables verified"
            )

        except Exception as e:
            self.logger.error(f"Database setup error: {str(e)}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def get_db_connection(self) -> Optional[Any]:
        """
        Get a connection to the database.

        Returns:
            Database connection or None if connection fails
        """
        if not PSYCOPG2_AVAILABLE:
            self.logger.warning(
                "psycopg2 not available. Cannot connect to database."
            )
            return None

        try:
            conn = psycopg2.connect(**self.db_config)
            return conn
        except Exception as e:
            self.logger.error(f"Database connection failed: {str(e)}")
            return None

    def log_prediction(
        self,
        prediction: int,
        confidence: float,
        input_type: Optional[str] = None,
        true_label: Optional[int] = None,
    ) -> Optional[int]:
        """
        Log a prediction to the database.

        Args:
            prediction: The predicted digit
            confidence: Confidence score (0-1)
            input_type: Type of input (canvas, upload, url)
            true_label: Optional true label for the digit

        Returns:
            int: ID of the logged prediction, or None if logging failed
        """
        if not self.log_to_db or not PSYCOPG2_AVAILABLE:
            self.logger.info("Database logging is disabled or not available")
            return None

        conn = self.get_db_connection()
        if not conn:
            return None

        try:
            cursor = conn.cursor()

            # Insert into prediction_logs table
            query = """
            INSERT INTO prediction_logs 
            (prediction, confidence, true_label, input_type)
            VALUES (%s, %s, %s, %s)
            RETURNING id;
            """

            cursor.execute(
                query, (prediction, confidence, true_label, input_type)
            )
            prediction_id = cursor.fetchone()[0]

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Prediction logged to database with ID: {prediction_id}"
            )
            return prediction_id

        except Exception as e:
            self.logger.error(
                f"Error logging prediction to database: {str(e)}"
            )
            if conn:
                conn.rollback()
                conn.close()
            return None

    def get_prediction_history(
        self,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
    ) -> Tuple[List[Dict[str, Any]], int]:
        """
        Get prediction history from the database.

        Args:
            limit: Maximum number of results to return
            offset: Offset for pagination
            sort_by: Column to sort by (timestamp, prediction, confidence)
            sort_order: Sort order (asc, desc)

        Returns:
            Tuple of (list of predictions, total count)
        """
        if not PSYCOPG2_AVAILABLE:
            self.logger.warning(
                "psycopg2 not available. Cannot retrieve history."
            )
            return [], 0

        conn = self.get_db_connection()
        if not conn:
            return [], 0

        try:
            # Use DictCursor to get results as dictionaries
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

            # Validate and sanitize sort parameters to prevent SQL injection
            valid_sort_columns = [
                "timestamp",
                "prediction",
                "confidence",
                "id",
            ]
            if sort_by not in valid_sort_columns:
                sort_by = "timestamp"

            valid_sort_orders = ["asc", "desc"]
            if sort_order.lower() not in valid_sort_orders:
                sort_order = "desc"

            # Get total count
            cursor.execute("SELECT COUNT(*) FROM prediction_logs")
            total_count = cursor.fetchone()[0]

            # Get predictions with pagination
            query = f"""
            SELECT id, timestamp, prediction, true_label, confidence, input_type
            FROM prediction_logs
            ORDER BY {sort_by} {sort_order}
            LIMIT %s OFFSET %s
            """

            cursor.execute(query, (limit, offset))
            rows = cursor.fetchall()

            # Convert to list of dictionaries
            predictions = []
            for row in rows:
                # Convert datetime to string for JSON serialization
                row_dict = dict(row)
                if isinstance(row_dict["timestamp"], datetime):
                    row_dict["timestamp"] = row_dict["timestamp"].isoformat()
                predictions.append(row_dict)

            cursor.close()
            conn.close()

            return predictions, total_count

        except Exception as e:
            self.logger.error(f"Error getting prediction history: {str(e)}")
            if conn:
                conn.close()
            return [], 0

    def update_true_label(self, prediction_id: int, true_label: int) -> bool:
        """
        Update the true label for a prediction.

        Args:
            prediction_id: ID of the prediction
            true_label: Correct digit value

        Returns:
            bool: True if update was successful, False otherwise
        """
        if not PSYCOPG2_AVAILABLE:
            self.logger.warning(
                "psycopg2 not available. Cannot update true label."
            )
            return False

        conn = self.get_db_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()

            # Update true_label in prediction_logs table
            query = """
            UPDATE prediction_logs
            SET true_label = %s
            WHERE id = %s
            """

            cursor.execute(query, (true_label, prediction_id))
            rows_affected = cursor.rowcount

            conn.commit()
            cursor.close()
            conn.close()

            success = rows_affected > 0
            if success:
                self.logger.info(
                    f"Updated true label for prediction {prediction_id} to {true_label}"
                )
            else:
                self.logger.warning(
                    f"No prediction found with ID {prediction_id}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error updating true label: {str(e)}")
            if conn:
                conn.rollback()
                conn.close()
            return False

    def delete_prediction(self, prediction_id: int) -> bool:
        """
        Delete a prediction from the database.

        Args:
            prediction_id: ID of the prediction to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        if not PSYCOPG2_AVAILABLE:
            self.logger.warning(
                "psycopg2 not available. Cannot delete prediction."
            )
            return False

        conn = self.get_db_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()

            # Delete from prediction_logs table
            query = """
            DELETE FROM prediction_logs
            WHERE id = %s
            """

            cursor.execute(query, (prediction_id,))
            rows_affected = cursor.rowcount

            conn.commit()
            cursor.close()
            conn.close()

            success = rows_affected > 0
            if success:
                self.logger.info(
                    f"Deleted prediction with ID {prediction_id}"
                )
            else:
                self.logger.warning(
                    f"No prediction found with ID {prediction_id}"
                )

            return success

        except Exception as e:
            self.logger.error(f"Error deleting prediction: {str(e)}")
            if conn:
                conn.rollback()
                conn.close()
            return False

    def clear_history(self) -> bool:
        """
        Clear all prediction history from the database.

        Returns:
            bool: True if clearing was successful, False otherwise
        """
        if not PSYCOPG2_AVAILABLE:
            self.logger.warning(
                "psycopg2 not available. Cannot clear history."
            )
            return False

        conn = self.get_db_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor()

            # Delete all records from prediction_logs table
            query = """
            DELETE FROM prediction_logs
            """

            cursor.execute(query)
            rows_affected = cursor.rowcount

            conn.commit()
            cursor.close()
            conn.close()

            self.logger.info(
                f"Cleared prediction history. {rows_affected} records deleted."
            )
            return True

        except Exception as e:
            self.logger.error(f"Error clearing prediction history: {str(e)}")
            if conn:
                conn.rollback()
                conn.close()
            return False


# Create a singleton instance
prediction_service = PredictionService()
