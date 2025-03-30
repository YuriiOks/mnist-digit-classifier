# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/core/database/database.py
# Description: [Brief description of the file's purpose]
# Created: 2025-03-30
# Updated: 2025-03-30

import os
import logging
import psycopg2
from psycopg2 import pool
from typing import Optional, Dict, Any, List, Tuple


class Database:
    """PostgreSQL database connection manager."""

    # Singleton instance
    _instance = None

    # Connection pool
    _pool = None

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the database connection manager."""
        if self._initialized:
            return

        self.logger = logging.getLogger(__name__)

        # Get database configuration from environment variables
        self.db_config = {
            "host": os.environ.get("DB_HOST", "db"),
            "database": os.environ.get("DB_NAME", "digit_predictions"),
            "user": os.environ.get("DB_USER", "mnist_app"),
            "password": os.environ.get("DB_PASSWORD", "secure_password"),
            "port": os.environ.get("DB_PORT", "5432"),
        }

        self.enabled = os.environ.get("USE_DATABASE", "true").lower() == "true"

        # Initialize connection pool if enabled
        if self.enabled:
            try:
                self._pool = pool.SimpleConnectionPool(
                    minconn=1, maxconn=10, **self.db_config
                )
                self.logger.info("Database connection pool initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize database pool: {str(e)}")
                self.enabled = False

        self._initialized = True

    def get_connection(self):
        """Get a connection from the pool."""
        if not self.enabled or self._pool is None:
            return None

        try:
            return self._pool.getconn()
        except Exception as e:
            self.logger.error(f"Failed to get database connection: {str(e)}")
            return None

    def release_connection(self, conn):
        """Release a connection back to the pool."""
        if self._pool is not None and conn is not None:
            self._pool.putconn(conn)

    def execute_query(
        self, query: str, params: tuple = None, fetch: bool = False
    ) -> Optional[List[Tuple]]:
        """Execute a SQL query with optional parameters.

        Args:
            query: SQL query to execute
            params: Optional query parameters
            fetch: Whether to fetch results

        Returns:
            Query results if fetch=True, otherwise None
        """
        if not self.enabled:
            self.logger.warning("Database is disabled. Skipping query.")
            return None

        conn = None
        try:
            conn = self.get_connection()
            if conn is None:
                return None

            with conn.cursor() as cursor:
                cursor.execute(query, params)

                if fetch:
                    result = cursor.fetchall()
                else:
                    result = None

                conn.commit()
                return result

        except Exception as e:
            self.logger.error(f"Database query error: {str(e)}")
            if conn:
                conn.rollback()
            return None
        finally:
            if conn:
                self.release_connection(conn)

    def insert_prediction(
        self,
        prediction: int,
        confidence: float,
        true_label: Optional[int] = None,
        input_type: Optional[str] = None,
        image_data: Optional[str] = None,
    ) -> Optional[int]:
        """Insert a prediction record and return its ID.

        Args:
            prediction: Predicted digit
            confidence: Confidence value (0-1)
            true_label: Optional true label if known
            input_type: Type of input (canvas, upload, url)
            image_data: Optional base64 encoded image data

        Returns:
            Inserted record ID or None if insert failed
        """
        if not self.enabled:
            self.logger.warning("Database is disabled. Skipping insert.")
            return None

        query = """
        INSERT INTO prediction_logs 
        (prediction, confidence, true_label, input_type, image_data)
        VALUES (%s, %s, %s, %s, %s)
        RETURNING id
        """

        # Limit image data size to prevent DB issues
        if image_data and len(image_data) > 100000:  # 100KB limit
            image_data = None
            self.logger.warning(
                "Image data too large for database storage. Skipping image."
            )

        result = self.execute_query(
            query,
            (prediction, confidence, true_label, input_type, image_data),
            fetch=True,
        )

        if result and len(result) > 0:
            return result[0][0]  # First column of first row (ID)
        return None

    def update_true_label(self, prediction_id: int, true_label: int) -> bool:
        """Update the true label for a prediction.

        Args:
            prediction_id: ID of the prediction to update
            true_label: Correct label value

        Returns:
            True if update was successful, False otherwise
        """
        if not self.enabled:
            self.logger.warning("Database is disabled. Skipping update.")
            return False

        query = """
        UPDATE prediction_logs
        SET true_label = %s
        WHERE id = %s
        """

        self.execute_query(query, (true_label, prediction_id))
        return True

    def get_predictions(
        self,
        limit: int = 20,
        offset: int = 0,
        digit_filter: Optional[int] = None,
        min_confidence: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Get prediction records with optional filtering.

        Args:
            limit: Maximum number of records to return
            offset: Number of records to skip
            digit_filter: Optional filter by predicted digit
            min_confidence: Minimum confidence threshold

        Returns:
            Tuple of (list of prediction records, total count)
        """
        if not self.enabled:
            self.logger.warning("Database is disabled. Returning empty results.")
            return [], 0

        # Base query conditions
        conditions = ["confidence >= %s"]
        params = [min_confidence]

        # Add digit filter if specified
        if digit_filter is not None:
            conditions.append("prediction = %s")
            params.append(digit_filter)

        # Build WHERE clause
        where_clause = " AND ".join(conditions)

        # Count query
        count_query = f"""
        SELECT COUNT(*) FROM prediction_logs
        WHERE {where_clause}
        """

        # Data query
        data_query = f"""
        SELECT id, timestamp, prediction, true_label, confidence, input_type
        FROM prediction_logs
        WHERE {where_clause}
        ORDER BY timestamp DESC
        LIMIT %s OFFSET %s
        """

        # Get total count
        count_result = self.execute_query(count_query, tuple(params), fetch=True)
        total_count = count_result[0][0] if count_result else 0

        # Add limit and offset to params
        params.extend([limit, offset])

        # Get records
        records_result = self.execute_query(data_query, tuple(params), fetch=True)

        # Convert to list of dictionaries
        records = []
        if records_result:
            for row in records_result:
                records.append(
                    {
                        "id": row[0],
                        "timestamp": row[1],
                        "digit": row[2],
                        "true_label": row[3],
                        "confidence": row[4],
                        "input_type": row[5] or "unknown",
                    }
                )

        return records, total_count

    def clear_history(self) -> bool:
        """Clear all prediction history.

        Returns:
            True if successful, False otherwise
        """
        if not self.enabled:
            self.logger.warning("Database is disabled. Skipping clear.")
            return False

        query = "DELETE FROM prediction_logs"
        self.execute_query(query)
        return True


# Create a singleton instance
db = Database()
