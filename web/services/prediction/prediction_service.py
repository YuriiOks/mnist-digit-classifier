# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/services/prediction/prediction_service.py
# Description: Service handling direct DB interaction for prediction logs.
# Created: 2025-03-17
# Updated: 2025-03-30

import logging
import os
from typing import Dict, Any, Tuple, Optional, List
from datetime import datetime

try:
    import psycopg2
    import psycopg2.extras

    PSYCOPG2_AVAILABLE = True
except ImportError:
    PSYCOPG2_AVAILABLE = False
    logging.warning("psycopg2 not found. DB functionality disabled.")

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for direct DB operations on prediction_logs."""

    # (Keep Singleton pattern __new__ if desired)
    _instance = None

    def __new__(cls):
        """
        Ensure only one instance of PredictionService exists.
        This is a Singleton pattern.
        """
        if cls._instance is None:
            cls._instance = super(PredictionService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """
        Initialize the PredictionService instance.
        This method sets up the database connection parameters,
        checks the database connection, and ensures the
        prediction_logs table exists.
        """

        if getattr(self, "_initialized", False):
            return

        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.db_config = {
            "host": os.environ.get("DB_HOST", "db"),
            "database": os.environ.get("DB_NAME", "digit_predictions"),
            "user": os.environ.get("DB_USER", "mnist_app"),
            "password": os.environ.get("DB_PASSWORD", "secure_password"),
            "port": os.environ.get("DB_PORT", "5432"),
        }
        self.log_to_db = os.environ.get("LOG_TO_DB", "true").lower() == "true"

        if PSYCOPG2_AVAILABLE and self.log_to_db:
            self._check_and_setup_database()
        else:
            self.logger.warning("DB logging disabled or psycopg2 unavailable.")

        self._initialized = True
        self.logger.info(f"{self.__class__.__name__} initialized.")

    def _check_and_setup_database(self) -> None:
        """Check DB connection and ensure prediction_logs table exists."""
        if not PSYCOPG2_AVAILABLE:
            return
        conn = None
        try:
            self.logger.info(
                f"DB Check: Connecting to "
                f"{self.db_config['host']}:{self.db_config['port']}"
            )
            conn = psycopg2.connect(**self.db_config, connect_timeout=5)
            cursor = conn.cursor()
            # Added image_data TEXT field if it's missing
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_logs (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    prediction INTEGER NOT NULL,
                    true_label INTEGER,
                    confidence FLOAT NOT NULL,
                    input_type VARCHAR(20),
                    image_data TEXT
                );
                -- Optional: Add column if it doesn't exist (less disruptive)
                -- ALTER TABLE prediction_logs ADD COLUMN IF NOT EXISTS image_data TEXT;
            """
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_timestamp "
                "ON prediction_logs (timestamp DESC);"
            )
            conn.commit()
            cursor.close()
            self.logger.info("DB Check: Connection successful, table verified.")
        except Exception as e:
            self.logger.error(f"🔥 DB Check/Setup error: {e}")
            if conn:
                conn.rollback()
        finally:
            if conn:
                conn.close()

    def get_db_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Establishes and returns a database connection."""
        if not PSYCOPG2_AVAILABLE:
            return None
        try:
            conn = psycopg2.connect(**self.db_config, connect_timeout=5)
            return conn
        except Exception as e:
            self.logger.error(f"🔥 Database connection failed: {e}")
            return None

    def log_prediction(
        self,
        prediction: int,
        confidence: float,
        input_type: Optional[str] = None,
        true_label: Optional[int] = None,
        image_data: Optional[str] = None,  # Expecting base64 string if logged
    ) -> Optional[int]:
        """Logs a prediction, returns the integer database ID."""
        if not self.log_to_db or not PSYCOPG2_AVAILABLE:
            return None
        conn = self.get_db_connection()
        if not conn:
            return None

        db_id = None
        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO prediction_logs
                (prediction, confidence, true_label, input_type, image_data)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id;
            """
            cursor.execute(
                query, (prediction, confidence, true_label, input_type, image_data)
            )
            result = cursor.fetchone()
            if result:
                db_id = result[0]  # Get the integer ID
            conn.commit()
            self.logger.info(f"✅ Prediction logged to DB with ID: {db_id}")
        except Exception as e:
            self.logger.error(f"🔥 Error logging prediction: {e}", exc_info=True)
            if conn:
                conn.rollback()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        return db_id  # Return integer ID or None

    def get_prediction_history(
        self,
        limit: int = 100,
        offset: int = 0,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
        # Add filters if needed by db_manager/direct query
        digit_filter: Optional[int] = None,
        min_confidence: float = 0.0,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Gets prediction history from the database (paginated)."""
        if not PSYCOPG2_AVAILABLE:
            return [], 0
        conn = self.get_db_connection()
        if not conn:
            return [], 0

        predictions = []
        total_count = 0
        try:
            cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
            # Basic Sanitization
            valid_sort_cols = ["id", "timestamp", "prediction", "confidence"]
            sort_by = sort_by if sort_by in valid_sort_cols else "timestamp"
            sort_order = "ASC" if sort_order.lower() == "asc" else "DESC"

            # Build WHERE clause (Example - adjust as needed)
            where_clauses = []
            params = []
            if digit_filter is not None:
                where_clauses.append("prediction = %s")
                params.append(digit_filter)
            if min_confidence > 0:
                where_clauses.append("confidence >= %s")
                params.append(min_confidence)
            where_sql = (
                ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
            )

            # Count Query
            count_query = f"SELECT COUNT(*) FROM prediction_logs {where_sql}"
            cursor.execute(count_query, tuple(params))
            total_count = cursor.fetchone()[0]

            # Data Query
            params.extend([limit, offset])
            data_query = f"""
                SELECT id, timestamp, prediction, true_label, confidence,
                       input_type, image_data
                FROM prediction_logs
                {where_sql}
                ORDER BY {sort_by} {sort_order}
                LIMIT %s OFFSET %s
            """
            cursor.execute(data_query, tuple(params))
            rows = cursor.fetchall()
            for row in rows:
                row_dict = dict(row)
                # Ensure timestamp is serializable
                if isinstance(row_dict.get("timestamp"), datetime):
                    row_dict["timestamp"] = row_dict["timestamp"].isoformat()
                # Decode image if needed? Usually done in UI.
                # if row_dict.get("image_data"):
                #     row_dict["image_data"] = base64.b64decode(...)
                predictions.append(row_dict)

        except Exception as e:
            self.logger.error(
                f"🔥 Error getting prediction history: {e}", exc_info=True
            )
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        return predictions, total_count

    def update_true_label(self, prediction_db_id: int, true_label: int) -> bool:
        """Updates the true label for a specific prediction log entry by its integer ID."""
        if not self.log_to_db or not PSYCOPG2_AVAILABLE:
            return False
        conn = self.get_db_connection()
        if not conn:
            return False

        success = False
        try:
            cursor = conn.cursor()
            query = "UPDATE prediction_logs SET true_label = %s WHERE id = %s"
            # Execute with INTEGER prediction_db_id
            cursor.execute(query, (true_label, prediction_db_id))
            rows_affected = cursor.rowcount
            conn.commit()
            success = rows_affected > 0
            if success:
                self.logger.info(
                    f"✅ Updated true_label for DB ID "
                    f"{prediction_db_id} to {true_label}"
                )
            else:
                self.logger.warning(
                    f"⚠️ No row found with DB ID " f"{prediction_db_id} to update."
                )
        except Exception as e:
            self.logger.error(
                f"🔥 Error updating true_label for DB ID " f"{prediction_db_id}: {e}",
                exc_info=True,
            )
            if conn:
                conn.rollback()
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()
        return success


prediction_service = PredictionService()