# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/core/database/db_manager.py
# Description: Thread-safe database manager for PostgreSQL database operations
# Created: 2025-03-24
# Updated: 2025-03-30 (Refactored for PostgreSQL with explicit connection args and retries)

import os
import json
import logging
import uuid
import threading
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import time

# --- PostgreSQL Imports ---
import psycopg2
from psycopg2 import pool
from psycopg2.extras import DictCursor

# ---

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Thread-safe PostgreSQL database manager for the MNIST Digit Classifier.
    Handles all database operations for the application.
    Uses a connection pool.
    """
    _instance = None
    _pool = None  # Class-level pool
    _lock = threading.Lock() # Lock for thread-safe pool initialization

    def __new__(cls):
        """Implement singleton pattern for DatabaseManager."""
        if cls._instance is None:
            with cls._lock: # Ensure thread-safe singleton creation
                if cls._instance is None:
                    cls._instance = super(DatabaseManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the database manager with connection pooling and retry logic."""
        if getattr(self, "_initialized", False):
            return

        with self._lock: # Ensure thread-safe initialization
            if getattr(self, "_initialized", False):
                 return

            self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

            # --- PostgreSQL Configuration ---
            self.db_config = {
                "host": os.environ.get("DB_HOST", "db"),
                "database": os.environ.get("DB_NAME", "digit_predictions"),
                "user": os.environ.get("DB_USER", "mnist_app"),
                "password": os.environ.get("DB_PASSWORD", "secure_password"),
                "port": os.environ.get("DB_PORT", "5432")
            }
            # ---

            max_retries = 5
            retry_delay = 2
            init_success = False # Flag to track if init succeeded in the loop
            for attempt in range(max_retries):
                try:
                    # Initialize Pool if it doesn't exist
                    if DatabaseManager._pool is None:
                        self._logger.info(f"Attempt {attempt + 1}: Initializing DB pool for host='{self.db_config['host']}', port='{self.db_config['port']}'...")
                        # Explicitly pass connection parameters
                        DatabaseManager._pool = psycopg2.pool.SimpleConnectionPool(
                            minconn=1,
                            maxconn=5, # Adjust as needed
                            database=self.db_config['database'],
                            user=self.db_config['user'],
                            password=self.db_config['password'],
                            host=self.db_config['host'], # Explicit host
                            port=self.db_config['port'],   # Explicit port
                            connect_timeout=5 # Reasonably short timeout for pool creation
                        )
                        self._logger.info(f"Pool object created for host: {self.db_config['host']}")
                        self._initialized = True # Set flag *after* pool creation attempt

                    # Test connection & verify schema
                    self._logger.info("Verifying DB connection and tables...")
                    conn = self._get_connection() # This should now use the pool
                    if conn:
                         try:
                             self._verify_or_create_tables(conn) # Verify tables
                             self._release_connection(conn)
                             self._logger.info("DatabaseManager initialization and table verification successful.")
                             init_success = True # Mark overall success
                             break # Exit loop on success
                         except Exception as verify_err:
                             self._logger.error(f"Table verification failed: {verify_err}", exc_info=True)
                             self._initialized = False # Reset flag on verify failure
                             if DatabaseManager._pool: # Close pool if verify failed
                                 DatabaseManager._pool.closeall()
                                 DatabaseManager._pool = None
                             raise ConnectionError("Table verification failed.") from verify_err # Force retry
                    else:
                         # If _get_connection returns None after pool init attempt
                         self._initialized = False # Ensure flag is False
                         if DatabaseManager._pool: # Close pool if getconn failed
                             DatabaseManager._pool.closeall()
                             DatabaseManager._pool = None
                         raise ConnectionError("Failed to get connection from pool even after initialization attempt.")

                except (psycopg2.OperationalError, ConnectionError) as e:
                    self._logger.warning(f"DB connection/setup attempt {attempt + 1} failed: {e}")
                    self._initialized = False # Ensure flag is False on error
                    if DatabaseManager._pool: # Close broken pool before retry
                         DatabaseManager._pool.closeall()
                         DatabaseManager._pool = None
                    if attempt + 1 == max_retries:
                        self._logger.error(f"🔥 Max DB connection retries reached. Initialization failed.")
                    else:
                        self._logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                except Exception as e: # Catch other potential errors during init
                    self._logger.error(f"🔥 Unexpected error during DB initialization: {e}", exc_info=True)
                    self._initialized = False # Ensure flag is False
                    if DatabaseManager._pool:
                         DatabaseManager._pool.closeall()
                         DatabaseManager._pool = None
                    break # Stop retrying on unexpected errors

            # Final check after the loop
            if not init_success:
                 self._logger.critical("DATABASE MANAGER FAILED TO INITIALIZE AFTER ALL RETRIES.")
                 self._initialized = False # Explicitly set to False if loop finished without success

    def _get_connection(self) -> Optional[psycopg2.extensions.connection]:
        """Get a connection from the pool, with re-initialization attempt."""
        if not DatabaseManager._pool:
            if not self._initialized:
                 self._logger.warning("Pool is None and manager not initialized. Attempting re-initialization...")
                 try:
                      self.__init__() # Attempt re-init
                 except Exception:
                      self._logger.error("Re-initialization failed.")
                      return None
                 if not DatabaseManager._pool:
                     self._logger.error("Pool is still None after re-initialization attempt.")
                     return None
            else:
                 self._logger.error("Pool is None but manager claims initialized. Resetting state.")
                 self._initialized = False
                 return None

        # If pool exists, try getting a connection
        try:
            conn = DatabaseManager._pool.getconn()
            self._logger.debug(f"Connection acquired by thread {threading.get_ident()}")
            return conn
        except Exception as e:
            self._logger.error(f"🔥 Error getting connection from pool: {e}", exc_info=True)
            # Attempt to recover if pool seems stale
            self._logger.warning("Closing pool due to getconn error and attempting re-initialization.")
            self.close_pool()
            try:
                 self.__init__() # Try re-init
                 if DatabaseManager._pool:
                      conn = DatabaseManager._pool.getconn() # Try getconn again
                      self._logger.debug(f"Connection acquired after re-initialization by thread {threading.get_ident()}")
                      return conn
                 else:
                      self._logger.error("Failed to get connection after pool re-initialization.")
                      return None
            except Exception as reinit_e:
                 self._logger.error(f"Re-initialization after getconn error failed: {reinit_e}")
                 return None

    def _release_connection(self, conn: Optional[psycopg2.extensions.connection]):
        """Release connection back to the pool."""
        if DatabaseManager._pool and conn:
            try:
                DatabaseManager._pool.putconn(conn)
                self._logger.debug(f"Connection released by thread {threading.get_ident()}")
            except Exception as e:
                self._logger.error(f"🔥 Error releasing connection to pool: {e}", exc_info=True)
                # If putconn fails, the connection might be broken, discard it
                try:
                    conn.close()
                except Exception:
                    pass


    def _verify_or_create_tables(self, conn) -> None:
        """Verify or create necessary tables using PostgreSQL syntax."""
        if not conn:
            self._logger.error("Cannot verify tables: No connection provided.")
            return
        try:
            with conn.cursor() as cursor:
                # Verify/Create prediction_logs table
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prediction_logs (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        prediction INTEGER NOT NULL,
                        true_label INTEGER,
                        confidence DOUBLE PRECISION NOT NULL,
                        input_type VARCHAR(50),
                        image_data TEXT,
                        metadata TEXT
                    );
                    """
                )
                # Verify/Create index
                cursor.execute(
                   """CREATE INDEX IF NOT EXISTS idx_timestamp ON prediction_logs (timestamp DESC);"""
                )
                # Verify/Create settings table (if db_manager handles settings)
                cursor.execute(
                   """
                   CREATE TABLE IF NOT EXISTS settings (
                       category TEXT NOT NULL,
                       key TEXT NOT NULL,
                       value TEXT NOT NULL,
                       PRIMARY KEY (category, key)
                   );
                   """
                )
                conn.commit()
                self._logger.info("PostgreSQL tables verified/created successfully.")
        except Exception as e:
            self._logger.error(f"🔥 Error verifying/creating PostgreSQL tables: {e}", exc_info=True)
            conn.rollback() # Rollback on error
            raise # Re-raise the exception to signal failure

    def close_pool(self) -> None:
        """Close all connections in the pool."""
        with self._lock:
             if DatabaseManager._pool:
                 try:
                      DatabaseManager._pool.closeall()
                      self._logger.info("PostgreSQL connection pool closed.")
                 except Exception as e:
                      self._logger.error(f"🔥 Error closing connection pool: {e}", exc_info=True)
                 finally:
                      DatabaseManager._pool = None
                      self._initialized = False # Mark as uninitialized

    # --- Data Methods ---

    def add_prediction(self, prediction_data: Dict[str, Any]) -> Optional[int]:
        """Add a prediction to the database. Returns the auto-generated integer ID."""
        conn = None
        generated_id = None
        try:
            conn = self._get_connection()
            if not conn:
                self._logger.error("Failed to add prediction: Could not get DB connection.")
                return None

            with conn.cursor() as cursor:
                metadata = prediction_data.get("metadata")
                metadata_json = json.dumps(metadata) if isinstance(metadata, dict) else metadata
                image_data_text = prediction_data.get("image_data")
                timestamp = prediction_data.get("timestamp", datetime.now())

                sql = """
                    INSERT INTO prediction_logs (
                        digit, confidence, timestamp, input_type,
                        image_data, metadata, true_label
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                """
                cursor.execute(
                    sql,
                    (
                        prediction_data.get("digit"),
                        prediction_data.get("confidence"),
                        timestamp,
                        prediction_data.get("input_type"),
                        image_data_text,
                        metadata_json,
                        prediction_data.get("true_label") # Use true_label
                    ),
                )
                result = cursor.fetchone()
                if result:
                    generated_id = result[0]
                conn.commit()
                self._logger.info(f"✅ Added prediction to PostgreSQL with auto-generated ID: {generated_id}")
                return generated_id
        except Exception as e:
            self._logger.error(f"🔥 Error adding prediction to PostgreSQL: {e}", exc_info=True)
            if conn: conn.rollback()
            return None
        finally:
            if conn: self._release_connection(conn)

    def get_prediction(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Get a prediction by its integer ID."""
        conn = None
        try:
            conn = self._get_connection()
            if not conn: return None
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    """
                    SELECT id, digit AS prediction, confidence, timestamp, input_type,
                           image_data, metadata, true_label
                    FROM prediction_logs WHERE id = %s;
                    """,
                    (prediction_id,)
                )
                row = cursor.fetchone()
            if not row:
                self._logger.warning(f"Prediction with ID {prediction_id} not found.")
                return None

            prediction = dict(row)
            if prediction.get("metadata"):
                try: prediction["metadata"] = json.loads(prediction["metadata"])
                except (json.JSONDecodeError, TypeError): prediction["metadata"] = {}
            if isinstance(prediction.get("timestamp"), datetime):
                prediction["timestamp"] = prediction["timestamp"].isoformat()
            return prediction
        except Exception as e:
            self._logger.error(f"🔥 Error getting prediction ID {prediction_id}: {e}", exc_info=True)
            return None
        finally:
            if conn: self._release_connection(conn)

    def get_predictions(
        self,
        limit: int = 50,
        offset: int = 0,
        digit_filter: Optional[int] = None,
        min_confidence: float = 0.0,
        sort_by: str = "timestamp",
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """Get predictions with filtering and sorting."""
        conn = None
        predictions = []
        try:
            conn = self._get_connection()
            if not conn: return []

            with conn.cursor(cursor_factory=DictCursor) as cursor:
                valid_sort_columns = {"timestamp", "confidence", "prediction", "id"}
                sort_col = sort_by if sort_by in valid_sort_columns else "timestamp"
                sort_dir = "ASC" if sort_order.lower() == "asc" else "DESC"

                where_clauses = []
                params = []
                if min_confidence > 0:
                    where_clauses.append("confidence >= %s")
                    params.append(min_confidence)
                if digit_filter is not None:
                    where_clauses.append("prediction = %s") # Use prediction column name
                    params.append(digit_filter)
                where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

                params_main_query = params + [limit, offset]

                query = f"""
                    SELECT id, prediction, confidence, timestamp, input_type,
                           image_data, metadata, true_label
                    FROM prediction_logs
                    {where_sql}
                    ORDER BY {sort_col} {sort_dir}
                    LIMIT %s OFFSET %s;
                """
                cursor.execute(query, tuple(params_main_query))
                rows = cursor.fetchall()

                for row in rows:
                    prediction = dict(row)
                    if prediction.get("metadata"):
                         try: prediction["metadata"] = json.loads(prediction["metadata"])
                         except (json.JSONDecodeError, TypeError): prediction["metadata"] = {}
                    if isinstance(prediction.get("timestamp"), datetime):
                         prediction["timestamp"] = prediction["timestamp"].isoformat()
                    predictions.append(prediction)
            return predictions
        except Exception as e:
            self._logger.error(f"🔥 Error getting predictions from PostgreSQL: {e}", exc_info=True)
            return []
        finally:
            if conn: self._release_connection(conn)

    def count_predictions(
        self, digit_filter: Optional[int] = None, min_confidence: float = 0.0
    ) -> int:
        """Count predictions with optional filtering."""
        conn = None
        try:
            conn = self._get_connection()
            if not conn: return 0
            with conn.cursor() as cursor:
                where_clauses = []
                params = []
                if min_confidence > 0:
                    where_clauses.append("confidence >= %s")
                    params.append(min_confidence)
                if digit_filter is not None:
                    where_clauses.append("prediction = %s") # Use prediction column name
                    params.append(digit_filter)
                where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

                query = f"SELECT COUNT(*) FROM prediction_logs {where_sql};"
                cursor.execute(query, tuple(params))
                count = cursor.fetchone()[0]
                return count or 0
        except Exception as e:
            self._logger.error(f"🔥 Error counting predictions in PostgreSQL: {e}", exc_info=True)
            return 0
        finally:
            if conn: self._release_connection(conn)

    def update_prediction(self, prediction_id: int, updates: Dict[str, Any]) -> bool:
        """Update 'true_label' or 'metadata' for a prediction by its integer ID."""
        conn = None
        if not updates or not isinstance(prediction_id, int):
            self._logger.warning("Invalid arguments for update_prediction.")
            return False

        try:
            conn = self._get_connection()
            if not conn: return False
            with conn.cursor() as cursor:
                query_parts = []
                params = []
                if "true_label" in updates:
                    query_parts.append("true_label = %s")
                    params.append(updates["true_label"])
                if "metadata" in updates:
                    metadata_val = updates["metadata"]
                    metadata_json = json.dumps(metadata_val) if isinstance(metadata_val, dict) else metadata_val
                    query_parts.append("metadata = %s")
                    params.append(metadata_json)

                if not query_parts:
                    self._logger.warning(f"No valid fields to update for prediction ID {prediction_id}.")
                    return False

                query = f"UPDATE prediction_logs SET {', '.join(query_parts)} WHERE id = %s;"
                params.append(prediction_id)

                cursor.execute(query, tuple(params))
                conn.commit()
                success = cursor.rowcount > 0
                if success: self._logger.info(f"✅ Updated prediction ID {prediction_id}.")
                else: self._logger.warning(f"⚠️ No prediction found with ID {prediction_id} to update.")
                return success
        except Exception as e:
            self._logger.error(f"🔥 Error updating prediction ID {prediction_id}: {e}", exc_info=True)
            if conn: conn.rollback()
            return False
        finally:
            if conn: self._release_connection(conn)

    def delete_prediction(self, prediction_id: int) -> bool:
        """Delete a prediction by its integer ID."""
        conn = None
        try:
            conn = self._get_connection()
            if not conn: return False
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM prediction_logs WHERE id = %s;", (prediction_id,))
                conn.commit()
                success = cursor.rowcount > 0
                if success: self._logger.info(f"✅ Deleted prediction ID: {prediction_id}")
                else: self._logger.warning(f"⚠️ No prediction found with ID {prediction_id} to delete.")
                return success
        except Exception as e:
            self._logger.error(f"🔥 Error deleting prediction ID {prediction_id}: {e}", exc_info=True)
            if conn: conn.rollback()
            return False
        finally:
            if conn: self._release_connection(conn)

    def clear_predictions(self) -> bool:
        """Delete all predictions from the database."""
        conn = None
        try:
            conn = self._get_connection()
            if not conn: return False
            with conn.cursor() as cursor:
                cursor.execute("DELETE FROM prediction_logs;")
                conn.commit()
                self._logger.info(f"Cleared all predictions ({cursor.rowcount} records) from PostgreSQL.")
                return True
        except Exception as e:
            self._logger.error(f"🔥 Error clearing predictions from PostgreSQL: {e}", exc_info=True)
            if conn: conn.rollback()
            return False
        finally:
            if conn: self._release_connection(conn)

    # --- Settings Methods ---
    # Keeping these here assumes db_manager is the sole interface.
    # Ensure the 'settings' table is created in _verify_or_create_tables.

    def get_setting(self, category: str, key: str, default: Any = None) -> Any:
        """Gets a setting from the 'settings' table."""
        conn = None
        try:
            conn = self._get_connection()
            if not conn: return default
            with conn.cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute("SELECT value FROM settings WHERE category = %s AND key = %s;", (category, key))
                row = cursor.fetchone()
                if row:
                    try: return json.loads(row['value'])
                    except (json.JSONDecodeError, TypeError): return row['value']
                return default
        except Exception as e:
            self._logger.error(f"🔥 Error getting setting {category}.{key}: {e}", exc_info=True)
            return default
        finally:
            if conn: self._release_connection(conn)

    def set_setting(self, category: str, key: str, value: Any) -> bool:
        """Sets a setting in the 'settings' table (upsert)."""
        conn = None
        try:
            conn = self._get_connection()
            if not conn: return False
            with conn.cursor() as cursor:
                value_json = json.dumps(value)
                cursor.execute(
                    """
                    INSERT INTO settings (category, key, value) VALUES (%s, %s, %s)
                    ON CONFLICT (category, key) DO UPDATE SET value = EXCLUDED.value;
                    """,
                    (category, key, value_json)
                )
                conn.commit()
                self._logger.debug(f"Set setting {category}.{key}")
                return True
        except Exception as e:
            self._logger.error(f"🔥 Error setting {category}.{key}: {e}", exc_info=True)
            if conn: conn.rollback()
            return False
        finally:
            if conn: self._release_connection(conn)

# --- Singleton Instance ---
# This line creates the single, globally accessible instance of the manager.
db_manager = DatabaseManager()