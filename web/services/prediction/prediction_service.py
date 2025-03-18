# MNIST Digit Classifier
# Copyright (c) 2025
# File: services/prediction/prediction_service.py
# Description: Service for handling predictions and database logging
# Created: 2025-03-17

import logging
import os
import json
import requests
import psycopg2
from typing import Dict, Any, Tuple, Optional
from datetime import datetime

class PredictionService:
    """Service for handling digit predictions and logging to database."""
    
    def __init__(self):
        """Initialize the prediction service."""
        self.logger = logging.getLogger(__name__)
        
        # Database connection parameters from environment variables
        self.db_config = {
            "host": os.environ.get("DB_HOST", "db"),
            "database": os.environ.get("DB_NAME", "digit_predictions"),
            "user": os.environ.get("DB_USER", "mnist_app"),
            "password": os.environ.get("DB_PASSWORD", "secure_password"),
            "port": os.environ.get("DB_PORT", "5432")
        }
        
        self.model_url = os.environ.get("MODEL_URL", "http://model:5000")
        self.log_to_db = os.environ.get("LOG_TO_DB", "true").lower() == "true"
        
    def get_db_connection(self) -> Optional[psycopg2.extensions.connection]:
        """
        Get a connection to the database.
        
        Returns:
            Database connection or None if connection fails
        """
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
        true_label: Optional[int] = None
    ) -> bool:
        """
        Log a prediction to the database.
        
        Args:
            prediction: The predicted digit
            confidence: Confidence score (0-1)
            true_label: Optional true label for the digit
            
        Returns:
            bool: True if logging was successful, False otherwise
        """
        if not self.log_to_db:
            self.logger.info("Database logging is disabled")
            return False
            
        conn = self.get_db_connection()
        if not conn:
            return False
            
        try:
            cursor = conn.cursor()
            
            # Insert into prediction_logs table
            query = """
            INSERT INTO prediction_logs (prediction, confidence, true_label)
            VALUES (%s, %s, %s)
            RETURNING id;
            """
            
            cursor.execute(query, (prediction, confidence, true_label))
            prediction_id = cursor.fetchone()[0]
            
            conn.commit()
            cursor.close()
            conn.close()
            
            self.logger.info(f"Prediction logged to database with ID: {prediction_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error logging prediction to database: {str(e)}")
            if conn:
                conn.close()
            return False
    
    def get_prediction_history(
        self, 
        limit: int = 100, 
        offset: int = 0
    ) -> Tuple[list, int]:
        """
        Get prediction history from the database.
        
        Args:
            limit: Maximum number of results to return
            offset: Offset for pagination
            
        Returns:
            Tuple of (list of predictions, total count)
        """
        conn = self.get_db_connection()
        if not conn:
            return [], 0
            
        try:
            cursor = conn.cursor()
            
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM prediction_logs")
            total_count = cursor.fetchone()[0]
            
            # Get predictions with pagination
            query = """
            SELECT id, timestamp, prediction, true_label, confidence
            FROM prediction_logs
            ORDER BY timestamp DESC
            LIMIT %s OFFSET %s
            """
            
            cursor.execute(query, (limit, offset))
            rows = cursor.fetchall()
            
            # Convert to list of dictionaries
            predictions = []
            for row in rows:
                predictions.append({
                    "id": row[0],
                    "timestamp": row[1],
                    "prediction": row[2],
                    "true_label": row[3],
                    "confidence": row[4]
                })
            
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
                self.logger.info(f"Updated true label for prediction {prediction_id} to {true_label}")
            else:
                self.logger.warning(f"No prediction found with ID {prediction_id}")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error updating true label: {str(e)}")
            if conn:
                conn.close()
            return False

# Create singleton instance
prediction_service = PredictionService()