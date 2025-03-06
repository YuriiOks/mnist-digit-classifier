import os
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime
import logging

# Set up logging
logger = logging.getLogger(__name__)

def get_connection():
    """
    Create a connection to the PostgreSQL database.
    
    Returns:
        connection: PostgreSQL database connection
    """
    try:
        # Get database configuration from environment variables
        db_host = os.environ.get("DB_HOST", "db")
        db_name = os.environ.get("DB_NAME", "digit_predictions")
        db_user = os.environ.get("DB_USER", "mnist_app")
        db_password = os.environ.get("DB_PASSWORD", "secure_password")
        
        logger.info(f"Connecting to database: {db_host}/{db_name}")
        
        # Create connection
        connection = psycopg2.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password
        )
        
        # Test connection
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
            
        logger.info("Database connection successful")
        return connection
    except Exception as e:
        logger.error(f"Database connection error: {str(e)}", exc_info=True)
        raise

def log_prediction(connection, prediction, true_label, confidence):
    """
    Log a prediction to the database.
    
    Args:
        connection: Database connection
        prediction: The model's prediction (int)
        true_label: The true label provided by user feedback (int)
        confidence: The model's confidence in its prediction (float)
    """
    try:
        if connection is None:
            logger.warning("No database connection. Prediction not logged.")
            return
            
        with connection.cursor() as cursor:
            cursor.execute(
                '''
                INSERT INTO prediction_logs (prediction, true_label, confidence)
                VALUES (%s, %s, %s)
                ''',
                (prediction, true_label, confidence)
            )
        connection.commit()
        logger.info(f"Logged prediction: {prediction} (true: {true_label}) with confidence {confidence:.2f}")
    except Exception as e:
        logger.error(f"Error logging prediction: {str(e)}", exc_info=True)
        connection.rollback()
        raise

def get_prediction_history(connection, limit=10):
    """
    Get prediction history from the database.
    
    Args:
        connection: Database connection
        limit: Maximum number of records to return
        
    Returns:
        list: List of prediction history records
    """
    try:
        if connection is None:
            logger.warning("No database connection. Cannot retrieve history.")
            return []
            
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                '''
                SELECT * FROM prediction_logs
                ORDER BY timestamp DESC
                LIMIT %s
                ''',
                (limit,)
            )
            history = cursor.fetchall()
        
        logger.info(f"Retrieved {len(history)} history records")
        return history
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}", exc_info=True)
        return [] 