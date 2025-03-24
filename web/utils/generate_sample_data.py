#!/usr/bin/env python3
# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/generate_sample_data.py
# Description: Utility to generate sample prediction data
# Created: 2025-03-24

import os
import sys
import random
import uuid
import sqlite3
import json
from datetime import datetime, timedelta
from pathlib import Path

def get_db_path():
    """Get the path to the database file."""
    # In Docker, use /app/data
    docker_path = Path("/app/data")
    if docker_path.exists():
        db_path = docker_path / "mnist_app.db"
        return str(db_path)
    
    # Otherwise use ~/.mnist_app
    home_dir = os.path.expanduser("~")
    data_dir = os.path.join(home_dir, ".mnist_app")
    os.makedirs(data_dir, exist_ok=True)
    db_path = os.path.join(data_dir, "mnist_app.db")
    return db_path

def create_tables(conn):
    """Create necessary tables if they don't exist."""
    cursor = conn.cursor()
    
    # Predictions table
    cursor.execute('''
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
    ''')
    
    # Settings table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS settings (
        category TEXT NOT NULL,
        key TEXT NOT NULL,
        value TEXT NOT NULL,
        PRIMARY KEY (category, key)
    )
    ''')
    
    conn.commit()
    print("Tables created successfully")

def add_prediction(conn, prediction_data):
    """Add a prediction to the database."""
    cursor = conn.cursor()
    
    # Generate ID if not provided
    pred_id = prediction_data.get('id') or str(uuid.uuid4())
    
    # Prepare metadata as JSON if present
    metadata = prediction_data.get('metadata')
    if metadata and isinstance(metadata, dict):
        metadata = json.dumps(metadata)
    
    # Extract image data if present
    image_data = prediction_data.get('image_data')
    
    # Convert timestamp to string if it's a datetime object
    timestamp = prediction_data.get('timestamp')
    if isinstance(timestamp, datetime):
        timestamp = timestamp.isoformat()
    elif not timestamp:
        timestamp = datetime.now().isoformat()
    
    cursor.execute('''
    INSERT INTO predictions (
        id, digit, confidence, timestamp, input_type, 
        image_data, metadata, user_correction
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        pred_id,
        prediction_data.get('digit', 0),
        prediction_data.get('confidence', 0.0),
        timestamp,
        prediction_data.get('input_type', 'unknown'),
        image_data,
        metadata,
        prediction_data.get('user_correction')
    ))
    
    conn.commit()
    return pred_id

def generate_sample_data(num_entries=20):
    """Generate sample prediction data."""
    print(f"Generating {num_entries} sample prediction entries...")
    
    db_path = get_db_path()
    print(f"Using database at: {db_path}")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    
    # Create tables if needed
    create_tables(conn)
    
    # Get current time
    now = datetime.now()
    
    # Input types
    input_types = ["canvas", "upload", "url"]
    
    # Generate entries
    for i in range(num_entries):
        # Generate a random digit (0-9)
        digit = random.randint(0, 9)
        
        # Generate a confidence score (0.5-1.0 with bias toward higher confidence)
        confidence = 0.5 + (random.random() * 0.5)
        
        # Generate a timestamp (within the last 30 days)
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 23)
        minutes_ago = random.randint(0, 59)
        timestamp = now - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        
        # Select a random input type
        input_type = random.choice(input_types)
        
        # Create prediction data
        prediction = {
            "id": str(uuid.uuid4()),
            "digit": digit,
            "confidence": confidence,
            "timestamp": timestamp,
            "input_type": input_type,
            "metadata": {
                "sample_data": True,
                "description": f"Sample prediction #{i+1}"
            }
        }
        
        # Randomly add a user correction (10% chance)
        if random.random() < 0.1:
            # Make sure correction is different from predicted digit
            correction = (digit + random.randint(1, 9)) % 10
            prediction["user_correction"] = correction
        
        # Add to database
        try:
            add_prediction(conn, prediction)
            print(f"Added prediction {i+1}/{num_entries}: digit={digit}, confidence={confidence:.4f}, input_type={input_type}")
        except Exception as e:
            print(f"ERROR adding prediction: {e}")
    
    # Close connection
    conn.close()
    
    print("Sample data generation complete!")

def check_database():
    """Check the database contents and display information."""
    db_path = get_db_path()
    
    print(f"Checking database at: {db_path}")
    
    if not os.path.exists(db_path):
        print("ERROR: Database file doesn't exist!")
        return
    
    print(f"Database file exists (size: {os.path.getsize(db_path) / 1024:.2f} KB)")
    
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        print(f"Tables in database: {', '.join(tables)}")
        
        # Check predictions
        if 'predictions' in tables:
            cursor.execute("SELECT COUNT(*) FROM predictions")
            count = cursor.fetchone()[0]
            print(f"Total predictions: {count}")
            
            if count > 0:
                cursor.execute("""
                SELECT digit, COUNT(*) as count, AVG(confidence) as avg_conf
                FROM predictions
                GROUP BY digit
                ORDER BY digit
                """)
                print("\nPredictions by digit:")
                print("Digit | Count | Avg Confidence")
                print("-" * 30)
                for row in cursor.fetchall():
                    print(f"{row['digit']:5d} | {row['count']:5d} | {row['avg_conf']:.4f}")
                
                # Get most recent predictions
                cursor.execute("""
                SELECT id, digit, confidence, timestamp, input_type, user_correction
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT 5
                """)
                print("\nMost recent predictions:")
                print("ID | Digit | Confidence | Timestamp | Input Type | Correction")
                print("-" * 80)
                for row in cursor.fetchall():
                    # Format timestamp if it's a string
                    if isinstance(row['timestamp'], str):
                        try:
                            timestamp = datetime.fromisoformat(row['timestamp'])
                            timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        except ValueError:
                            timestamp_str = row['timestamp']
                    else:
                        timestamp_str = row['timestamp']
                    
                    # Truncate ID for display
                    id_short = row['id'][:8] + "..." if len(row['id']) > 8 else row['id']
                    
                    print(f"{id_short:11s} | {row['digit']:5d} | {row['confidence']:.4f} | {timestamp_str} | {row['input_type']:10s} | {row['user_correction'] if row['user_correction'] is not None else 'None'}")
        
        # Check settings
        if 'settings' in tables:
            cursor.execute("SELECT COUNT(*) FROM settings")
            count = cursor.fetchone()[0]
            print(f"\nTotal settings: {count}")
            
            if count > 0:
                cursor.execute("SELECT category, key, value FROM settings")
                print("\nSettings:")
                print("Category | Key | Value")
                print("-" * 50)
                for row in cursor.fetchall():
                    # Truncate value for display
                    value = row['value']
                    if len(value) > 30:
                        value = value[:27] + "..."
                    print(f"{row['category']:10s} | {row['key']:15s} | {value}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"ERROR accessing database: {e}")

if __name__ == "__main__":
    # Get number of entries from command line argument
    num_entries = 20
    if len(sys.argv) > 1:
        try:
            num_entries = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of entries: {sys.argv[1]}. Using default: 20")
    
    generate_sample_data(num_entries)
    
    # Run the database checker to verify
    print("\nVerifying database contents:")
    check_database()