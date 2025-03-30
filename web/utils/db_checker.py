#!/usr/bin/env python3
# MNIST Digit Classifier
# Copyright (c) 2025
# File: utils/db_checker.py
# Description: Utility to check database contents
# Created: 2025-03-24

import os
import sys
import sqlite3
import json
from datetime import datetime
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
    db_path = os.path.join(data_dir, "mnist_app.db")
    return db_path


def check_database():
    """Check the database contents and display information."""
    db_path = get_db_path()

    print(f"Checking database at: {db_path}")

    if not os.path.exists(db_path):
        print("ERROR: Database file doesn't exist!")
        return

    print(
        f"Database file exists (size: {os.path.getsize(db_path) / 1024:.2f} KB)"
    )

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        print(f"Tables in database: {', '.join(tables)}")

        # Check predictions
        if "predictions" in tables:
            cursor.execute("SELECT COUNT(*) FROM predictions")
            count = cursor.fetchone()[0]
            print(f"Total predictions: {count}")

            if count > 0:
                cursor.execute(
                    """
                SELECT digit, COUNT(*) as count, AVG(confidence) as avg_conf
                FROM predictions
                GROUP BY digit
                ORDER BY digit
                """
                )
                print("\nPredictions by digit:")
                print("Digit | Count | Avg Confidence")
                print("-" * 30)
                for row in cursor.fetchall():
                    print(
                        f"{row['digit']:5d} | {row['count']:5d} | {row['avg_conf']:.4f}"
                    )

                # Get most recent predictions
                cursor.execute(
                    """
                SELECT id, digit, confidence, timestamp, input_type, user_correction
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT 5
                """
                )
                print("\nMost recent predictions:")
                print(
                    "ID | Digit | Confidence | Timestamp | Input Type | Correction"
                )
                print("-" * 80)
                for row in cursor.fetchall():
                    # Format timestamp if it's a string
                    if isinstance(row["timestamp"], str):
                        try:
                            timestamp = datetime.fromisoformat(
                                row["timestamp"]
                            )
                            timestamp_str = timestamp.strftime(
                                "%Y-%m-%d %H:%M:%S"
                            )
                        except ValueError:
                            timestamp_str = row["timestamp"]
                    else:
                        timestamp_str = row["timestamp"]

                    # Truncate ID for display
                    id_short = (
                        row["id"][:8] + "..."
                        if len(row["id"]) > 8
                        else row["id"]
                    )

                    print(
                        f"{id_short:11s} | {row['digit']:5d} | {row['confidence']:.4f} | {timestamp_str} | {row['input_type']:10s} | {row['user_correction'] if row['user_correction'] is not None else 'None'}"
                    )

        # Check settings
        if "settings" in tables:
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
                    value = row["value"]
                    if len(value) > 30:
                        value = value[:27] + "..."
                    print(
                        f"{row['category']:10s} | {row['key']:15s} | {value}"
                    )

        conn.close()

    except sqlite3.Error as e:
        print(f"ERROR accessing database: {e}")


if __name__ == "__main__":
    check_database()
