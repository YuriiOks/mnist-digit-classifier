# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: core/database/__init__.py
# Description: Database package initialization
# Created: 2025-03-24

"""Database management for the MNIST Digit Classifier."""

from core.database.db_manager import DatabaseManager, db_manager

__all__ = ["DatabaseManager", "db_manager", "initialize_database"]


def initialize_database():
    """Initialize the database and ensure required tables exist."""
    # The singleton instance is created and initialized automatically when imported
    # This function mainly serves as a hook for explicit initialization in app startup
    return db_manager
