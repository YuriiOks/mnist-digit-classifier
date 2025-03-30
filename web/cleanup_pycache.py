#!/usr/bin/env python3
# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/cleanup_pycache.py
# Description: [Brief description of the file's purpose]
# Created: 2025-03-30
# Updated: 2025-03-30

"""
Script to recursively delete __pycache__ directories and .pyc files
"""

import os
import shutil
import sys


def cleanup_pycache(start_path="."):
    """Recursively find and delete __pycache__ directories and .pyc files"""
    deleted_dirs = 0
    deleted_files = 0

    print(
        f"Searching for Python cache files starting from: {os.path.abspath(start_path)}"
    )

    # Walk through all directories starting from start_path
    for root, dirs, files in os.walk(start_path, topdown=False):
        # Delete .pyc files
        for file in files:
            if file.endswith(".pyc"):
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                    deleted_files += 1
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

        # Delete __pycache__ directories
        for dir in dirs:
            if dir == "__pycache__":
                dir_path = os.path.join(root, dir)
                try:
                    shutil.rmtree(dir_path)
                    print(f"Deleted directory: {dir_path}")
                    deleted_dirs += 1
                except Exception as e:
                    print(f"Error deleting {dir_path}: {e}")

    print(f"\nCleanup complete!")
    print(
        f"Deleted {deleted_dirs} __pycache__ directories and {deleted_files} .pyc files"
    )


if __name__ == "__main__":
    # Use command line argument as start path if provided, otherwise use current directory
    start_path = sys.argv[1] if len(sys.argv) > 1 else "."
    cleanup_pycache(start_path)
