# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: setup.py
# Description: Setup script for the MNIST Digit Classifier project
# Created: 2025-03-28
# Updated: 2025-03-30

from setuptools import setup, find_packages

setup(
    name="mnist_classifier_project",  # Choose a suitable name
    version="0.1.0",
    packages=find_packages(
        include=["utils*", "scripts*", "model*", "web*"]
    )
)
