from setuptools import setup, find_packages

setup(
    name="mnist_classifier_project",  # Choose a suitable name
    version="0.1.0",
    packages=find_packages(
        include=["utils*", "scripts*", "model*", "web*"]
    ),  # Include your main packages
    # You might add author, description etc. later
)
