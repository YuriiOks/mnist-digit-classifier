import os


def setup_directory_structure():
    """Ensure all required directories exist."""
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define directories that need to exist
    dirs = [
        os.path.join(base_dir, "static", "css", "components"),
        os.path.join(base_dir, "static", "css", "themes"),
        os.path.join(base_dir, "static", "js"),
        os.path.join(base_dir, "templates", "components")
    ]
    
    # Create directories if they don't exist
    for directory in dirs:
        os.makedirs(directory, exist_ok=True)
        
    print("Directory structure setup complete!")


if __name__ == "__main__":
    setup_directory_structure() 