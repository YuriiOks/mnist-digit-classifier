import os
import streamlit as st
import string

class ResourceLoader:
    """Utility class for loading external resources like CSS, JS, and HTML templates."""
    
    @staticmethod
    def get_app_dir():
        """Get the application's root directory."""
        current_file = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to reach app root from utils directory
        return os.path.dirname(current_file)
    
    @staticmethod
    def load_css(css_files):
        """Load CSS files and inject them into the Streamlit app.
        
        Args:
            css_files: List of CSS files to load (relative to static/css)
        """
        try:
            for css_file in css_files:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                css_path = os.path.join(base_dir, "static", css_file)
                
                with open(css_path, "r", encoding="utf-8") as f:
                    css = f.read()
                
                # Inject the CSS into the app
                st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading CSS: {str(e)}")
    
    @staticmethod
    def load_js(js_files):
        """Load JavaScript files and inject them into the Streamlit app.
        
        Args:
            js_files: List of JS files to load (relative to static/js)
        """
        try:
            for js_file in js_files:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                js_path = os.path.join(base_dir, "static", "js", js_file)
                
                with open(js_path, "r", encoding="utf-8") as f:
                    js = f.read()
                
                # Inject the JavaScript into the app
                st.markdown(f"<script>{js}</script>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading JavaScript: {str(e)}")
    
    @staticmethod
    def load_template(template_path, **kwargs):
        """Load an HTML template and replace any placeholders.
        
        Args:
            template_path: Path to the template relative to templates directory
            **kwargs: Key-value pairs for placeholder replacement
            
        Returns:
            Processed HTML content with placeholders replaced
        """
        try:
            # Build the full path to the template
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            full_path = os.path.join(base_dir, "templates", template_path)
            
            # Read the template file
            with open(full_path, "r", encoding="utf-8") as f:
                template = f.read()
            
            # Replace placeholders using string.Template
            if kwargs:
                template = string.Template(template).substitute(kwargs)
            
            return template
        except Exception as e:
            st.error(f"Error loading template {template_path}: {str(e)}")
            return f"<div>Error loading template: {str(e)}</div>" 