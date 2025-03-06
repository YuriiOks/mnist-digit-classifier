import os
import streamlit as st

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
        """Load CSS files from static directory.
        
        Args:
            css_files: List of CSS file paths relative to static directory
        """
        app_dir = ResourceLoader.get_app_dir()
        for css_file in css_files:
            css_path = os.path.join(app_dir, "static", css_file)
            if os.path.exists(css_path):
                with open(css_path, "r") as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            else:
                st.warning(f"CSS file not found: {css_path}")
    
    @staticmethod
    def load_js(js_files):
        """Load JavaScript files from static/js directory.
        
        Args:
            js_files: List of JS file names in the static/js directory
        """
        app_dir = ResourceLoader.get_app_dir()
        js_code = ""
        for js_file in js_files:
            js_path = os.path.join(app_dir, "static", "js", js_file)
            if os.path.exists(js_path):
                with open(js_path, "r") as f:
                    js_code += f.read() + "\n"
            else:
                st.warning(f"JS file not found: {js_path}")
        
        if js_code:
            st.markdown(f"<script>{js_code}</script>", unsafe_allow_html=True)
    
    @staticmethod
    def load_template(template_name, **kwargs):
        """Load and render HTML template with placeholders replaced.
        
        Args:
            template_name: Template file path relative to templates directory
            **kwargs: Key-value pairs to replace in the template
            
        Returns:
            Rendered HTML template string
        """
        app_dir = ResourceLoader.get_app_dir()
        template_path = os.path.join(app_dir, "templates", template_name)
        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                template = f.read()
            
            # Replace all placeholders in the template
            for key, value in kwargs.items():
                placeholder = f"{{{{{key}}}}}"
                template = template.replace(placeholder, str(value))
            
            return template
        else:
            st.warning(f"Template not found: {template_path}")
            return "" 