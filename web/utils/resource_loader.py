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
            css_files: List of CSS file paths relative to the static directory
        """
        try:
            all_css = ""
            # Always include fonts.css first if it exists
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fonts_path = os.path.join(base_dir, "static", "css", "fonts.css")
            
            try:
                with open(fonts_path, "r", encoding="utf-8") as f:
                    all_css += f.read() + "\n"
            except:
                pass
                
            # Then load the requested CSS files
            for css_file in css_files:
                css_path = os.path.join(base_dir, "static", css_file)
                
                with open(css_path, "r", encoding="utf-8") as f:
                    all_css += f.read() + "\n"
            
            # Inject all CSS at once
            if all_css:
                st.markdown(f"<style>{all_css}</style>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading CSS: {str(e)}")
    
    @staticmethod
    def load_js(js_files):
        """Load JavaScript files and inject them into the Streamlit app.
        
        Args:
            js_files: List of JS file paths relative to the static directory
        """
        try:
            all_js = ""
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            
            for js_file in js_files:
                js_path = os.path.join(base_dir, "static", js_file)
                
                with open(js_path, "r", encoding="utf-8") as f:
                    all_js += f.read() + "\n"
            
            # Inject all JS at once
            if all_js:
                st.markdown(f"<script>{all_js}</script>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading JavaScript: {str(e)}")
    
    @staticmethod
    def load_template(template_path, **kwargs):
        """Load an HTML template and replace variables with provided values.
        
        Args:
            template_path: Path to the template file relative to the templates directory
            **kwargs: Key-value pairs to replace in the template
            
        Returns:
            The processed template with variables replaced
        """
        try:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            template_full_path = os.path.join(base_dir, "templates", template_path)
            
            with open(template_full_path, "r", encoding="utf-8") as f:
                template = f.read()
            
            # Replace variables in the template
            for key, value in kwargs.items():
                template = template.replace(f"${{{key}}}", str(value))
            
            return template
        except Exception as e:
            st.error(f"Error loading template {template_path}: {str(e)}")
            return ""
    
    @staticmethod
    def render_content_card(title=None, content="", card_class=""):
        """Render a content card with the given title and content.
        
        Args:
            title: Optional card title (string or HTML)
            content: Card content (string or HTML)
            card_class: Additional CSS classes for the card
            
        Returns:
            HTML string for the content card
        """
        # Prepare the class string
        class_str = ""
        if card_class:
            class_str = f" {card_class}"
        
        # Prepare the header
        header_html = ""
        if title:
            header_html = f"<h2 class='card-title'>{title}</h2>"
        
        # Load the template
        try:
            template = ResourceLoader.load_template(
                "components/content_card.html",
                CARD_CLASS=class_str,
                CARD_HEADER=header_html,
                CARD_CONTENT=content
            )
            return template
        except Exception as e:
            # Fallback simple version
            return f"""
            <div class="content-card{class_str}">
                {header_html}
                <div class="card-content">
                    {content}
                </div>
            </div>
            """ 