import streamlit as st
import datetime
import os
from pathlib import Path

def render_footer():
    """Render the application footer with attribution and copyright."""
    
    # Determine component directory
    component_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(component_dir)))
    
    # Load CSS for footer
    css_path = os.path.join(app_dir, "web", "static", "css", "components", "footer.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Get current year for copyright
    current_year = datetime.datetime.now().year
    
    # Load footer template
    template_path = os.path.join(app_dir, "web", "templates", "footer.html")
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            footer_template = f.read()
        
        # Replace placeholders in the template
        footer_template = footer_template.replace("{{CURRENT_YEAR}}", str(current_year))
        
        # Force a bit of space before the footer
        st.markdown("<div style='height: 30px'></div>", unsafe_allow_html=True)
        
        # Render the footer
        st.markdown(footer_template, unsafe_allow_html=True)
    else:
        # Fallback to inline HTML if template is not found
        st.markdown(f"""
        <div style='height: 30px'></div>
        <div class="footer-container">
            <div class="footer-content">
                <div class="footer-info">
                    <div class="project-name">MNIST Digit Classifier</div>
                    <div class="developer-info">
                        Developed by <a href="https://github.com/YuriiOks" target="_blank">YuriiOks</a>
                    </div>
                </div>
                <div class="copyright-info">
                    © {current_year} All Rights Reserved
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True) 