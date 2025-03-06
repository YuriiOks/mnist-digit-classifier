import streamlit as st
import os
from pathlib import Path

def render_theme_toggle():
    """Render the dark mode toggle component."""
    
    # Initialize dark mode in session state
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # Define a callback function to toggle dark mode
    def toggle_dark_mode():
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    # Determine component directory
    component_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(os.path.dirname(os.path.dirname(component_dir)))
    
    # Load CSS for theme toggle
    css_path = os.path.join(app_dir, "web", "static", "css", "components", "theme_toggle.css")
    if os.path.exists(css_path):
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Place a hidden button that will be clicked by JavaScript
    with st.container():
        st.button("Toggle Dark Mode", key="toggle_dark_mode_button", 
                 on_click=toggle_dark_mode, help="Toggle dark mode")
    
    # Load and format the theme toggle template
    template_path = os.path.join(app_dir, "web", "templates", "components", "theme_toggle.html")
    if os.path.exists(template_path):
        with open(template_path, "r") as f:
            toggle_template = f.read()
        
        # Replace placeholder with the correct icon based on current mode
        toggle_template = toggle_template.replace(
            "{{THEME_ICON}}", 
            " 🌙" if st.session_state.dark_mode else " ☀️"
        )
        
        # Render the theme toggle
        st.markdown(toggle_template, unsafe_allow_html=True)
    else:
        # Fallback to inline HTML if template is not found
        st.markdown(f"""
        <div class="dark-mode-toggle" onclick="document.getElementById('toggle_dark_mode_button').click()">
            {" 🌙" if st.session_state.dark_mode else " ☀️"}
        </div>
        <script>
            function toggleDarkMode() {{
                document.getElementById('toggle_dark_mode_button').click();
            }}
        </script>
        """, unsafe_allow_html=True)
    
    # Apply dark mode styles if enabled
    if st.session_state.dark_mode:
        dark_mode_css_path = os.path.join(app_dir, "web", "static", "css", "themes", "dark_mode.css")
        if os.path.exists(dark_mode_css_path):
            with open(dark_mode_css_path, "r") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        else:
            # Fallback to inline dark mode styles
            st.markdown("""
            <style>
            .stApp {
                background-color: #121212;
                color: #f0f0f0;
            }
            .header-container {
                background: linear-gradient(to right, #1a1a2e, #16213e);
            }
            .footer-container {
                background: linear-gradient(to right, #1a1a2e, #16213e);
            }
            .dark-mode-toggle {
                background: rgba(255, 255, 255, 0.2);
            }
            </style>
            """, unsafe_allow_html=True) 