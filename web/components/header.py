# Updated header.py with Google Font import
import streamlit as st
import os

def render_header(title, icon):
    """Render the application header with logo and theme toggle."""
    # Apply theme toggle
    if 'theme' not in st.session_state:
        st.session_state.theme = 'light'
    
    # Add Google Fonts import
    google_fonts_css = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    """
    st.markdown(google_fonts_css, unsafe_allow_html=True)
    
    # Theme-specific CSS with updated font family
    theme_css = """
    <style>
    :root {
        --bg-primary: %s;
        --bg-secondary: %s;
        --text-primary: %s;
        --text-primary-rgb: %s;
        --text-secondary: %s;
        --accent-primary: %s;
        --accent-secondary: %s;
        --border-color: %s;
        --shadow: %s;
        --font-family: 'Poppins', sans-serif;
    }

    /* Apply font family to all elements */
    body, button, input, select, textarea, .stMarkdown, .stText, h1, h2, h3, h4, h5, h6 {
        font-family: var(--font-family) !important;
    }

    /* Force dark mode to apply immediately */
    body {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
        transition: background-color 0.3s ease, color 0.3s ease;
    }

    .stApp {
        background-color: var(--bg-primary) !important;
    }

    /* Override Streamlit elements with theme variables */
    .stTextInput > div > div {
        background-color: var(--bg-secondary) !important;
        color: var(--text-primary) !important;
    }
    
    .stButton > button {
        background-color: var(--accent-primary) !important;
        color: white !important;
        font-family: var(--font-family) !important;
    }
    
    .css-1cpxqw2 {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Enforce dark mode for Streamlit components */
    %s
    </style>
    """ % (
        "#f8f9fa" if st.session_state.theme == "light" else "#121212",
        "#ffffff" if st.session_state.theme == "light" else "#1e1e1e",
        "#212529" if st.session_state.theme == "light" else "#e0e0e0",
        "33, 37, 41" if st.session_state.theme == "light" else "224, 224, 224",
        "#495057" if st.session_state.theme == "light" else "#ababab",
        "#4361ee" if st.session_state.theme == "light" else "#4cc9f0",
        "#3a0ca3" if st.session_state.theme == "light" else "#7209b7",
        "#dee2e6" if st.session_state.theme == "light" else "#333333",
        "0 4px 20px rgba(0,0,0,0.08)" if st.session_state.theme == "light" else "0 4px 20px rgba(0,0,0,0.4)",
        # Additional dark mode overrides for Streamlit components
        """
        .dark-mode .stTextInput input, .dark-mode .stSelectbox select {
            background-color: #1e1e1e !important;
            color: #e0e0e0 !important;
        }
        
        .dark-mode .stTabs [role="tab"][aria-selected="true"] {
            background-color: #333333 !important;
            color: #e0e0e0 !important;
        }
        
        .dark-mode .stTabs [role="tab"] {
            color: #ababab !important;
        }
        """ if st.session_state.theme == "dark" else ""
    )
    
    st.markdown(theme_css, unsafe_allow_html=True)
    
    # Load base CSS
    css_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "styles.css")
    try:
        with open(css_path, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading CSS: {str(e)}")
        try:
            # Fallback: Use relative path
            with open("static/styles.css", "r") as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        except Exception as e2:
            st.error(f"Error with fallback CSS: {str(e2)}")
    
    # Simple theme toggle with direct Streamlit functionality
    cols = st.columns([6, 1])
    with cols[0]:
        st.markdown(f"<h1 class='header-title'>{icon} {title}</h1>", unsafe_allow_html=True)
    with cols[1]:
        current_theme = st.session_state.theme
        if st.button("🌙" if current_theme == "light" else "☀️", key="theme_toggle"):
            # Toggle theme
            st.session_state.theme = "dark" if current_theme == "light" else "light"
            # Force rerun to apply theme
            st.experimental_rerun()
    
    # Apply dark mode class to body
    if st.session_state.theme == "dark":
        st.markdown("""
        <script>
            document.body.classList.add('dark-mode');
        </script>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <script>
            document.body.classList.remove('dark-mode');
        </script>
        """, unsafe_allow_html=True)