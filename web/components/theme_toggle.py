import streamlit as st

def theme_toggle():
    """Renders a sleek theme toggle button that persists user preference."""
    if 'theme' not in st.session_state:
        # Default to system preference using media query
        st.session_state.theme = 'light'
    
    col1, col2 = st.columns([0.8, 0.2])
    with col2:
        theme_icon = "🌙" if st.session_state.theme == "light" else "☀️"
        if st.button(theme_icon, key="theme_toggle"):
            st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
            st.experimental_rerun()
    
    # Inject theme-specific CSS
    theme_css = """
    <style>
    :root {
        --bg-primary: %s;
        --bg-secondary: %s;
        --text-primary: %s;
        --text-secondary: %s;
        --accent-primary: %s;
        --accent-secondary: %s;
        --border-color: %s;
        --shadow: %s;
    }
    </style>
    """ % (
        "#f8f9fa" if st.session_state.theme == "light" else "#121212",
        "#ffffff" if st.session_state.theme == "light" else "#1e1e1e",
        "#212529" if st.session_state.theme == "light" else "#e0e0e0",
        "#495057" if st.session_state.theme == "light" else "#ababab",
        "#4361ee" if st.session_state.theme == "light" else "#4cc9f0",
        "#3a0ca3" if st.session_state.theme == "light" else "#7209b7",
        "#dee2e6" if st.session_state.theme == "light" else "#333333",
        "0 4px 20px rgba(0,0,0,0.08)" if st.session_state.theme == "light" else "0 4px 20px rgba(0,0,0,0.4)"
    )
    
    st.markdown(theme_css, unsafe_allow_html=True) 