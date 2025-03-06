import streamlit as st

def render_header():
    """Render the application header with logo and title."""
    
    # Initialize dark mode in session state
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # Define a callback function to toggle dark mode
    def toggle_dark_mode():
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()
    
    # Place a hidden button in the container that will be clicked by our JavaScript
    with st.container():
        st.button("Toggle Dark Mode", key="toggle_dark_mode_button", on_click=toggle_dark_mode, help="Toggle dark mode")
    
    # Header styling with streamlit native components
    st.markdown("""
    <style>
    /* Header styles */
    .header-container {
        display: flex;
        justify-content: center;
        align-items: center;
        background: linear-gradient(to right, #2c3e50, #4ca1af);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative;
    }
    .header-logo-title {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .header-logo {
        font-size: 2rem;
        margin-right: 10px;
        animation: pulse 2s infinite ease-in-out;
    }
    .header-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* Dark mode toggle positioned within header */
    .dark-mode-toggle {
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background: rgba(0, 0, 0, 0.2);
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        font-size: 1.2rem;
    }
    .dark-mode-toggle:hover {
        transform: translateY(-50%) scale(1.1);
        background: rgba(0, 0, 0, 0.3);
    }
    
    /* Completely hide toggle button */
    button[kind="secondary"] {
        display: none !important;
    }
    
    /* Hide button container div */
    div[data-testid="element-container"] button[kind="secondary"] {
        display: none !important;
    }
    
    div[data-testid="element-container"]:has(button[kind="secondary"]) {
        height: 0 !important;
        min-height: 0 !important;
        visibility: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
        position: absolute !important;
        pointer-events: none !important;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header content with integrated dark mode toggle
    st.markdown(f"""
    <div class="header-container">
        <div class="header-logo-title">
            <div class="header-logo">✏️</div>
            <h1 class="header-title">MNIST Digit Classifier</h1>
        </div>
        <div class="dark-mode-toggle" onclick="document.getElementById('toggle_dark_mode_button').click()">
            {" 🌙" if st.session_state.dark_mode else " ☀️"}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Apply dark mode styles
    if st.session_state.dark_mode:
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
        .stButton > button {
            background-color: #2c3e50;
            color: white;
        }
        .stTextInput > div > div {
            background-color: #333;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
