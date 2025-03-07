import streamlit as st
from streamlit_option_menu import option_menu
from utils.theme_manager import ThemeManager
from utils.resource_loader import ResourceLoader

def render_sidebar():
    """Render the sidebar with navigation and theme toggle."""
    with st.sidebar:
        # st.markdown('<div class="navigation-title">Navigation</div>', unsafe_allow_html=True)
        
        # Load sidebar CSS
        ResourceLoader.load_css(["css/components/sidebar.css"])
        
        # Theme toggle in sidebar using template
        theme_icon = ThemeManager.get_theme_icon()
        sidebar_toggle_html = ResourceLoader.load_template(
            "components/sidebar_toggle.html",
            THEME_ICON=theme_icon
        )
        st.markdown(sidebar_toggle_html, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
        
        # Option menu for navigation
        selected = option_menu(
            "", 
            ["Home", "Drawing", "History", "Settings"], 
            icons=["house", "pencil", "clock-history", "gear"], 
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "5px", "background-color": "transparent"},
                "icon": {"color": "#4CA1AF", "font-size": "20px"}, 
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "0px", 
                    "--hover-color": "#eee",
                    "transition": "all 0.3s ease",
                },
                "nav-link-selected": {
                    "background-color": "#2C3E50",
                    "font-weight": "600",
                },
                "menu-title": {
                    "margin-bottom": "15px",
                    "font-size": "18px",
                    "color": "#2C3E50" if not st.session_state.dark_mode else "#64c4d2"
                }
            }
        )
        
        return selected

def render_theme_toggle():
    """Render the theme toggle button in the sidebar."""
    theme_icon = "☀️" if not st.session_state.dark_mode else "🌙"
    st.markdown(f"""
    <style>
    /* Match the sidebar button styling to the floating button */
    .theme-toggle-sidebar-container > div > .stButton > button {{
        width: 45px !important;
        height: 45px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        margin: 0 auto !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        background: rgba(0, 0, 0, 0.2) !important;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3) !important;
        border: 2px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(5px) !important;
        -webkit-backdrop-filter: blur(5px) !important;
    }}
    
    /* Dark mode version */
    .dark .theme-toggle-sidebar-container > div > .stButton > button {{
        background: rgba(255, 255, 255, 0.2) !important;
        border: 2px solid rgba(255, 255, 255, 0.2) !important;
    }}
    
    /* Hover state */
    .theme-toggle-sidebar-container > div > .stButton > button:hover {{
        transform: scale(1.1) !important;
        background: rgba(0, 0, 0, 0.3) !important;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4) !important;
    }}
    
    /* Active state */
    .theme-toggle-sidebar-container > div > .stButton > button:active {{
        transform: scale(0.95) !important;
    }}
    
    /* Animation for the icon */
    @keyframes float {{
        0% {{ transform: translateY(0px); }}
        50% {{ transform: translateY(-3px); }}
        100% {{ transform: translateY(0px); }}
    }}
    
    /* Apply animation to the icon */
    .theme-toggle-sidebar-container > div > .stButton > button > div > p {{
        animation: float 2s ease-in-out infinite !important;
        margin: 0 !important;
    }}
    </style>
    <div class="theme-toggle-sidebar-container">
    """, unsafe_allow_html=True)
    
    if st.button(theme_icon, key="theme_toggle_sidebar", help="Toggle dark/light mode"):
        ThemeManager.toggle_dark_mode()
    
    st.markdown("</div>", unsafe_allow_html=True) 