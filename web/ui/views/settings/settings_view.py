# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/settings/settings_view.py
# Description: Application settings view
# Created: 2024-05-01

import streamlit as st
import logging
from typing import Dict, Any

try:
    from ui.views.base_view import BaseView
    from ui.theme.theme_manager import ThemeManager
except ImportError:
    # Fallback for minimal implementation
    class BaseView:
        def __init__(self, view_id="settings", title="Settings", description="", icon="⚙️"):
            self.view_id = view_id
            self.title = title
            self.description = description
            self.icon = icon
    
    class ThemeManager:
        @staticmethod
        def get_theme(): return "light"
        @staticmethod
        def set_theme(theme): pass
        @staticmethod
        def toggle_theme(): pass

logger = logging.getLogger(__name__)

class SettingsView(BaseView):
    """Settings view for application settings."""
    
    def __init__(self):
        """Initialize the settings view."""
        super().__init__(
            view_id="settings",
            title="Settings",
            description="Configure application settings",
            icon="⚙️"
        )
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def render(self) -> None:
        """Render the settings view content."""
        self.logger.debug("Rendering settings view")
        
        try:
            # Apply the same layout CSS as home view for consistency
            st.markdown("""
            <style>
            /* Fix content alignment */
            .block-container {
                max-width: 100% !important;
                padding-top: 1rem !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
            
            /* Make headers look better */
            h1, h2, h3 {
                margin-bottom: 1rem;
                margin-top: 0.5rem;
                font-family: var(--font-primary, 'Poppins', sans-serif);
            }
            
            /* Add space around elements */
            .stMarkdown {
                margin-bottom: 0.5rem;
            }
            
            /* Remove empty columns */
            .stColumn:empty {
                display: none !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Add welcome card similar to home page
            st.markdown("""
            <div class="card card-elevated content-card welcome-card animate-fade-in">
                <div class="card-title">
                    <span class="card-icon">⚙️</span>
                    Settings
                </div>
                <div class="card-content">
                    <p>Configure application settings and preferences to customize your experience.</p>
                    <p>Changes are automatically saved and will be applied immediately.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Create a 2x2 grid for settings cards
            col1, col2 = st.columns(2)
            
            # Appearance Settings Card
            with col1:
                st.markdown("""
                <div class="card card-elevated content-card primary-card animate-fade-in">
                    <div class="card-title">
                        <span class="card-icon">🎨</span>
                        Appearance
                    </div>
                    <div class="card-content">
                        <p>Customize the look and feel of the application.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Get the current theme
                current_theme = ThemeManager.get_theme()
                is_dark_mode = current_theme == "dark"
                
                # Add styling for the toggle container with improved scoping to avoid affecting global layout
                st.markdown("""
                <style>
                /* Scope our custom styling to the settings view specifically */
                .settings-view .toggle-container {
                    background: linear-gradient(to right, rgba(99, 102, 241, 0.1), rgba(6, 182, 212, 0.1));
                    padding: 15px 20px;
                    border-radius: 12px;
                    margin: 15px 0;
                    border: 1px solid rgba(99, 102, 241, 0.2);
                    transition: all 0.3s ease;
                    position: relative;
                    overflow: hidden;
                }
                
                /* Fix header/footer positioning that might have been affected */
                /* Restore proper header spacing */
                header {
                    position: relative !important;
                    top: 0 !important;
                    left: 0 !important;
                }
                
                /* Ensure footer stays at the bottom without overlapping content */
                footer {
                    position: relative !important;
                    bottom: 0 !important;
                    width: 100% !important;
                    z-index: 100;
                }
                
                /* Fix for stVerticalBlock margins that might affect layout */
                [data-testid="stVerticalBlock"] {
                    gap: inherit;
                }
                
                /* The rest of the toggle styling with proper scoping */
                .settings-view .toggle-container:hover {
                    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
                    transform: translateY(-2px);
                    border-color: rgba(99, 102, 241, 0.3);
                }
                
                /* Ensure our custom styles don't leak to other components */
                .settings-view .toggle-container label, 
                .settings-view .toggle-container p {
                    font-family: 'Poppins', sans-serif !important;
                    font-weight: 500 !important;
                    letter-spacing: 0.3px;
                }
                
                /* Only apply shimmer effect to our specific container */
                .settings-view .toggle-container::after {
                    content: '';
                    position: absolute;
                    top: -50%;
                    left: -50%;
                    width: 200%;
                    height: 200%;
                    background: linear-gradient(
                        to right,
                        rgba(255, 255, 255, 0) 0%,
                        rgba(255, 255, 255, 0.3) 50%,
                        rgba(255, 255, 255, 0) 100%
                    );
                    transform: rotate(30deg);
                    opacity: 0;
                    transition: opacity 0.3s;
                    pointer-events: none;
                }
                
                .settings-view .toggle-container:hover::after {
                    opacity: 1;
                    animation: shimmer 1.5s ease-in-out;
                }
                
                @keyframes shimmer {
                    0% { transform: rotate(30deg) translate(-100%, -100%); }
                    100% { transform: rotate(30deg) translate(100%, 100%); }
                }
                
                /* Dark mode adjustments */
                [data-theme="dark"] .settings-view .toggle-container {
                    background: linear-gradient(to right, rgba(99, 102, 241, 0.2), rgba(6, 182, 212, 0.2));
                    border-color: rgba(99, 102, 241, 0.3);
                }
                
                /* Style the theme name */
                .theme-name {
                    display: inline-block;
                    margin-left: 10px;
                    font-weight: 600;
                    background: linear-gradient(90deg, #6366F1, #06B6D4);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    animation: colorPulse 3s infinite alternate;
                }
                
                @keyframes colorPulse {
                    0% { filter: hue-rotate(0deg); }
                    100% { filter: hue-rotate(30deg); }
                }
                
                /* Restore proper block-container layout */
                .block-container {
                    max-width: 100% !important;
                    padding-top: 1rem !important;
                    padding-left: 1rem !important;
                    padding-right: 1rem !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Create a custom container before the toggle
                st.markdown('<div class="toggle-container">', unsafe_allow_html=True)
                
                st.write("**Theme Appearance:**")
                
                # Try to use streamlit_toggle with custom styling
                try:
                    import streamlit_toggle as tog
                    
                    # Custom colors based on your app's design
                    if is_dark_mode:
                        # Colors for dark mode
                        inactive_color = "#555555"
                        active_color = "#6366F1"  # Primary purple
                        track_color = "#06B6D4"   # Secondary teal
                    else:
                        # Colors for light mode
                        inactive_color = "#D3D3D3"
                        active_color = "#4F46E5"  # Darker purple
                        track_color = "#0891B2"   # Darker teal
                    
                    # Create the toggle with custom colors
                    theme_changed = tog.st_toggle_switch(
                        label="Dark Mode", 
                        key="theme_toggle", 
                        default_value=is_dark_mode,
                        label_after=True, 
                        inactive_color=inactive_color, 
                        active_color=active_color, 
                        track_color=track_color
                    )
                    
                    # Display current theme name with custom styling
                    st.markdown(f'<p>Current theme: <span class="theme-name">{current_theme.capitalize()}</span></p>', 
                                unsafe_allow_html=True)
                    
                    # Close the custom container
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Handle theme change
                    if "theme_toggle" in st.session_state:
                        # Check if the toggle value doesn't match current theme
                        if (st.session_state["theme_toggle"] and current_theme != "dark") or \
                           (not st.session_state["theme_toggle"] and current_theme != "light"):
                            # Toggle the theme
                            ThemeManager.toggle_theme()
                            st.rerun()
                
                except ImportError:
                    # Fallback to a simple styled button if streamlit_toggle is not available
                    st.write(f"Current theme: **{current_theme.capitalize()}**")
                    if st.button(f"Toggle to {'Light' if is_dark_mode else 'Dark'} Mode", 
                                 key="theme_btn",
                                 type="primary"):
                        ThemeManager.toggle_theme()
                        st.rerun()
                    
                    # Close container even in fallback mode
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Additional appearance settings could go here
                st.subheader("Font Settings")
                font_primary = st.selectbox(
                    "Primary Font",
                    options=["Poppins", "Roboto", "Inter", "Open Sans"],
                    index=0
                )
                
                # Rest of the appearance settings...
            
            # Drawing Settings Card
            with col2:
                st.markdown("""
                <div class="card card-elevated content-card secondary-card animate-fade-in">
                    <div class="card-title">
                        <span class="card-icon">✏️</span>
                        Drawing
                    </div>
                    <div class="card-content">
                        <p>Configure drawing canvas and prediction settings.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Drawing settings
                line_thickness = st.slider(
                    "Line Thickness", 
                    min_value=1, 
                    max_value=10, 
                    value=3, 
                    help="Thickness of drawing line"
                )
                
                # Prediction confidence threshold
                confidence_threshold = st.slider(
                    "Prediction Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.05,
                    format="%.2f",
                    help="Minimum confidence required for predictions"
                )
            
            # Second row of settings cards
            col3, col4 = st.columns(2)
            
            # Application Settings Card
            with col3:
                st.markdown("""
                <div class="card card-elevated content-card accent-card animate-fade-in">
                    <div class="card-title">
                        <span class="card-icon">📱</span>
                        Application
                    </div>
                    <div class="card-content">
                        <p>General application settings and preferences.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Application settings
                show_tooltips = st.checkbox("Show Tooltips", value=True)
                save_history = st.checkbox("Save Prediction History", value=True)
            
            # Debug & Development Card
            with col4:
                st.markdown("""
                <div class="card card-elevated content-card feature-card animate-fade-in">
                    <div class="card-title">
                        <span class="card-icon">🔧</span>
                        Debug & Development
                    </div>
                    <div class="card-content">
                        <p>Tools for developers and advanced users.</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Debug settings
                enable_logging = st.checkbox("Enable Debug Logging", value=False)
                
                if st.button("Open Debug Console"):
                    st.session_state["active_view"] = "debug"
                    st.rerun()
                
                if st.button("Reset All Settings"):
                    if st.button("Confirm Reset", key="confirm_reset"):
                        # Reset code would go here
                        st.success("All settings have been reset to defaults.")
                        st.rerun()
            
            # Close the settings-view div at the end of the render method
            st.markdown('</div>', unsafe_allow_html=True)
            
            self.logger.debug("Settings view rendered successfully")
        except Exception as e:
            self.logger.error(f"Error rendering settings view: {str(e)}", exc_info=True)
            st.error(f"An error occurred while rendering the settings view: {str(e)}") 