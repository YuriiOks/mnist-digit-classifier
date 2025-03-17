# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/settings_view.py
# Description: Settings view implementation
# Created: 2025-03-17

import streamlit as st
import logging
from typing import Optional

from ui.views.base_view import View
from core.app_state.settings_state import SettingsState
from ui.theme.theme_manager import theme_manager

class SettingsView(View):
    """Settings view for the MNIST Digit Classifier application."""
    
    def __init__(self):
        """Initialize the settings view."""
        super().__init__(
            name="settings",
            title="Settings",
            description="Configure your preferences for the MNIST Digit Classifier."
        )
        
    def _initialize_session_state(self) -> None:
        """Initialize session state variables for the settings view."""
        if "active_settings_tab" not in st.session_state:
            st.session_state.active_settings_tab = "theme"
    
    def _render_tab_buttons(self) -> None:
        """Render tab selection buttons."""
        tab_cols = st.columns(4)
        
        with tab_cols[0]:
            if st.button("Theme Settings", 
                      key="tab_theme", 
                      type="primary" if st.session_state.active_settings_tab == "theme" else "secondary",
                      use_container_width=True):
                st.session_state.active_settings_tab = "theme"
                st.rerun()
        
        with tab_cols[1]:
            if st.button("Canvas Settings", 
                      key="tab_canvas", 
                      type="primary" if st.session_state.active_settings_tab == "canvas" else "secondary",
                      use_container_width=True):
                st.session_state.active_settings_tab = "canvas"
                st.rerun()
        
        with tab_cols[2]:
            if st.button("Prediction Settings", 
                      key="tab_prediction", 
                      type="primary" if st.session_state.active_settings_tab == "prediction" else "secondary",
                      use_container_width=True):
                st.session_state.active_settings_tab = "prediction"
                st.rerun()
        
        with tab_cols[3]:
            if st.button("App Settings", 
                      key="tab_app", 
                      type="primary" if st.session_state.active_settings_tab == "app" else "secondary",
                      use_container_width=True):
                st.session_state.active_settings_tab = "app"
                st.rerun()
    
    def _render_theme_settings(self) -> None:
        """Render theme settings tab content."""
        st.markdown("<h3>Theme Settings</h3>", unsafe_allow_html=True)
        
        # Theme Mode
        st.markdown("<h4>Theme Mode</h4>", unsafe_allow_html=True)
        
        # Get current theme
        current_theme = theme_manager.get_current_theme()
        
        # Create theme selector with visual preview
        col1, col2 = st.columns(2)
        
        with col1:
            light_selected = current_theme == "light"
            if st.button(
                "Light Theme", 
                key="light_theme_btn",
                type="primary" if light_selected else "secondary",
                use_container_width=True
            ):
                # Apply the theme and save to settings
                theme_manager.apply_theme("light")
                SettingsState.set_setting("theme", "mode", "light")
                st.rerun()
                
            st.markdown("""
            <div style="
                border-radius: 8px;
                border: 1px solid #dee2e6;
                padding: 1rem;
                margin-top: 0.5rem;
                background-color: #f8f9fa;
                color: #212529;
                text-align: center;
            ">
                <div style="display: flex; justify-content: center; gap: 0.5rem;">
                    <div style="width: 24px; height: 24px; border-radius: 4px; background-color: #4361ee;"></div>
                    <div style="width: 24px; height: 24px; border-radius: 4px; background-color: #4cc9f0;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            dark_selected = current_theme == "dark"
            if st.button(
                "Dark Theme", 
                key="dark_theme_btn",
                type="primary" if dark_selected else "secondary",
                use_container_width=True
            ):
                # Apply the theme and save to settings
                theme_manager.apply_theme("dark")
                SettingsState.set_setting("theme", "mode", "dark")
                st.rerun()
                
            st.markdown("""
            <div style="
                border-radius: 8px;
                border: 1px solid #383838;
                padding: 1rem;
                margin-top: 0.5rem;
                background-color: #121212;
                color: #f8f9fa;
                text-align: center;
            ">
                <div style="font-weight: bold; margin-bottom: 0.5rem;">Dark Mode</div>
                <div style="display: flex; justify-content: center; gap: 0.5rem;">
                    <div style="width: 24px; height: 24px; border-radius: 4px; background-color: #ee4347;"></div>
                    <div style="width: 24px; height: 24px; border-radius: 4px; background-color: #f0c84c;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Font settings
        st.markdown("<h4>Typography</h4>", unsafe_allow_html=True)
        
        font_options = ["System Default", "Sans-serif", "Serif", "Monospace"]
        current_font = SettingsState.get_setting("theme", "font_family", "System Default")
        
        selected_font = st.selectbox(
            "Font Family", 
            options=font_options,
            index=font_options.index(current_font) if current_font in font_options else 0,
            key="font_family"
        )
        
        if selected_font != current_font:
            SettingsState.set_setting("theme", "font_family", selected_font)
            
            # Apply font change via CSS
            font_css = ""
            if selected_font == "Sans-serif":
                font_css = "body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif !important; }"
            elif selected_font == "Serif":
                font_css = "body { font-family: Georgia, 'Times New Roman', Times, serif !important; }"
            elif selected_font == "Monospace":
                font_css = "body { font-family: SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace !important; }"
            elif selected_font == "System Default":
                font_css = "body { font-family: var(--font-primary) !important; }"
            
            if font_css:
                st.markdown(f"<style>{font_css}</style>", unsafe_allow_html=True)
    
    def _render_canvas_settings(self) -> None:
        """Render canvas settings tab content."""
        st.markdown("<h3>Canvas Settings</h3>", unsafe_allow_html=True)
        
        # Canvas size
        st.markdown("<h4>Drawing Canvas</h4>", unsafe_allow_html=True)
        
        canvas_size = SettingsState.get_setting("canvas", "canvas_size", 280)
        new_canvas_size = st.slider(
            "Canvas Size",
            min_value=200,
            max_value=400,
            value=canvas_size,
            step=20,
            key="canvas_size_slider"
        )
        
        if new_canvas_size != canvas_size:
            SettingsState.set_setting("canvas", "canvas_size", new_canvas_size)
            # Reset canvas key to ensure it reloads with new size
            if "canvas_key" in st.session_state:
                import time
                st.session_state.canvas_key = f"canvas_{hash(time.time())}"
        
        # Stroke width
        stroke_width = SettingsState.get_setting("canvas", "stroke_width", 15)
        new_stroke_width = st.slider(
            "Default Stroke Width",
            min_value=5,
            max_value=30,
            value=stroke_width,
            step=1,
            key="stroke_width_slider"
        )
        
        if new_stroke_width != stroke_width:
            SettingsState.set_setting("canvas", "stroke_width", new_stroke_width)
        
        # Colors
        st.markdown("<h4>Colors</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            stroke_color = SettingsState.get_setting("canvas", "stroke_color", "#000000")
            new_stroke_color = st.color_picker(
                "Stroke Color",
                value=stroke_color,
                key="stroke_color_picker"
            )
            
            if new_stroke_color != stroke_color:
                SettingsState.set_setting("canvas", "stroke_color", new_stroke_color)
        
        with col2:
            bg_color = SettingsState.get_setting("canvas", "background_color", "#FFFFFF")
            new_bg_color = st.color_picker(
                "Background Color",
                value=bg_color,
                key="bg_color_picker"
            )
            
            if new_bg_color != bg_color:
                SettingsState.set_setting("canvas", "background_color", new_bg_color)
        
        # Grid settings
        enable_grid = SettingsState.get_setting("canvas", "enable_grid", False)
        new_enable_grid = st.toggle("Show Grid on Canvas", value=enable_grid, key="grid_toggle")
        
        if new_enable_grid != enable_grid:
            SettingsState.set_setting("canvas", "enable_grid", new_enable_grid)
            # Apply grid CSS
            if new_enable_grid:
                grid_css = """
                .canvas-container canvas {
                    background-image: linear-gradient(#ddd 1px, transparent 1px), 
                                      linear-gradient(90deg, #ddd 1px, transparent 1px);
                    background-size: 20px 20px;
                }
                """
                st.markdown(f"<style>{grid_css}</style>", unsafe_allow_html=True)
    
    def _render_prediction_settings(self) -> None:
        """Render prediction settings tab content."""
        st.markdown("<h3>Prediction Settings</h3>", unsafe_allow_html=True)
        
        # Auto-predict
        st.markdown("<h4>Prediction Behavior</h4>", unsafe_allow_html=True)
        
        auto_predict = SettingsState.get_setting("prediction", "auto_predict", False)
        new_auto_predict = st.toggle("Auto-predict after drawing", value=auto_predict, key="auto_predict_toggle")
        
        if new_auto_predict != auto_predict:
            SettingsState.set_setting("prediction", "auto_predict", new_auto_predict)
        
        # Confidence settings
        st.markdown("<h4>Confidence Display</h4>", unsafe_allow_html=True)
        
        show_confidence = SettingsState.get_setting("prediction", "show_confidence", True)
        new_show_confidence = st.toggle("Show confidence percentage", value=show_confidence, key="confidence_toggle")
        
        if new_show_confidence != show_confidence:
            SettingsState.set_setting("prediction", "show_confidence", new_show_confidence)
        
        min_confidence = SettingsState.get_setting("prediction", "min_confidence", 0.5)
        new_min_confidence = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=min_confidence,
            step=0.05,
            format="%.2f",
            key="min_confidence_slider"
        )
        
        if new_min_confidence != min_confidence:
            SettingsState.set_setting("prediction", "min_confidence", new_min_confidence)
        
        # Show alternatives
        show_alternatives = SettingsState.get_setting("prediction", "show_alternatives", True)
        new_show_alternatives = st.toggle("Show alternative predictions", value=show_alternatives, key="alternatives_toggle")
        
        if new_show_alternatives != show_alternatives:
            SettingsState.set_setting("prediction", "show_alternatives", new_show_alternatives)
    
    def _render_app_settings(self) -> None:
        """Render application settings tab content."""
        st.markdown("<h3>Application Settings</h3>", unsafe_allow_html=True)
        
        # History settings
        st.markdown("<h4>History</h4>", unsafe_allow_html=True)
        
        save_history = SettingsState.get_setting("app", "save_history", True)
        new_save_history = st.toggle("Save prediction history", value=save_history, key="save_history_toggle")
        
        if new_save_history != save_history:
            SettingsState.set_setting("app", "save_history", new_save_history)
        
        max_history = SettingsState.get_setting("app", "max_history", 50)
        new_max_history = st.slider(
            "Maximum History Items",
            min_value=10,
            max_value=100,
            value=max_history,
            step=10,
            key="max_history_slider"
        )
        
        if new_max_history != max_history:
            SettingsState.set_setting("app", "max_history", new_max_history)
        
        # UI settings
        st.markdown("<h4>User Interface</h4>", unsafe_allow_html=True)
        
        show_tooltips = SettingsState.get_setting("app", "show_tooltips", True)
        new_show_tooltips = st.toggle("Show tooltips", value=show_tooltips, key="tooltips_toggle")
        
        if new_show_tooltips != show_tooltips:
            SettingsState.set_setting("app", "show_tooltips", new_show_tooltips)
            # Apply tooltip CSS
            if not new_show_tooltips:
                st.markdown("<style>[data-tooltip]{display:none !important;}</style>", unsafe_allow_html=True)
        
        debug_mode = SettingsState.get_setting("app", "debug_mode", False)
        new_debug_mode = st.toggle("Debug mode", value=debug_mode, key="debug_mode_toggle")
        
        if new_debug_mode != debug_mode:
            SettingsState.set_setting("app", "debug_mode", new_debug_mode)
    
    def _render_reset_button(self) -> None:
        """Render reset settings button."""
        st.markdown("<hr>", unsafe_allow_html=True)
        
        if st.button("Reset All Settings to Defaults", key="reset_settings", type="secondary"):
            # Show confirmation dialog
            reset_confirmed = st.checkbox("Confirm reset - this will revert all settings to their default values", key="reset_confirm")
            
            if reset_confirmed:
                # Reset all settings categories
                SettingsState.reset_to_defaults()
                
                # Reset theme to default
                theme_manager.apply_theme(theme_manager.DEFAULT_THEME)
                
                # Force a rerun to update the UI with default values
                st.session_state.active_settings_tab = "theme"  # Reset to first tab
                st.success("All settings have been reset to defaults.")
                st.rerun()
        
    def render(self) -> None:
        """Render the settings view content."""
        # Initialize session state variables
        self._initialize_session_state()
        
        # Render tab buttons
        self._render_tab_buttons()
        
        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Render tab content based on active tab
        if st.session_state.active_settings_tab == "theme":
            self._render_theme_settings()
        elif st.session_state.active_settings_tab == "canvas":
            self._render_canvas_settings()
        elif st.session_state.active_settings_tab == "prediction":
            self._render_prediction_settings()
        elif st.session_state.active_settings_tab == "app":
            self._render_app_settings()
        
        # Reset button
        self._render_reset_button()