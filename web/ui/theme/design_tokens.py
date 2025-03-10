from typing import Dict, Any, Optional
import streamlit as st

class DesignTokens:
    """Access design tokens from the current theme."""
    
    @staticmethod
    def color(name: str) -> str:
        """Get a color from the current theme."""
        if not hasattr(st.session_state, "theme_colors"):
            return ""
        return st.session_state.theme_colors.get(name, "")
    
    @staticmethod
    def spacing(multiplier: int = 1) -> str:
        """Get spacing value based on the base spacing unit."""
        base = 8  # Base spacing in pixels
        return f"{base * multiplier}px"
    
    @staticmethod
    def border_radius() -> str:
        """Get the current theme's border radius."""
        if not hasattr(st.session_state, "theme"):
            return "4px"
        return st.session_state.theme.get("settings", {}).get("border_radius", "4px")
    
    @staticmethod
    def font(type_name: str) -> str:
        """Get font family for the specified type."""
        if not hasattr(st.session_state, "theme"):
            return "system-ui, sans-serif"
        return st.session_state.theme.get("fonts", {}).get(type_name, "system-ui, sans-serif")
    
    @staticmethod
    def get_card_style() -> Dict[str, str]:
        """Get styling for a card component based on current theme."""
        return {
            "backgroundColor": DesignTokens.color("card"),
            "borderRadius": DesignTokens.border_radius(),
            "padding": DesignTokens.spacing(2),
            "border": f"1px solid {DesignTokens.color('border')}",
            "marginBottom": DesignTokens.spacing(2)
        }
    
    @staticmethod
    def get_button_style(variant: str = "primary") -> Dict[str, str]:
        """Get styling for a button based on current theme and variant."""
        base_style = {
            "borderRadius": DesignTokens.border_radius(),
            "padding": f"{DesignTokens.spacing(1)} {DesignTokens.spacing(2)}",
            "fontFamily": DesignTokens.font("body"),
            "cursor": "pointer",
            "transition": "all 0.2s ease-in-out"
        }
        
        variant_styles = {
            "primary": {
                "backgroundColor": DesignTokens.color("primary"),
                "color": "white",
                "border": "none"
            },
            "secondary": {
                "backgroundColor": DesignTokens.color("secondary"),
                "color": "white",
                "border": "none"
            },
            "outline": {
                "backgroundColor": "transparent",
                "color": DesignTokens.color("primary"),
                "border": f"1px solid {DesignTokens.color('primary')}"
            }
        }
        
        # Merge base style with variant style
        return {**base_style, **(variant_styles.get(variant, variant_styles["primary"]))} 