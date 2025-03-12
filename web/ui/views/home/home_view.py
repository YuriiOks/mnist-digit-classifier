import streamlit as st
import logging
import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from ui.views.base_view import BaseView
from ui.components.cards.card import Card
from core.app_state.navigation_state import NavigationState
from utils.template_loader import TemplateLoader
from utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


class HomeView(BaseView):
    def __init__(self):
        super().__init__(view_id="home",
                         title="Welcome to MNIST Digit Classifier", icon="🏠")
        self.template_loader = TemplateLoader()
        self.data_loader = DataLoader()

    def load_feature_cards(self) -> List[Dict[str, Any]]:
        return self.data_loader.load_json("/home/feature_cards.json") or []

    def render(self) -> None:
        try:
            self.apply_common_layout()

            # Load home-specific CSS
            self._load_home_css()

            welcome_data = self.data_loader.load_json("/home/welcome_card.json")
            if not welcome_data:
                # Try fallback locations
                welcome_data = self.data_loader.load_json("/home/welcome_card.json")

            welcome_card = Card(
                title=welcome_data.get("title", ""),
                content=welcome_data.get("content", ""),
                icon=welcome_data.get("icon", ""),  
                elevated=True,
                size="large",
                classes=["welcome_card"]

            )
            welcome_card.display(self.template_loader.render_template)

            section_headers = self.template_loader.load_template("home/section_headers.html")
            how_it_works_header, _, cta_header = section_headers.partition("<!-- Call to Action Section Header -->")

            st.markdown(how_it_works_header, unsafe_allow_html=True)

            feature_cards = self.load_feature_cards()
            cols = st.columns(3)

            for col, card_data in zip(cols, feature_cards):
                with col:

                    card = Card(
                        title=card_data["title"],
                        content=card_data["content"],
                        icon=card_data["icon"],
                        elevated=True,
                        size=card_data["size"],
                        classes=["feature_card"]
                    )
                    card.display(self.template_loader.render_template)

            st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

            # Add a div with a specific class to target with CSS
            st.markdown('<div class="cta-section">', unsafe_allow_html=True)

            cta_col1, cta_col2 = st.columns([2, 3])

            with cta_col1:
                st.markdown(cta_header, unsafe_allow_html=True)

            with cta_col2:
                if st.button("Start Drawing", type="primary"):
                    NavigationState.set_active_view("draw")
                    st.rerun()

            # Close the div
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            logger.error(f"Error rendering home view: {e}", exc_info=True)
            st.error(f"An error occurred: {e}")

    def _setup(self) -> None:
        pass

    def get_view_data(self) -> Dict[str, Any]:
        data = super().get_view_data()
        data.update({"sections": ["welcome", "features", "how-it-works", "cta"]})
        return data

    def _load_home_css(self):
        """Load CSS specific to home view."""
        try:
            css_path = Path(__file__).parent.parent.parent.parent / "assets" / "css" / "views" / "home.css"
            if css_path.exists():
                with open(css_path, "r") as f:
                    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                    self.logger.debug("Home view CSS loaded successfully")
            else:
                self.logger.warning(f"Home CSS file not found at {css_path}")
        except Exception as e:
            self.logger.error(f"Error loading home CSS: {str(e)}")
