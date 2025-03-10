import streamlit as st
import logging
import os
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from ui.views.base_view import BaseView
from ui.components.cards.content_card import ContentCard
from core.app_state.navigation_state import NavigationState
from utils.template_loader import TemplateLoader
from utils.data_loader import DataLoader
from utils.ui import create_welcome_card


logger = logging.getLogger(__name__)


class HomeView(BaseView):
    def __init__(self):
        super().__init__(view_id="home", title="Welcome to MNIST Digit Classifier", icon="🏠")
        self.template_loader = TemplateLoader()
        self.data_loader = DataLoader()

    # def create_welcome_card(self, title: str, icon: str, content: str) -> str:
    #     formatted_content = ''.join(f'<p>{p.strip()}</p>' for p in content.split('\n') if p.strip())
    #     return self.template_loader.render_template("/home/welcome_card.html", {
    #         "title": title,
    #         "icon": icon,
    #         "content": formatted_content
    #     })

    def load_feature_cards(self) -> List[Dict[str, Any]]:
        return self.data_loader.load_json("/home/feature_cards.json") or []

    def _apply_css(self):
        for css_module in ["button_css", "card_css"]:
            try:
                from ....utils.css import button_css, card_css
                getattr(locals()[f"{css_module}"], f"load_{css_module}")()
            except (ImportError, ValueError):
                try:
                    from utils.css import button_css, card_css
                    getattr(locals()[f"{css_module}"], f"load_{css_module}")()
                except ImportError:
                    logger.warning(f"CSS module {css_module} not found")

    def render(self) -> None:
        try:
            self.apply_common_layout()
            self._apply_css()

            welcome_data = self.data_loader.load_json("/home/welcome_card.json")
            if not welcome_data:
                # Try fallback locations
                welcome_data = self.data_loader.load_json("/home/welcome_card.json")
                
            if welcome_data:
                welcome_card = create_welcome_card(
                    welcome_data.get("title", ""),
                    welcome_data.get("icon", ""),
                    welcome_data.get("content", ""),
                    self.template_loader.render_template
                )
                st.markdown(welcome_card, unsafe_allow_html=True)

            section_headers = self.template_loader.load_template("home/section_headers.html")
            how_it_works_header, _, cta_header = section_headers.partition("<!-- Call to Action Section Header -->")

            st.markdown(how_it_works_header, unsafe_allow_html=True)

            feature_cards = self.load_feature_cards()
            cols = st.columns(3)

            for col, card_data in zip(cols, feature_cards):
                with col:
                    card = ContentCard(
                        title=card_data["title"],
                        icon=card_data["icon"],
                        content=card_data["content"],
                        elevated=True,
                        size=card_data.get("size", "medium"),
                        classes=["feature-card"]
                    )
                    card.display()

            st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

            cta_col1, cta_col2 = st.columns([2, 3])

            with cta_col1:
                st.markdown(cta_header, unsafe_allow_html=True)

            with cta_col2:
                if st.button("Start Drawing", type="primary"):
                    NavigationState.set_active_view("draw")
                    st.rerun()

        except Exception as e:
            logger.error(f"Error rendering home view: {e}", exc_info=True)
            st.error(f"An error occurred: {e}")

    def _setup(self) -> None:
        pass

    def get_view_data(self) -> Dict[str, Any]:
        data = super().get_view_data()
        data.update({"sections": ["welcome", "features", "how-it-works", "cta"]})
        return data
