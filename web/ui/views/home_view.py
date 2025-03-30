# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: ui/views/home_view.py
# Description: Home view implementation
# Created: 2025-03-17

import streamlit as st
import logging
from typing import Optional

from ui.views.base_view import View
from utils.resource_manager import resource_manager
from core.app_state.navigation_state import NavigationState
from ui.components.cards.card import WelcomeCard, FeatureCard


class HomeView(View):
    """Home view for the MNIST Digit Classifier application."""

    def __init__(self):
        """Initialize the home view."""
        super().__init__(
            name="home",
            title="MNIST Digit Classifier",
            description="Welcome to the MNIST Digit Classifier.",
        )
        self.show_header = False  # We'll use the welcome card instead

    def render(self) -> None:
        """Render the home view content."""
        # Load welcome card data from JSON
        welcome_data = resource_manager.load_json_resource("home/welcome_card.json")

        # Create and display welcome card
        welcome_card = WelcomeCard(
            title=welcome_data.get("title", "MNIST Digit Classifier"),
            content=welcome_data.get(
                "content", "Welcome to the MNIST Digit Classifier."
            ),
            icon=welcome_data.get("icon", "👋"),
        )
        welcome_card.display()

        st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

        # Display "How It Works" section
        st.markdown("<h2 id='how-it-works'>How It Works</h2>", unsafe_allow_html=True)

        # Load feature cards data from JSON
        feature_cards = resource_manager.load_json_resource("home/feature_cards.json")

        # Create columns for feature cards
        cols = st.columns(len(feature_cards))

        # Display feature cards
        for i, card_data in enumerate(feature_cards):
            with cols[i]:
                feature_card = FeatureCard(
                    title=card_data.get("title", f"Feature {i+1}"),
                    content=card_data.get("content", ""),
                    icon=card_data.get("icon", ""),
                )
                feature_card.display()

        st.markdown("<hr style='margin: 2rem 0;'>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        # CTA section
        # st.markdown('<div class="cta-section">', unsafe_allow_html=True)

        # cta_col1, cta_col2 = st.columns([2, 3])

        with col1:
            st.markdown(
                "<h2 id='ready-to-try-it-out' style='margin: 0;'>Ready to try it out?</h2>",
                unsafe_allow_html=True,
            )

        with col2:
            if st.button("Start Drawing", type="primary"):
                NavigationState.set_active_view("draw")
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)
