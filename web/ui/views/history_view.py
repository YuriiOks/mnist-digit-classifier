# MNIST Digit Classifier
# Copyright (c) 2025 YuriODev (YuriiOks)
# File: web/ui/views/history_view.py
# Description: History view implementation with database integration
# Created: 2025-03-17
# Updated: 2025-03-30

import streamlit as st
import datetime
import pandas as pd
import logging
from typing import Optional, Tuple, List, Dict, Any

from ui.views.base_view import View
from core.app_state.history_state import HistoryState
from core.app_state.navigation_state import NavigationState
from utils.resource_manager import resource_manager
from ui.components.cards.card import WelcomeCard, FeatureCard


class HistoryView(View):
    """History view for the MNIST Digit Classifier application."""

    def __init__(self):
        """Initialize the history view."""
        super().__init__(
            name="history",
            title="Prediction History",
            description="View and manage your past digit predictions.",
        )
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def _initialize_session_state(self) -> None:
        """Initialize session state variables for history filtering and pagination."""
        # Use a dictionary to simplify initialization of session state variables
        default_states = {
            # Filter controls
            "history_filter_date": None,
            "history_filter_digit": None,
            "history_filter_min_confidence": 0.0,
            "history_sort_by": "newest",
            # Pagination controls
            "history_page": 1,
            "history_items_per_page": 12,
            # Entry deletion state
            "delete_id": None,
            "show_delete_confirm": False,
            "clear_all_confirm": False,
        }

        # Initialize all missing session state variables
        for key, default_value in default_states.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def _load_view_data(self) -> Dict:
        """Load necessary JSON data for the History view."""
        data = resource_manager.load_json_resource("history/history_view.json")
        return data or {}  # Return empty dict as fallback

    def _render_empty_state(self) -> None:
        """Render the empty state when no history is available."""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            empty_state_content = """<div style="text-align: center; padding: 2rem 1rem;">
                <div style="font-size: 3.5rem; margin-bottom: 1rem;">📊</div>
                <h3 style="margin-bottom: 0.75rem; color: #334155; font-weight: 600;">No Prediction History Yet</h3>
                <p style="color: #64748b; margin-bottom: 1.5rem;">Your prediction history will appear here once you make some predictions.</p>
            </div>
            """

            self._render_card(
                "📈", "History", empty_state_content, "small animate-fade-in"
            )

            # Add a button to navigate to draw view
            if st.button(
                "Go to Draw View",
                key="go_to_draw",
                type="primary",
                use_container_width=True,
            ):
                NavigationState.set_active_view("draw")
                st.rerun()

    def _render_card(
        self, icon: str, title: str, content: str, extra_classes: str = ""
    ) -> None:
        """Helper method to render a card with consistent styling."""
        st.markdown(
            f"""
        <div class="card card-elevated content-card feature-card {extra_classes}">
            <div class="card-title">
                <span class="card-icon">{icon}</span> {title}
            </div>
            <div class="card-content">
                {content}""",
            unsafe_allow_html=True,
        )

    def _render_filters(self) -> None:
        """Render filter controls for the history view."""
        st.markdown("<div style='margin-bottom: 1rem;'>", unsafe_allow_html=True)

        filter_col1, filter_col2, filter_col3 = st.columns(3)

        # Digit filter
        with filter_col1:
            digit_options = [None] + list(range(10))
            format_digit = lambda x: "All Digits" if x is None else str(x)
            digit_filter = st.selectbox(
                "Filter by Digit",
                options=digit_options,
                format_func=format_digit,
                key="digit_filter",
                index=(
                    0
                    if st.session_state.history_filter_digit is None
                    else digit_options.index(st.session_state.history_filter_digit)
                ),
            )
            st.session_state.history_filter_digit = digit_filter

        # Confidence filter
        with filter_col2:
            confidence_filter = st.slider(
                "Minimum Confidence",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.history_filter_min_confidence,
                step=0.1,
                format="%.1f",
                key="confidence_filter",
            )
            st.session_state.history_filter_min_confidence = confidence_filter

        # Sort options
        with filter_col3:
            sort_options = {
                "newest": "Newest First",
                "oldest": "Oldest First",
                "highest_conf": "Highest Confidence",
                "lowest_conf": "Lowest Confidence",
            }
            sort_by = st.selectbox(
                "Sort By",
                options=list(sort_options.keys()),
                format_func=lambda x: sort_options[x],
                key="sort_by",
                index=list(sort_options.keys()).index(st.session_state.history_sort_by),
            )
            st.session_state.history_sort_by = sort_by

        st.markdown("</div>", unsafe_allow_html=True)

        # Update pagination if filters changed
        if self._filters_changed():
            st.session_state.history_page = 1

        # Update previous filter values for change detection
        st.session_state.prev_digit_filter = st.session_state.history_filter_digit
        st.session_state.prev_confidence_filter = (
            st.session_state.history_filter_min_confidence
        )
        st.session_state.prev_sort_by = st.session_state.history_sort_by

    def _filters_changed(self) -> bool:
        """Check if filters have changed since the last render."""
        return (
            st.session_state.get("prev_digit_filter")
            != st.session_state.history_filter_digit
            or st.session_state.get("prev_confidence_filter")
            != st.session_state.history_filter_min_confidence
            or st.session_state.get("prev_sort_by") != st.session_state.history_sort_by
        )

    def _get_confidence_color(self, confidence: float) -> str:
        """Get color based on confidence value."""
        if confidence >= 0.8:
            return "#10b981"  # Green for high confidence
        elif confidence >= 0.5:
            return "#f59e0b"  # Yellow/orange for medium confidence
        else:
            return "#ef4444"  # Red for low confidence

    def _render_pagination(
        self, total_items: int, total_pages: int, position: str = "top"
    ) -> None:
        """Render pagination controls with page navigation."""
        # Calculate start and end indices
        items_per_page = st.session_state.history_items_per_page
        start_idx = (st.session_state.history_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)

        # Create pagination content
        pagination_content = f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <div style="font-size: 0.875rem; color: #64748b;">
                Showing <span style="font-weight: 500; color: #334155;">
                {start_idx + 1 if total_items > 0 else 0}-{end_idx} of {total_items}</span> predictions
            </div>
            <div style="font-size: 0.875rem; color: #64748b;">
                Page <span style="font-weight: 500; color: #334155;">
                {st.session_state.history_page} of {max(1, total_pages)}</span>
            </div>
        </div>
        """

        # Render the pagination card - closing the divs properly
        st.markdown(
            f"""
        <div class="card card-elevated content-card feature-card small animate-fade-in">
            <div class="card-title">
                <span class="card-icon">📄</span> Prediction Pages
            </div>
            <div class="card-content">
                {pagination_content}""",
            unsafe_allow_html=True,
        )

        # Pagination controls
        if total_pages > 1:
            self._render_pagination_controls(total_pages, position)

    def _render_pagination_controls(self, total_pages: int, position: str) -> None:
        """Render the pagination control buttons and slider."""
        col1, col2, col3 = st.columns([1, 3, 1])

        # Create unique keys for each button based on position
        prev_key = f"prev_page_{position}"
        next_key = f"next_page_{position}"

        with col1:
            if st.button(
                "← Previous",
                key=prev_key,
                disabled=st.session_state.history_page <= 1,
                use_container_width=True,
            ):
                st.session_state.history_page -= 1
                st.rerun()

        with col2:
            # Page selector
            page_numbers = list(range(1, total_pages + 1))
            selected_page = st.select_slider(
                "Page selector",
                options=page_numbers,
                value=st.session_state.history_page,
                key=f"page_selector_{position}",
                label_visibility="collapsed",
            )

            if selected_page != st.session_state.history_page:
                st.session_state.history_page = selected_page
                st.rerun()

        with col3:
            if st.button(
                "Next →",
                key=next_key,
                disabled=st.session_state.history_page >= total_pages,
                use_container_width=True,
            ):
                st.session_state.history_page += 1
                st.rerun()

    def _render_clear_all_button(self) -> None:
        """Render the clear history button with confirmation."""
        st.write("")

        if not st.session_state.clear_all_confirm:
            # Show the clear all button
            action_content = """<div style="text-align: center; padding: 1rem 0;">
                <p style="color: #64748b; margin-bottom: 1rem;">
                    Want to start fresh? You can clear your entire prediction history.
                </p>
            </div>
            """

            self._render_card(
                "🗑️", "Manage History", action_content, "small animate-fade-in"
            )

            # Center the button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(
                    "Clear All History",
                    key="clear_history",
                    type="secondary",
                    use_container_width=True,
                ):
                    st.session_state.clear_all_confirm = True
                    st.rerun()
        else:
            # Show confirmation dialog
            self._render_clear_all_confirmation()

    def _render_clear_all_confirmation(self) -> None:
        """Render the confirmation dialog for clearing all history."""
        confirm_content = """
        <div style="text-align: center; padding: 1rem 0;">
            <p style="color: #ef4444; font-weight: 500; margin-bottom: 1rem;">
                ⚠️ This will permanently delete all your prediction history.
            </p>
            <p style="color: #64748b; margin-bottom: 1rem;">
                This action cannot be undone. Are you sure you want to continue?
            </p>
        </div>
        """

        self._render_card(
            "⚠️", "Confirm Deletion", confirm_content, "small animate-fade-in"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cancel", key="cancel_clear", use_container_width=True):
                st.session_state.clear_all_confirm = False
                st.rerun()
        with col2:
            if st.button(
                "Yes, Delete All",
                key="confirm_clear",
                type="primary",
                use_container_width=True,
            ):
                # Clear history and reset filters
                HistoryState.clear_history()
                self._reset_filters()
                st.success("✅ Prediction history cleared successfully.")
                st.rerun()

    def _reset_filters(self) -> None:
        """Reset all filters and pagination to default values."""
        st.session_state.history_filter_date = None
        st.session_state.history_filter_digit = None
        st.session_state.history_filter_min_confidence = 0.0
        st.session_state.history_sort_by = "newest"
        st.session_state.history_page = 1
        st.session_state.clear_all_confirm = False

    def _handle_delete(self) -> None:
        """Handle deletion of individual history entries."""
        # Check if we have a pending deletion
        if st.session_state.delete_id and st.session_state.show_delete_confirm:
            entry_id = st.session_state.delete_id

            # Show confirmation dialog
            confirm_content = """
            <div style="text-align: center; padding: 1rem 0;">
                <p style="color: #64748b; margin-bottom: 1rem;">
                    Are you sure you want to delete this prediction?
                </p>
                <p style="color: #ef4444; font-size: 0.875rem; margin-bottom: 1rem;">
                    This action cannot be undone.
                </p>
            </div>
            """

            self._render_card(
                "🗑️",
                "Confirm Deletion",
                confirm_content,
                "small animate-fade-in",
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("Cancel", key="cancel_delete", use_container_width=True):
                    st.session_state.delete_id = None
                    st.session_state.show_delete_confirm = False
                    st.rerun()
            with col2:
                if st.button(
                    "Delete",
                    key="confirm_delete",
                    type="primary",
                    use_container_width=True,
                ):
                    self._logger.info(f"Deleting history entry: {entry_id}")

                    # Delete the entry
                    success = HistoryState.delete_entry(entry_id)

                    if success:
                        st.success("✅ Entry deleted successfully.")
                    else:
                        st.error("❌ Failed to delete entry.")

                    # Reset delete state
                    st.session_state.delete_id = None
                    st.session_state.show_delete_confirm = False
                    st.rerun()

    def _get_card_title_and_icon(self, item: Dict[str, Any]) -> Tuple[str, str]:
        """Generate smart card title and icon based on prediction status."""
        # Extract necessary data
        digit_value = self._extract_digit_value(item)
        corrected_digit = item.get("user_correction")
        confidence = item.get("confidence", 0.0)

        # Get timestamp for date inclusion
        timestamp = self._parse_timestamp(item.get("timestamp"))
        date_str = (
            timestamp.strftime("%b %d")
            if isinstance(timestamp, datetime.datetime)
            else ""
        )

        # Create smart title based on prediction status
        if corrected_digit is not None:
            # This was a corrected prediction
            title = f"Corrected {digit_value}→{corrected_digit}"
            icon = "❌"  # Recycling symbol for correction
        elif confidence >= 0.9:
            # High confidence prediction
            title = f"Strong Prediction: {digit_value}"
            icon = "🎯"  # Target/bullseye for high confidence
        elif confidence >= 0.7:
            # Good confidence prediction
            title = f"Good Prediction: {digit_value}"
            icon = "✅"  # Checkmark for good confidence
        elif confidence >= 0.5:
            # Moderate confidence prediction
            title = f"Fair Prediction: {digit_value}"
            icon = "⚠️"  # Warning for moderate confidence
        else:
            # Low confidence prediction
            title = f"Uncertain: {digit_value}"
            icon = "❓"  # Question mark for low confidence

        # Add date to the title
        title = f"{date_str} • {title}"

        return (title, icon)

    def _extract_digit_value(self, item: Dict[str, Any]) -> str:
        """Extract the digit value from a prediction item."""
        digit_value = item.get("digit", "?")
        if isinstance(digit_value, str) and "<" in digit_value:
            import re

            digit_match = re.search(r">(\d)<", digit_value)
            if digit_match:
                digit_value = digit_match.group(1)
        return digit_value

    def _parse_timestamp(self, timestamp) -> datetime.datetime:
        """Parse timestamp string to datetime object."""
        if isinstance(timestamp, str):
            try:
                return datetime.datetime.fromisoformat(timestamp)
            except (ValueError, TypeError):
                return datetime.datetime.now()
        return (
            timestamp
            if isinstance(timestamp, datetime.datetime)
            else datetime.datetime.now()
        )

    def _render_history_entries(self, page_items: List[Dict[str, Any]]) -> None:
        """Render history entries in a grid layout."""
        if not page_items:
            st.info("No entries match your filter criteria. Try adjusting the filters.")
            return

        # Create grid layout with 3 columns
        num_items = len(page_items)
        rows = (num_items + 2) // 3  # Ceiling division

        # Add some spacing
        st.write("")

        for row in range(rows):
            # Create columns for this row
            cols = st.columns(3)

            for col in range(3):
                idx = row * 3 + col
                if idx < num_items:
                    with cols[col]:
                        self._render_history_card(page_items[idx])

    def _render_history_card(self, item: Dict[str, Any]) -> None:
        """Render a single history card with prediction visualization inside."""
        # Format timestamp
        timestamp = self._parse_timestamp(item.get("timestamp"))
        timestamp_str = (
            timestamp.strftime("%H:%M")
            if isinstance(timestamp, datetime.datetime)
            else ""
        )

        # Format confidence
        confidence = item.get("confidence", 0.0)
        confidence_pct = f"{confidence * 100:.1f}%"
        confidence_color = self._get_confidence_color(confidence)

        # Extract digit and correction status
        digit_value = self._extract_digit_value(item)
        corrected_digit = item.get("user_correction")

        # Determine input type
        input_type = item.get("input_type", "canvas")
        input_icon = (
            "✏️" if input_type == "canvas" else "📷" if input_type == "upload" else "🔗"
        )

        # Get dynamic card title and icon
        card_title, card_icon = self._get_card_title_and_icon(item)

        # Use Streamlit container to maintain proper card structure
        with st.container():
            # Define all HTML parts first
            card_header_html = f"""<div class="card card-elevated content-card feature-card small animate-fade-in">
                    <div class="card-title">
                        <span class="card-icon">{card_icon}</span> {card_title}
                    </div>
                    <div class="card-content">
                """

            input_type_html = f"""<div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem; padding: 0 0.5rem;">
                        <span style="font-weight: 500;"><b>Input method:</b> {input_icon} {input_type.capitalize()}</span>
                        <span style="color: {confidence_color}; font-weight: 500;">{confidence_pct}</span>
                    </div>
                """

            # Different HTML for corrected vs. regular digits
            if corrected_digit is not None:
                digit_content_html = f"""<div style="display: flex; justify-content: center; align-items: center; 
                                margin: 1rem 0; padding: 0.5rem; background-color: #f8fafc; border-radius: 0.5rem;">
                        <div style="text-align: center; font-size: 2.5rem; color: #94a3b8; margin-right: 0.75rem;">
                            {digit_value}
                        </div>
                        <div style="margin: 0 0.75rem; color: #64748b; font-size: 1.5rem;">
                            →
                        </div>
                        <div style="text-align: center; font-size: 2.5rem; color: #0ea5e9; font-weight: 500; margin-left: 0.75rem;">
                            {corrected_digit}
                        </div>
                    </div>
                    """
            else:
                digit_content_html = f"""<div style="text-align: center; margin: 1rem 0; padding: 1rem 0.5rem;
                                background-color: #f8fafc; border-radius: 0.5rem;">
                        <div style="font-size: 3.5rem; font-weight: 500; color: #334155;">
                            {digit_value}
                        </div>
                    </div>
                    """

            timestamp_html = f"""<div style="text-align: center; color: #111212; font-size: 0.875rem; padding: 0 0.5rem 0.5rem 0.5rem;">
                        {timestamp_str}
                    </div>
                """

            # Combine and render all HTML parts in a single markdown call
            full_card_html = f"{card_header_html}{input_type_html}{digit_content_html}{timestamp_html}"
            st.markdown(full_card_html, unsafe_allow_html=True)

            # Delete button outside the card but within the container
            entry_id = item.get("id")
            if st.button(
                "Delete",
                key=f"delete_{entry_id}",
                type="secondary",
                use_container_width=True,
            ):
                st.session_state.delete_id = entry_id
                st.session_state.show_delete_confirm = True
                st.rerun()

    def render(self) -> None:
        """Render the history view content."""
        # Initialize session state variables
        self._initialize_session_state()

        # Load view configuration data
        data = self._load_view_data()

        # Render welcome card if it exists in the data
        welcome = data.get("welcome_card", {})
        if welcome:
            welcome_card = WelcomeCard(
                title=welcome.get("title", "Your Prediction History"),
                content=welcome.get(
                    "content",
                    "Browse, filter, and manage your past predictions.",
                ),
                icon=welcome.get("icon", "📊"),
            )
            welcome_card.display()

        # Create two columns for filters and display options
        filter_col, options_col = st.columns([3, 1])

        with filter_col:
            # Render filters
            self._render_filters()

        # Handle any pending deletions - place this after filters but before content
        self._handle_delete()

        # Get total count with filters applied
        total_items = HistoryState.get_history_size(
            digit_filter=st.session_state.history_filter_digit,
            min_confidence=st.session_state.history_filter_min_confidence,
        )

        # If no items, show empty state
        if total_items == 0:
            self._render_empty_state()
            return

        # Calculate pagination
        items_per_page = st.session_state.history_items_per_page
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

        # Ensure current page is valid
        st.session_state.history_page = min(
            max(1, st.session_state.history_page), total_pages
        )

        # Get items for current page with filters
        page_items = HistoryState.get_paginated_history(
            page=st.session_state.history_page,
            page_size=items_per_page,
            digit_filter=st.session_state.history_filter_digit,
            min_confidence=st.session_state.history_filter_min_confidence,
            sort_by=st.session_state.history_sort_by,
        )

        # Render pagination controls at the top
        self._render_pagination(total_items, total_pages, position="top")

        # Render history entries
        self._render_history_entries(page_items)

        # Render pagination controls at the bottom
        self._render_pagination(total_items, total_pages, position="bottom")

        # Render clear all button
        self._render_clear_all_button()

        # Render tips card if available
        tips_data = data.get("tips", {})
        if tips_data:
            st.write("")
            FeatureCard(
                title=tips_data.get("title", "Tips"),
                content="<ul class='tips-list'>"
                + "".join(f"<li>{tip}</li>" for tip in tips_data.get("items", []))
                + "</ul>",
                icon="💡",
            ).display()
