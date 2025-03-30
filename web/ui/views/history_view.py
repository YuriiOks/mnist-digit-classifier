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
        digit_value = item.get("prediction")

        return str(digit_value) if digit_value is not None else "?"

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
        """Render a single history card using st components and markdown."""
        db_id = item.get("id")
        if db_id is None:
            self._logger.warning("Skipping card: Missing ID.")
            return

        timestamp = self._parse_timestamp(item.get("timestamp"))
        # Must be in format of like March 30, 2025
        date_str = (
            timestamp.strftime("%b %d")
            if isinstance(timestamp, datetime.datetime)
            else ""
        )
        # Must be in format of like 12:34
        timestamp_str = (
            timestamp.strftime("%H:%M")
            if isinstance(timestamp, datetime.datetime)
            else ""
        )
        confidence = item.get("confidence", 0.0)
        confidence_pct = f"{confidence * 100:.1f}%"
        confidence_color = self._get_confidence_color(confidence)

        predicted_digit = item.get("prediction")  # Original prediction
        true_label = item.get("true_label")  # Corrected value (might be None)

        # Determine the primary digit to display and if it was corrected
        is_corrected = true_label is not None
        # The originally predicted digit, always shown
        original_display_digit = (
            str(predicted_digit) if predicted_digit is not None else "?"
        )
        # The final confirmed/corrected digit
        final_digit = str(true_label) if is_corrected else original_display_digit

        input_type = item.get("input_type", "unknown").capitalize()
        input_icon = (
            "✏️"
            if input_type == "Canvas"
            else (
                "📷"
                if input_type == "Upload"
                else "🔗" if input_type == "Url" else "❓"
            )
        )

        # --- Card Header Logic ---
        status_icon = ""
        header_detail = ""
        if is_corrected:
            status_icon = "❌"  # Icon for corrected entries
            # Show Original -> Corrected format only if they differ
            if str(true_label) != str(predicted_digit):
                header_detail = f"<span style='color: var(--color-text);'>(Corrected: {original_display_digit} → {final_digit})</span>"
            else:  # User confirmed the original prediction was correct
                status_icon = "✅"  # Use checkmark for confirmed correct
                header_detail = f"<span style='color: var(--color-success);'>(Confirmed: {final_digit})</span>"
        else:  # No feedback given yet
            status_icon = (
                "🎯" if confidence > 0.9 else "✅" if confidence > 0.6 else "⚠️"
            )
            header_detail = f"<span style='color: var(--color-primary);'>(Pred: {original_display_digit})</span>"

        header_text = f"{date_str} {header_detail}"
        # --- End Card Header Logic ---

        # --- Main Digit Display Logic ---
        if is_corrected and str(true_label) != str(predicted_digit):
            # Show strikethrough original + arrow + corrected
            digit_display_html = f"""<div style="display: flex; justify-content: center; align-items: center; gap: 0.75rem;">
                    <span style="font-size: 2rem; text-decoration: line-through; color: var(--color-text-muted);">{original_display_digit}</span>
                    <span style="font-size: 1.5rem; color: var(--color-text-light);">→</span>
                    <span style="font-size: 2.5rem; font-weight: bold; color: var(--color-info);">{final_digit}</span>
                </div>
            """
        else:
            # Show just the final/predicted digit (could be confirmed correct or unconfirmed)
            digit_display_html = f"""<span style="font-size: 2.5rem; font-weight: bold; color: var(--color-primary);">{final_digit}</span>
            """

        timestamp_html = f"""<div style="text-align: center; color: #111111; font-size: 0.875rem; padding: 0 0.5rem 0.5rem 0.5rem;">
                {timestamp_str}
            </div>
        """
        # --- End Main Digit Display Logic ---

        # Use st.container for better isolation and potential borders
        with st.container():
            # Use markdown for layout within the container
            st.markdown(
                f"""<div class="history-card" style="border: 1px solid var(--color-border); border-radius: var(--border-radius-md); padding: 0.8rem; margin-bottom: 1rem; background-color: var(--color-card); box-shadow: var(--shadow-sm);">
                 <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem; padding-bottom: 0.4rem; border-bottom: 1px dashed var(--color-border);">
                     <span style="font-weight: 500; font-size: 0.95rem; color: var(--color-text)">{status_icon} {header_text}</span>
                 </div>
                 <div style="display: flex; justify-content: space-between; align-items: center; font-size: 0.9rem; margin-bottom: 0.75rem;">
                     <span style="color: var(--color-text);">{input_icon} {input_type}</span>
                     <span style="font-weight: 600; color: {confidence_color};">{confidence_pct}</span>
                 </div>
                 <div style="text-align: center; padding: 0.5rem 0; background-color: var(--color-background-alt); border-radius: var(--border-radius-md); margin-bottom: 0.75rem;">
                     {digit_display_html}{timestamp_html}
             """,
                unsafe_allow_html=True,
            )

            # Delete button associated with this card
            if db_id is not None:
                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{db_id}",
                    type="secondary",
                    help="Delete this record",
                    use_container_width=True,
                ):
                    st.session_state.delete_db_id = db_id
                    st.session_state.show_delete_confirm = True
                    st.rerun()  # Rerun to show confirmation

    def render(self) -> None:
        """Render the history view content."""
        self._initialize_session_state()
        data = self._load_view_data()

        welcome = data.get("welcome_card", {})
        if welcome:
            WelcomeCard(
                title=welcome.get("title", "Prediction History"),
                content=welcome.get("content", "Browse predictions."),
                icon=welcome.get("icon", "📊"),
            ).display()
            st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

        self._render_filters()
        self._handle_delete()  # Render confirmation dialog if needed

        # Fetch paginated data directly using HistoryState (which calls db_manager)
        try:
            current_page = st.session_state.get("history_page", 1)
            items_per_page = st.session_state.get("history_items_per_page", 12)
            current_digit_filter = st.session_state.get("history_filter_digit")
            current_min_confidence = st.session_state.get(
                "history_filter_min_confidence", 0.0
            )
            current_sort_by = st.session_state.get("history_sort_by", "timestamp")
            current_sort_order = st.session_state.get("history_sort_order", "desc")

            # Call the static method on HistoryState
            page_items, total_items = HistoryState.get_paginated_history(
                page=current_page,
                page_size=items_per_page,
                digit_filter=current_digit_filter,
                min_confidence=current_min_confidence,
                sort_by=current_sort_by,
                sort_order=current_sort_order,
            )

            if total_items == 0:
                if current_digit_filter is not None or current_min_confidence > 0:
                    self._render_empty_state("No predictions match your filters.")
                else:
                    self._render_empty_state()
                return

            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

            if current_page > total_pages:
                st.session_state.history_page = total_pages
                self._logger.info(f"Adjusted current page to {total_pages}. Rerunning.")
                st.rerun()
                return

            self._render_pagination(total_items, total_pages, position="top")
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            self._render_history_entries(page_items)
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            self._render_pagination(total_items, total_pages, position="bottom")
            self._render_clear_all_button()

        except Exception as e:
            self._logger.error(f"Failed to fetch or render history: {e}", exc_info=True)
            st.error("🚨 Could not load prediction history. Please check logs.")

        tips_data = data.get("tips", {})
        if tips_data and tips_data.get("items"):
            st.markdown("<hr style='margin: 2rem 0 1rem 0;'>", unsafe_allow_html=True)
            list_items = "".join(
                [f"<li>{tip}</li>" for tip in tips_data.get("items", [])]
            )
            content = f"<ul style='padding-left: 20px; margin: 0;'>{list_items}</ul>"
            FeatureCard(
                title=tips_data.get("title", "💡 Tips"), content=content, icon=""
            ).display()

    def render(self) -> None:
        """Render the history view content."""
        self._initialize_session_state()  # Ensure state keys are attempted to be initialized
        data = self._load_view_data()

        welcome = data.get("welcome_card", {})
        if welcome:
            WelcomeCard(
                title=welcome.get("title", "Prediction History"),
                content=welcome.get("content", "Browse predictions."),
                icon=welcome.get("icon", "📊"),
            ).display()
            st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)

        self._render_filters()
        self._handle_delete()  # Handle delete confirmation logic if active

        # Fetch paginated data directly using HistoryState (which calls db_manager)
        try:
            # Use .get() for robustness when accessing state for arguments
            current_page = st.session_state.get("history_page", 1)
            items_per_page = st.session_state.get("history_items_per_page", 12)
            current_digit_filter = st.session_state.get(
                "history_filter_digit"
            )  # Can be None
            current_min_confidence = st.session_state.get(
                "history_filter_min_confidence", 0.0
            )
            current_sort_by = st.session_state.get(
                "history_sort_by", "timestamp"
            )  # Default sort
            current_sort_order = st.session_state.get(
                "history_sort_order", "desc"
            )  # Default order

            # Call the static method on HistoryState
            page_items, total_items = HistoryState.get_paginated_history(
                page=current_page,
                page_size=items_per_page,
                digit_filter=current_digit_filter,
                min_confidence=current_min_confidence,
                sort_by=current_sort_by,
                sort_order=current_sort_order,  # Pass the safely retrieved value
            )

            if total_items == 0:
                # Check if filters are active to show a more specific message
                if current_digit_filter is not None or current_min_confidence > 0:
                    self._render_empty_state("No predictions match your filters.")
                else:
                    self._render_empty_state()  # Default empty message
                return  # Stop rendering if no items

            total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)

            # Ensure current page is valid after filtering/deletion might change total pages
            if current_page > total_pages:
                st.session_state.history_page = total_pages  # Adjust if needed
                # We might need to re-fetch data if the page number changed, simple approach is rerun:
                self._logger.info(
                    f"Adjusted current page from {current_page} to {total_pages}. Rerunning."
                )
                st.rerun()
                return  # Stop current render pass after rerun request

            # Render pagination controls at the top
            self._render_pagination(total_items, total_pages, position="top")
            st.markdown(
                "<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True
            )  # Spacer

            # Render history entries for the current page
            self._render_history_entries(page_items)
            st.markdown(
                "<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True
            )  # Spacer

            # Render pagination controls at the bottom
            self._render_pagination(total_items, total_pages, position="bottom")

            # Render clear all button
            self._render_clear_all_button()

        except Exception as e:
            self._logger.error(f"Failed to fetch or render history: {e}", exc_info=True)
            st.error(
                "🚨 Could not load prediction history. Please check the database connection and logs."
            )

        # Render tips card if available
        tips_data = data.get("tips", {})
        if tips_data and tips_data.get("items"):
            st.markdown("<hr style='margin: 2rem 0 1rem 0;'>", unsafe_allow_html=True)
            list_items = "".join(
                [f"<li>{tip}</li>" for tip in tips_data.get("items", [])]
            )
            content = f"<ul style='padding-left: 20px; margin: 0;'>{list_items}</ul>"
            FeatureCard(
                title=tips_data.get("title", "💡 Tips"), content=content, icon=""
            ).display()
