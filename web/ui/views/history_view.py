# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/history_view.py
# Description: History view implementation with database integration
# Created: 2025-03-17
# Updated: 2025-03-24

import streamlit as st
import datetime
import pandas as pd
import logging
import json
from typing import Optional

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
            description="View and manage your past digit predictions."
        )
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def _initialize_session_state(self) -> None:
        """Initialize session state variables for history filtering and pagination."""
        # Filter controls
        if "history_filter_date" not in st.session_state:
            st.session_state.history_filter_date = None
        if "history_filter_digit" not in st.session_state:
            st.session_state.history_filter_digit = None
        if "history_filter_min_confidence" not in st.session_state:
            st.session_state.history_filter_min_confidence = 0.0
        if "history_sort_by" not in st.session_state:
            st.session_state.history_sort_by = "newest"
            
        # Pagination controls
        if "history_page" not in st.session_state:
            st.session_state.history_page = 1
        if "history_items_per_page" not in st.session_state:
            st.session_state.history_items_per_page = 12
            
        # Entry deletion state
        if "delete_id" not in st.session_state:
            st.session_state.delete_id = None

    def _load_view_data(self):
        """
        Load necessary JSON data for the History view.
        """
        data = resource_manager.load_json_resource("history/history_view.json")
        if not data:
            data = {}  # fallback

        return data
    
    def _render_empty_state(self) -> None:
        """Render the empty state when no history is available."""
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; background-color: var(--color-background-alt); border-radius: 8px; margin: 2rem 0;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h3>No Prediction History Yet</h3>
            <p>Your prediction history will appear here once you make some predictions.</p>
            <p>Go to the Draw view to get started!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a button to navigate to draw view
        if st.button("Go to Draw View", key="go_to_draw", type="primary"):
            NavigationState.set_active_view("draw")
            st.rerun()
    
    def _render_filters(self) -> None:
        """
        Render filter controls and return filter parameters.
        
        Returns:
            Tuple of (digit_filter, min_confidence, sort_by)
        """
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        # Digit filter
        with filter_col1:
            digit_options = [None] + list(range(10))
            digit_filter = st.selectbox(
                "Filter by Digit",
                options=digit_options,
                format_func=lambda x: "All Digits" if x is None else str(x),
                key="digit_filter",
                index=0 if st.session_state.history_filter_digit is None else 
                      digit_options.index(st.session_state.history_filter_digit)
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
                key="confidence_filter"
            )
            st.session_state.history_filter_min_confidence = confidence_filter
        
        # Sort options
        with filter_col3:
            sort_options = {
                "newest": "Newest First",
                "oldest": "Oldest First",
                "highest_conf": "Highest Confidence",
                "lowest_conf": "Lowest Confidence"
            }
            sort_by = st.selectbox(
                "Sort By",
                options=list(sort_options.keys()),
                format_func=lambda x: sort_options[x],
                key="sort_by",
                index=list(sort_options.keys()).index(st.session_state.history_sort_by)
            )
            st.session_state.history_sort_by = sort_by
            
        # Update pagination if filters changed
        if (st.session_state.get('prev_digit_filter') != st.session_state.history_filter_digit or
            st.session_state.get('prev_confidence_filter') != st.session_state.history_filter_min_confidence or
            st.session_state.get('prev_sort_by') != st.session_state.history_sort_by):
            st.session_state.history_page = 1
        
        # Update previous filter values
        st.session_state.prev_digit_filter = st.session_state.history_filter_digit
        st.session_state.prev_confidence_filter = st.session_state.history_filter_min_confidence
        st.session_state.prev_sort_by = st.session_state.history_sort_by
    
    def _render_history_entries(self, page_items: list) -> None:
        """
        Render history entries in a grid layout.
        
        Args:
            page_items: List of prediction entries to display
        """
        if not page_items:
            st.info("No entries match your filter criteria. Try adjusting the filters.")
            return
            
        # Create grid layout with 3 columns
        num_items = len(page_items)
        rows = (num_items + 2) // 3  # Ceiling division
        
        for row in range(rows):
            cols = st.columns(3)
            for col in range(3):
                idx = row * 3 + col
                if idx < num_items:
                    item = page_items[idx]
                    with cols[col]:
                        # Format timestamp
                        timestamp = item.get('timestamp')
                        if isinstance(timestamp, str):
                            try:
                                timestamp = datetime.datetime.fromisoformat(timestamp)
                            except (ValueError, TypeError):
                                timestamp = datetime.datetime.now()
                        
                        timestamp_str = timestamp.strftime("%b %d, %Y %H:%M") if isinstance(timestamp, datetime.datetime) else "Unknown"
                        
                        # Format confidence
                        confidence_pct = f"{item.get('confidence', 0) * 100:.1f}%"
                        
                        # Determine input type icon
                        input_type = item.get("input_type", "canvas")
                        input_icon = "✏️" if input_type == "canvas" else "📷" if input_type == "upload" else "🔗"
                        
                        # Create entry container
                        st.container().markdown(f"""
                        <div style="border: 1px solid var(--color-border); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background-color: var(--color-card);">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                <div style="font-size: 0.8rem; color: var(--color-text);">{timestamp_str}</div>
                                <div style="font-size: 0.8rem; color: var(--color-text);"><span>Confidence: {confidence_pct}</span></div>
                            </div>
                            <div style="display: flex; gap: 1rem; align-items: center;">
                                <div style="width: 80px; height: 80px; display: flex; justify-content: center; align-items: center; background-color: var(--color-background); border-radius: 4px;">
                                    <span style="font-size: 2.5rem; font-weight: bold; color: var(--color-primary);">{item.get('digit', '?')}</span>
                                </div>
                                <div>
                                    <div style="font-weight: bold; margin-bottom: 0.25rem;">Prediction: {item.get('digit', '?')}</div>
                                    {f'<div style="color: var(--color-success); font-size: 0.9rem;">Corrected to: {item.get("user_correction")}</div>' if item.get("user_correction") is not None else ''}
                                    <div style="font-size: 0.9rem; color: var(--color-text);">Input: {input_icon} {input_type.capitalize()}</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Add delete button as a normal Streamlit button
                        # This avoids issues with HTML rendering and JavaScript
                        if st.button("Delete", key=f"delete_{item.get('id')}", type="secondary"):
                            st.session_state.delete_id = item.get('id')
                            st.rerun()
    
    def _render_pagination(self, total_items: int, total_pages: int, position: str = "top") -> None:
        """
        Render pagination controls.
        
        Args:
            total_items: Total number of items
            total_pages: Total number of pages
            position: Position of pagination controls ("top" or "bottom")
        """
        # Calculate start and end indices
        items_per_page = st.session_state.history_items_per_page
        start_idx = (st.session_state.history_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        
        # Display stats and pagination info
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; margin-bottom: 1rem;">
            <div>
                <span>Showing {start_idx + 1 if total_items > 0 else 0}-{end_idx} of {total_items} predictions</span>
            </div>
            <div>
                <span>Page {st.session_state.history_page} of {max(1, total_pages)}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Pagination controls
        if total_pages > 1:
            pagination_cols = st.columns([1, 3, 1])
            
            # Create unique keys for each button based on position
            prev_key = f"prev_page_{position}"
            next_key = f"next_page_{position}"
            
            with pagination_cols[0]:
                if st.button("← Previous", key=prev_key, disabled=st.session_state.history_page <= 1):
                    st.session_state.history_page -= 1
                    st.rerun()
            
            with pagination_cols[1]:
                # Page selector
                page_numbers = list(range(1, total_pages + 1))
                selected_page = st.select_slider(
                    "Page selector",  # Added a label to fix accessibility warning
                    options=page_numbers,
                    value=st.session_state.history_page,
                    key=f"page_selector_{position}",
                    label_visibility="collapsed"  # Hide the label but still provide it
                )
                
                if selected_page != st.session_state.history_page:
                    st.session_state.history_page = selected_page
                    st.rerun()
            
            with pagination_cols[2]:
                if st.button("Next →", key=next_key, disabled=st.session_state.history_page >= total_pages):
                    st.session_state.history_page += 1
                    st.rerun()
    
    def _render_clear_all_button(self) -> None:
        """Render the clear history button with confirmation."""
        st.markdown("<hr>", unsafe_allow_html=True)
        
        if st.button("Clear All History", key="clear_history", type="secondary"):
            # Show confirmation checkbox
            confirm_clear = st.checkbox("Confirm that you want to delete all prediction history", key="confirm_clear")
            
            if confirm_clear:
                # Clear history and reset filters
                HistoryState.clear_history()
                st.session_state.history_filter_date = None
                st.session_state.history_filter_digit = None
                st.session_state.history_filter_min_confidence = 0.0
                st.session_state.history_sort_by = "newest"
                st.session_state.history_page = 1
                st.success("Prediction history cleared successfully.")
                st.rerun()
    
    def _handle_delete(self) -> None:
        """Handle deletion of history entries."""
        # Check if we have a pending deletion
        if st.session_state.delete_id:
            entry_id = st.session_state.delete_id
            self._logger.info(f"Deleting history entry: {entry_id}")
            
            # Delete the entry
            success = HistoryState.delete_entry(entry_id)
            
            if success:
                st.success(f"Entry deleted successfully.", icon="✅")
            else:
                st.error(f"Failed to delete entry.", icon="❌")
            
            # Reset delete_id
            st.session_state.delete_id = None
            
            # Force a rerun to refresh the view
            st.rerun()
        
    def render(self) -> None:
        """Render the history view content."""
        # Initialize session state variables
        self._initialize_session_state()
        
        # Handle any pending deletions first
        self._handle_delete()

        # Load view configuration data
        data = self._load_view_data()

        # Render welcome card if it exists in the data
        welcome = data.get("welcome_card", {})
        if welcome:
            welcome_card = WelcomeCard(
                title=welcome.get("title", "History"),
                content=welcome.get("content", "View your predictions."),
                icon=welcome.get("icon", "📊")
            )
            welcome_card.display()
        
        # Render filters
        self._render_filters()
        
        # Get total count with filters applied
        total_items = HistoryState.get_history_size(
            digit_filter=st.session_state.history_filter_digit,
            min_confidence=st.session_state.history_filter_min_confidence
        )
        
        # If no items, show empty state
        if total_items == 0:
            self._render_empty_state()
            return
        
        # Calculate pagination
        items_per_page = st.session_state.history_items_per_page
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
        
        # Ensure current page is valid
        st.session_state.history_page = min(max(1, st.session_state.history_page), total_pages)
        
        # Get items for current page with filters
        page_items = HistoryState.get_paginated_history(
            page=st.session_state.history_page,
            page_size=items_per_page,
            digit_filter=st.session_state.history_filter_digit,
            min_confidence=st.session_state.history_filter_min_confidence,
            sort_by=st.session_state.history_sort_by
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
            FeatureCard(
                title=tips_data.get("title", "Tips"),
                content="<ul>" + "".join(f"<li>{tip}</li>" for tip in tips_data.get("items", [])) + "</ul>",
                icon="💡"
            ).display()