# MNIST Digit Classifier
# Copyright (c) 2025
# File: ui/views/history_view.py
# Description: History view implementation
# Created: 2025-03-17

import streamlit as st
import datetime
import pandas as pd
import logging
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
        
    def _initialize_session_state(self) -> None:
        """Initialize session state variables for history filtering and pagination."""
        if not hasattr(st.session_state, 'history_filter_date'):
            st.session_state.history_filter_date = None
        if not hasattr(st.session_state, 'history_filter_digit'):
            st.session_state.history_filter_digit = None
        if not hasattr(st.session_state, 'history_filter_min_confidence'):
            st.session_state.history_filter_min_confidence = 0.0
        if not hasattr(st.session_state, 'history_sort_by'):
            st.session_state.history_sort_by = "newest"
        if not hasattr(st.session_state, 'history_page'):
            st.session_state.history_page = 1
        if not hasattr(st.session_state, 'history_items_per_page'):
            st.session_state.history_items_per_page = 12

    def _load_view_data(self):
        """
        Load necessary JSON data for the History/Settings view.
        """
        data = resource_manager.load_json_resource("history/history_view.json")  # or "settings/settings_view.json"
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
    
    def _render_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Render filter controls and apply filters to the dataframe."""
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        # Digit filter
        with filter_col1:
            digit_filter = st.selectbox(
                "Filter by Digit",
                options=[None, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                format_func=lambda x: "All Digits" if x is None else str(x),
                key="digit_filter"
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
                key="sort_by"
            )
            st.session_state.history_sort_by = sort_by
        
        # Apply filters
        filtered_df = df.copy()
        
        if st.session_state.history_filter_digit is not None:
            filtered_df = filtered_df[filtered_df['digit'] == st.session_state.history_filter_digit]
        
        if st.session_state.history_filter_min_confidence > 0:
            filtered_df = filtered_df[filtered_df['confidence'] >= st.session_state.history_filter_min_confidence]
        
        # Apply sorting
        if st.session_state.history_sort_by == "newest":
            filtered_df = filtered_df.sort_values('timestamp', ascending=False)
        elif st.session_state.history_sort_by == "oldest":
            filtered_df = filtered_df.sort_values('timestamp', ascending=True)
        elif st.session_state.history_sort_by == "highest_conf":
            filtered_df = filtered_df.sort_values('confidence', ascending=False)
        elif st.session_state.history_sort_by == "lowest_conf":
            filtered_df = filtered_df.sort_values('confidence', ascending=True)
        
        # Reset page number if filters changed
        if (st.session_state.get('prev_digit_filter') != st.session_state.history_filter_digit or
            st.session_state.get('prev_confidence_filter') != st.session_state.history_filter_min_confidence):
            st.session_state.history_page = 1
        
        # Update previous filter values
        st.session_state.prev_digit_filter = st.session_state.history_filter_digit
        st.session_state.prev_confidence_filter = st.session_state.history_filter_min_confidence
        
        return filtered_df
    
    def _render_history_entries(self, page_items: pd.DataFrame) -> None:
        """Render history entries in a grid layout."""
        if not page_items.empty:
            # Create grid layout with 3 columns
            num_items = len(page_items)
            rows = (num_items + 2) // 3  # Ceiling division
            
            for row in range(rows):
                cols = st.columns(3)
                for col in range(3):
                    idx = row * 3 + col
                    if idx < num_items:
                        item = page_items.iloc[idx]
                        with cols[col]:
                            # Format timestamp
                            timestamp_str = item['timestamp'].strftime("%b %d, %Y %H:%M")
                            
                            # Format confidence
                            confidence_pct = f"{item['confidence'] * 100:.1f}%"
                            
                            # Create card with prediction info
                            st.markdown(f"""
                            <div style="border: 1px solid var(--color-border); border-radius: 8px; padding: 1rem; margin-bottom: 1rem; background-color: var(--color-card);">
                                <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                                    <div style="font-size: 0.8rem; color: var(--color-text);">{timestamp_str}</div>
                                    <div style="font-size: 0.8rem; color: var(--color-text);"><span class="highlight">Confidence: {confidence_pct}</span></div>
                                </div>
                                <div style="display: flex; gap: 1rem; align-items: center;">
                                    <div style="width: 80px; height: 80px; display: flex; justify-content: center; align-items: center; background-color: var(--color-background); border-radius: 4px;">
                                        <span style="font-size: 2.5rem; font-weight: bold; color: var(--color-primary);">{item['digit']}</span>
                                    </div>
                                    <div>
                                        <div style="font-weight: bold; margin-bottom: 0.25rem;">Prediction: {item['digit']}</div>
                                        {f'<div style="color: var(--color-success); font-size: 0.9rem;">Corrected to: {item["corrected_digit"]}</div>' if item["corrected_digit"] is not None else ''}
                                        <div style="font-size: 0.9rem; color: var(--color-text);">Input: {item["input_type"].capitalize()}</div>
                                    </div>
                                </div>
                                <div style="display: flex; justify-content: flex-end; margin-top: 0.5rem;">
                                    <button
                                        onclick="Streamlit.setComponentValue({{action: 'delete', id: '{item['id']}'}});"
                                        style="background: none; border: none; cursor: pointer; color: var(--color-error); font-size: 0.8rem;"
                                    >
                                        Delete
                                    </button>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
# Replace the _render_pagination method in history_view.py with this version

    def _render_pagination(self, total_items: int, total_pages: int) -> None:
        """Render pagination controls using streamlit-pagination."""
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
        
        # Generate a unique key for this specific instance
        import time
        import random
        unique_key = f"history_pagination_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Simple manual pagination - more reliable than the library
        col1, col2, col3 = st.columns([1, 4, 1])
        
        with col1:
            if st.button("← Previous", 
                        key=f"prev_page_{unique_key}", 
                        disabled=st.session_state.history_page <= 1,
                        use_container_width=True):
                st.session_state.history_page = max(1, st.session_state.history_page - 1)
                st.rerun()
        
        with col2:
            # Create page number buttons for reasonable number of pages
            if total_pages <= 10:
                page_cols = st.columns(min(total_pages, 10))
                for i, col in enumerate(page_cols):
                    page_num = i + 1
                    with col:
                        if st.button(f"{page_num}", 
                                    key=f"page_{page_num}_{unique_key}",
                                    type="primary" if page_num == st.session_state.history_page else "secondary"):
                            st.session_state.history_page = page_num
                            st.rerun()
            else:
                # For many pages, use a number input
                page_num = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=max(1, total_pages),
                    value=st.session_state.history_page,
                    step=1,
                    key=f"page_number_{unique_key}",
                    label_visibility="collapsed"
                )
                
                if page_num != st.session_state.history_page:
                    st.session_state.history_page = page_num
                    st.rerun()
        
        with col3:
            if st.button("Next →", 
                        key=f"next_page_{unique_key}", 
                        disabled=st.session_state.history_page >= total_pages,
                        use_container_width=True):
                st.session_state.history_page = min(total_pages, st.session_state.history_page + 1)
                st.rerun()
    def _render_clear_all_button(self) -> None:
        """Render the clear history button."""
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
    
    def _render_delete_handler(self) -> None:
        """Add JavaScript for handling delete buttons."""
        # Handle individual delete actions
        if "delete_id" in st.session_state and st.session_state.delete_id:
            entry_id = st.session_state.delete_id
            # Delete the entry
            # In a real implementation, you would call a method to delete specific entries
            # For example: HistoryState.delete_entry(entry_id)
            st.session_state.delete_id = None
            st.rerun()
        
        # Add JavaScript to handle delete buttons
        st.markdown("""
        <script>
        // Listen for messages from component
        window.addEventListener('message', function(event) {
            if (event.data.type === 'streamlit:componentOutput') {
                const data = event.data.value;
                if (data && data.action === 'delete') {
                    window.parent.postMessage({
                        type: 'streamlit:setComponentValue',
                        value: {
                            delete_id: data.id
                        }
                    }, '*');
                }
            }
        });
        </script>
        """, unsafe_allow_html=True)
        
    def render(self) -> None:
        """Render the history view content."""
        # Initialize session state variables
        self._initialize_session_state()

        data = self._load_view_data()

        # Example: Render a welcome card if it exists
        welcome = data.get("welcome_card", {})
        if welcome:
            welcome_card = WelcomeCard(
                title=welcome.get("title", "History"),
                content=welcome.get("content", "View your predictions."),
                icon=welcome.get("icon", "📊")
            )
            welcome_card.display()
        
        # Get all predictions from history state
        all_predictions = HistoryState.get_predictions()
        
        # If no predictions available, show empty state
        if not all_predictions:
            self._render_empty_state()
            return
        
        # Convert predictions to DataFrame for easier filtering and sorting
        history_data = []
        for pred in all_predictions:
            # Extract timestamp
            if isinstance(pred.get('timestamp'), str):
                try:
                    timestamp = datetime.datetime.fromisoformat(pred['timestamp'])
                except (ValueError, TypeError):
                    timestamp = datetime.datetime.now()
            elif isinstance(pred.get('timestamp'), datetime.datetime):
                timestamp = pred['timestamp']
            else:
                timestamp = datetime.datetime.now()
                
            # Extract other data
            history_data.append({
                'id': pred.get('id', ''),
                'digit': pred.get('digit', 0),
                'confidence': pred.get('confidence', 0.0),
                'timestamp': timestamp,
                'corrected_digit': pred.get('user_correction'),
                'image': pred.get('image'),
                'input_type': pred.get('input_type', 'canvas')
            })
        
        df = pd.DataFrame(history_data)
        
        # Render and apply filters
        filtered_df = self._render_filters(df)
        
        # Calculate pagination
        total_items = len(filtered_df)
        items_per_page = st.session_state.history_items_per_page
        total_pages = max(1, (total_items + items_per_page - 1) // items_per_page)
        
        # Ensure current page is valid
        st.session_state.history_page = min(max(1, st.session_state.history_page), total_pages)
        
        # Select items for current page
        start_idx = (st.session_state.history_page - 1) * items_per_page
        end_idx = min(start_idx + items_per_page, total_items)
        page_items = filtered_df.iloc[start_idx:end_idx]
        
        # Render pagination
        self._render_pagination(total_items, total_pages)
        
        # Render history entries
        self._render_history_entries(page_items)
        
        # Render pagination again at the bottom
        self._render_pagination(total_items, total_pages)
        
        # Render clear all button
        self._render_clear_all_button()
        
        # Render delete handler
        self._render_delete_handler()

        tips_data = data.get("tips", {})
        if tips_data:
            FeatureCard(
                title=tips_data.get("title", "Tips"),
                content="<ul>" + "".join(f"<li>{tip}</li>" for tip in tips_data.get("items", [])) + "</ul>",
                icon="💡"
            ).display()