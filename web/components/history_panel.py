import streamlit as st
from utils.db import get_prediction_history

def render_history_panel(db_connection):
    """Render the prediction history section."""
    st.markdown('<div class="history-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Prediction History</div>', unsafe_allow_html=True)
    
    # Get history from database
    try:
        if db_connection is not None:
            history = get_prediction_history(db_connection)
            
            if history and len(history) > 0:
                # Create a table for history
                history_table = """
                <div class="history-table">
                    <div class="history-table-header">
                        <div class="history-cell">Time</div>
                        <div class="history-cell">Prediction</div>
                        <div class="history-cell">Actual</div>
                        <div class="history-cell">Confidence</div>
                    </div>
                """
                
                # Add rows
                for item in history:
                    history_table += f"""
                    <div class="history-table-row">
                        <div class="history-cell">{item['timestamp'].strftime('%H:%M:%S')}</div>
                        <div class="history-cell">{item['prediction']}</div>
                        <div class="history-cell">{item['true_label'] if item['true_label'] is not None else '-'}</div>
                        <div class="history-cell">{item['confidence']:.2%}</div>
                    </div>
                    """
                
                history_table += "</div>"
                st.markdown(history_table, unsafe_allow_html=True)
            else:
                # Show empty state with explanation
                st.markdown("""
                <div class="empty-history">
                    <div class="empty-icon">📊</div>
                    <div class="empty-text">Your prediction history will appear here after you make predictions and provide feedback.</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # No database connection
            st.warning("Database connection not available. Prediction history will not be saved.")
            
            # Use session-based history if available
            if "digit_history" in st.session_state and len(st.session_state.digit_history) > 0:
                history_table = """
                <div class="history-table">
                    <div class="history-table-header">
                        <div class="history-cell">Time</div>
                        <div class="history-cell">Prediction</div>
                        <div class="history-cell">Actual</div>
                        <div class="history-cell">Confidence</div>
                    </div>
                """
                
                for item in st.session_state.digit_history:
                    history_table += f"""
                    <div class="history-table-row">
                        <div class="history-cell">{item['timestamp']}</div>
                        <div class="history-cell">{item['prediction']}</div>
                        <div class="history-cell">{item['true_label']}</div>
                        <div class="history-cell">{item['confidence']:.2%}</div>
                    </div>
                    """
                
                history_table += "</div>"
                st.markdown(history_table, unsafe_allow_html=True)
            else:
                st.info("No prediction history yet. Start drawing and predicting!")
    except Exception as e:
        st.error(f"Error retrieving history: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True) 