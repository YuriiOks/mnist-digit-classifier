import streamlit as st
from utils.resource_loader import ResourceLoader

def render_history():
    """Render the prediction history page."""
    # Load history page styles
    ResourceLoader.load_css(["css/views/history.css"])
    
    st.markdown("""
    <div class="content-card">
        <h1>Prediction History</h1>
        <p>Review your previous predictions and model performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.markdown("""
        <div class="content-card">
            <div class="empty-history">
                <div class="empty-history-icon">📊</div>
                <div class="empty-history-text">
                    No prediction history yet. Start drawing digits to build your history!
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Display history in a table
        st.markdown("""
        <div class="content-card">
            <h2>Recent Predictions</h2>
        """, unsafe_allow_html=True)
        
        # Create columns for the table header
        st.markdown("""
        <div class="history-table-header">
            <div class="history-col time-col">Time</div>
            <div class="history-col image-col">Image</div>
            <div class="history-col prediction-col">Prediction</div>
            <div class="history-col confidence-col">Confidence</div>
            <div class="history-col actual-col">Actual</div>
            <div class="history-col correct-col">Correct</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get only the 10 most recent items
        recent_history = list(reversed(st.session_state.history))[:10]
        
        # Display each history item
        for i, item in enumerate(recent_history):
            # Start row
            st.markdown('<div class="history-table-row">', unsafe_allow_html=True)
            
            # Time column with method badge
            method = item.get("input_method", "draw")  # Default to "draw" if not specified
            method_badge = f'<span class="method-badge {method}">{method}</span>'
            
            st.markdown(
                f'<div class="history-col time-col">{method_badge} {item["time"]}</div>',
                unsafe_allow_html=True
            )
            
            # Image column
            col_image = st.container()
            with col_image:
                st.markdown('<div class="history-col image-col">', unsafe_allow_html=True)
                st.image(item["image"], width=40)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Prediction column
            st.markdown(
                f'<div class="history-col prediction-col">{item["prediction"]}</div>',
                unsafe_allow_html=True
            )
            
            # Confidence column with progress bar
            confidence = item["confidence"]
            conf_class = "high" if confidence > 0.9 else "medium" if confidence > 0.7 else "low"
            
            st.markdown(f"""
            <div class="history-col confidence-col">
                <div class="confidence-bar-container">
                    <div class="confidence-bar {conf_class}" style="width: {confidence*100}%;"></div>
                </div>
                <div class="confidence-value">{confidence:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Actual digit column
            actual = item.get("actual")
            actual_text = "Not provided" if actual is None else actual
            st.markdown(
                f'<div class="history-col actual-col">{actual_text}</div>',
                unsafe_allow_html=True
            )
            
            # Correct column
            correct_mark = "—"
            if actual is not None:
                is_correct = item["prediction"] == actual
                correct_mark = "✅" if is_correct else "❌"
            
            st.markdown(
                f'<div class="history-col correct-col">{correct_mark}</div>',
                unsafe_allow_html=True
            )
            
            # End row
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add analytics section
        render_analytics()

def render_analytics():
    """Render performance analytics section."""
    st.markdown("""
    <div class="content-card">
        <h2>Performance Analytics</h2>
    """, unsafe_allow_html=True)
    
    # Calculate accuracy from feedback
    feedback_items = [item for item in st.session_state.history if item.get("actual") is not None]
    if feedback_items:
        correct_items = [item for item in feedback_items if item["prediction"] == item["actual"]]
        accuracy = len(correct_items) / len(feedback_items)
        
        # Determine accuracy class
        accuracy_class = "high" if accuracy > 0.8 else "medium" if accuracy > 0.5 else "low"
        
        # Display accuracy
        st.markdown(f"""
        <div class="analytics-container">
            <div class="accuracy-label">Accuracy</div>
            <div class="accuracy-value {accuracy_class}">{accuracy:.0%}</div>
            <div class="feedback-count">Based on {len(feedback_items)} predictions with feedback</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="no-feedback">
            No feedback provided yet. Provide feedback by confirming if predictions are correct.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True) 