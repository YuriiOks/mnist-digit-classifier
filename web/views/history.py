import streamlit as st

def render_history():
    """Render the prediction history page."""
    st.markdown("""
    <div class="content-card">
        <h1>Prediction History</h1>
        <p>Review your previous predictions and model performance.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.markdown("""
        <div class="content-card">
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 200px;">
                <div style="font-size: 3rem; color: #ccc; margin-bottom: 1rem;">📊</div>
                <div style="text-align: center; color: #666;">
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
        cols = st.columns([0.2, 0.15, 0.2, 0.2, 0.15, 0.1])
        cols[0].markdown("<b>Time</b>", unsafe_allow_html=True)
        cols[1].markdown("<b>Image</b>", unsafe_allow_html=True)
        cols[2].markdown("<b>Prediction</b>", unsafe_allow_html=True)
        cols[3].markdown("<b>Confidence</b>", unsafe_allow_html=True)
        cols[4].markdown("<b>Actual</b>", unsafe_allow_html=True)
        cols[5].markdown("<b>Correct</b>", unsafe_allow_html=True)
        
        # Add a separator
        st.markdown("<hr style='margin: 0.5rem 0; border-color: #eee;'>", unsafe_allow_html=True)
        
        # Display each history item
        for i, item in enumerate(reversed(st.session_state.history)):
            cols = st.columns([0.2, 0.15, 0.2, 0.2, 0.15, 0.1])
            cols[0].write(item["time"])
            
            # For the image thumbnail
            cols[1].image(item["image"], width=40)
            
            # Prediction with custom styling
            cols[2].markdown(f"<div style='font-size: 1.2rem; font-weight: bold;'>{item['prediction']}</div>", 
                            unsafe_allow_html=True)
            
            # Confidence with progress bar
            confidence = item["confidence"]
            conf_color = "green" if confidence > 0.9 else "orange" if confidence > 0.7 else "red"
            cols[3].markdown(f"""
            <div style='margin-top: 5px;'>
                <div style='background-color: #eee; border-radius: 10px; height: 10px; width: 100%;'>
                    <div style='background-color: {conf_color}; border-radius: 10px; height: 10px; width: {confidence*100}%;'></div>
                </div>
                <div style='text-align: center; font-size: 0.8rem;'>{confidence:.0%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Actual digit (if provided through feedback)
            actual = item.get("actual")
            cols[4].write("Not provided" if actual is None else actual)
            
            # Check if prediction was correct
            if actual is not None:
                is_correct = item["prediction"] == actual
                cols[5].markdown(f"{'✅' if is_correct else '❌'}", unsafe_allow_html=True)
            else:
                cols[5].write("—")
            
            # Add a separator between rows
            if i < len(st.session_state.history) - 1:
                st.markdown("<hr style='margin: 0.5rem 0; border-color: #eee;'>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Add some analytics
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
        
        # Display accuracy
        st.markdown(f"""
        <div style='text-align: center; margin: 1rem 0;'>
            <div style='font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem;'>Accuracy</div>
            <div style='font-size: 3rem; font-weight: bold; color: {'green' if accuracy > 0.8 else 'orange' if accuracy > 0.5 else 'red'};'>
                {accuracy:.0%}
            </div>
            <div style='margin-top: 0.5rem; color: #666;'>Based on {len(feedback_items)} predictions with feedback</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='text-align: center; margin: 1rem 0;'>
            <div style='font-size: 1.2rem; color: #666;'>
                No feedback provided yet. Provide feedback by confirming if predictions are correct.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True) 