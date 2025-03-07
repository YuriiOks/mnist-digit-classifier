import streamlit as st

def render_home():
    """Render the home page."""
    # Home page with cards
    st.markdown("""
    <div class="content-card">
        <h1>MNIST Digit Classifier</h1>
        <p style="font-size: 1.2rem; margin-top: 1rem;">
            Welcome to the MNIST Digit Classifier application! This app uses machine learning to recognize handwritten digits.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="content-card">
            <h3 style="color: #4CA1AF;">✏️ Draw a Digit</h3>
            <p>Use the interactive canvas to draw any digit from 0-9 and see the prediction in real-time.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="content-card">
            <h3 style="color: #4CA1AF;">🔍 High Accuracy</h3>
            <p>Our model is trained on thousands of handwritten digit samples for high prediction accuracy.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="content-card">
            <h3 style="color: #4CA1AF;">📊 Track History</h3>
            <p>Review your previous predictions and track the model's performance over time.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # How to use section
    st.markdown("""
    <div class="content-card">
        <h2>How to Use</h2>
        <ol style="margin-left: 1.5rem; margin-top: 1rem;">
            <li>Go to the <b>Drawing</b> section using the sidebar menu</li>
            <li>Draw a digit on the canvas provided</li>
            <li>Click the <b>Predict</b> button to see the result</li>
            <li>Check your prediction history in the <b>History</b> section</li>
        </ol>
    </div>
    """, unsafe_allow_html=True) 