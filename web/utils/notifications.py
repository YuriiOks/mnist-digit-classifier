import streamlit as st
from datetime import datetime
import uuid

def create_toast_container():
    """Create container for toast notifications."""
    if 'toast_messages' not in st.session_state:
        st.session_state.toast_messages = []
    
    toast_html = """
    <div id="toast-container"></div>
    
    <style>
    #toast-container {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
    }
    
    .toast {
        background-color: var(--bg-secondary);
        color: var(--text-primary);
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;
        min-width: 280px;
        max-width: 320px;
        box-shadow: var(--shadow);
        display: flex;
        align-items: center;
        overflow: hidden;
        animation: toast-in 0.3s ease-out forwards;
        position: relative;
    }
    
    .toast.closing {
        animation: toast-out 0.3s ease-in forwards;
    }
    
    .toast-success {
        border-left: 4px solid var(--success);
    }
    
    .toast-error {
        border-left: 4px solid var(--danger);
    }
    
    .toast-info {
        border-left: 4px solid var(--accent-primary);
    }
    
    .toast-warning {
        border-left: 4px solid var(--warning);
    }
    
    .toast-icon {
        margin-right: 12px;
        font-size: 20px;
    }
    
    .toast-message {
        flex: 1;
    }
    
    .toast-close {
        cursor: pointer;
        margin-left: 12px;
        opacity: 0.6;
    }
    
    .toast-close:hover {
        opacity: 1;
    }
    
    .toast-progress {
        position: absolute;
        bottom: 0;
        left: 0;
        height: 3px;
        background-color: rgba(0, 0, 0, 0.1);
    }
    
    @keyframes toast-in {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes toast-out {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    @keyframes progress {
        from { width: 100%; }
        to { width: 0; }
    }
    </style>
    
    <script>
    function createToast(type, message, duration) {
        const container = document.getElementById('toast-container');
        if (!container) return;
        
        const id = 'toast-' + Date.now();
        const icons = {
            'success': '✅',
            'error': '❌',
            'info': 'ℹ️',
            'warning': '⚠️'
        };
        
        const toast = document.createElement('div');
        toast.id = id;
        toast.className = 'toast toast-' + type;
        toast.innerHTML = `
            <div class="toast-icon">${icons[type]}</div>
            <div class="toast-message">${message}</div>
            <div class="toast-close" onclick="removeToast('${id}')">✕</div>
            <div class="toast-progress" style="animation: progress ${duration/1000}s linear;"></div>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => removeToast(id), duration);
    }
    
    function removeToast(id) {
        const toast = document.getElementById(id);
        if (toast && !toast.classList.contains('closing')) {
            toast.classList.add('closing');
            setTimeout(() => {
                if (toast.parentNode) {
                    toast.parentNode.removeChild(toast);
                }
            }, 300);
        }
    }
    </script>
    """
    
    st.markdown(toast_html, unsafe_allow_html=True)

def show_toast(message, type="info", duration=3000):
    """
    Show a toast notification.
    
    Args:
        message: Message to display
        type: Type of notification (success, error, info, warning)
        duration: Duration in milliseconds
    """
    # Add to session state for tracking
    toast_id = str(uuid.uuid4())
    st.session_state.toast_messages.append({
        "id": toast_id,
        "message": message,
        "type": type,
        "duration": duration,
        "time": datetime.now().isoformat()
    })
    
    # Create JavaScript to show the toast
    js = f"""
    <script>
    createToast('{type}', '{message}', {duration});
    </script>
    """
    
    st.markdown(js, unsafe_allow_html=True) 