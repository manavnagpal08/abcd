import streamlit as st
# Import logging functions
from utils.logger import log_user_action, update_metrics_summary

# --- Logout Style ---
st.markdown("""
<style>
.logout-box {
    margin-top: 4rem;
    padding: 2rem;
    background: linear-gradient(135deg, #fdfbfb, #ebedee);
    border-radius: 20px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    animation: fadeSlideOut 0.8s ease;
}
@keyframes fadeSlideOut {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
.logout-box h2 {
    color: #00cec9;
    font-weight: 700;
}
.logout-box p {
    font-size: 1.1rem;
    color: #555;
}
</style>
""", unsafe_allow_html=True)

# --- Logout Logic ---
# Log the logout action before clearing the session state
if 'user_email' in st.session_state and st.session_state.user_email:
    log_user_action(st.session_state.user_email, "LOGOUT_SUCCESS")
    update_metrics_summary("total_logouts", 1)
    update_metrics_summary("user_logouts", 1, user_email=st.session_state.user_email)
else:
    # Handle cases where user_email might not be set (e.g., direct access to logout page)
    log_user_action("unknown_user", "LOGOUT_ATTEMPT_UNAUTHENTICATED")

# Clear authentication status
st.session_state.authenticated = False
# Optionally clear other sensitive session state data upon logout
# For example:
if 'user_email' in st.session_state:
    del st.session_state.user_email
if 'screening_results' in st.session_state:
    del st.session_state.screening_results
# You might want to keep other session state variables if they persist across logins,
# but for a complete logout, clearing is often preferred.


# --- Message UI ---
st.markdown("""
<div class="logout-box">
    <h2>🚪 You’ve been logged out!</h2>
    <p>Thank you for using the HR Admin Panel.<br>
    You can close this tab or <a href="/">log in again</a> anytime.</p>
</div>
""", unsafe_allow_html=True)
