import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# Import logging and metrics retrieval functions
from utils.logger import get_user_activity_logs, get_system_events_logs, get_metrics_summary

def admin_panel_page():
    st.markdown('<div class="dashboard-header">🔒 Admin Panel</div>', unsafe_allow_html=True)
    st.write("Manage system logs, user activities, and performance metrics.")

    # --- Simple Role-Based Access Control (RBAC) ---
    # This is a basic check. For production, you'd want more robust RBAC
    # where the user's role is stored in session_state upon login.
    # Example: if st.session_state.get('user_role') != 'admin':
    if st.session_state.get('user_email') != 'admin@example.com': # Replace with your actual admin email
        st.warning("You do not have administrative access to this page.")
        return

    # --- Dynamic Styling for Dark Mode ---
    dark_mode = st.session_state.get('dark_mode_main', False) # Get dark mode status from session state

    st.subheader("Overview")
    metrics = get_metrics_summary()
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    # Safely get metrics, providing default 0 if key/date is missing
    total_screened_today = metrics.get('total_resumes_screened', {}).get(today_str, 0)
    total_emails_sent_today = metrics.get('total_emails_sent', {}).get(today_str, 0)
    # You'd need to log 'total_jds_managed' in manage_jds.py to get this
    total_jds_managed_today = metrics.get('total_jds_managed', {}).get(today_str, 0) 


    col1, col2, col3 = st.columns(3)
    col1.metric("Resumes Screened Today", total_screened_today)
    col2.metric("Emails Sent Today", total_emails_sent_today)
    col3.metric("JDs Managed Today", total_jds_managed_today) # Display the count

    st.markdown("---")

    # --- User Activity Log Section ---
    st.subheader("👤 User Activity Log")
    user_logs = get_user_activity_logs()
    df_user_logs = pd.DataFrame(user_logs)

    if not df_user_logs.empty:
        df_user_logs['timestamp'] = pd.to_datetime(df_user_logs['timestamp'])
        df_user_logs = df_user_logs.sort_values(by='timestamp', ascending=False)

        # Filters for User Activity Log
        col_ua1, col_ua2, col_ua3 = st.columns([1, 1, 2])
        with col_ua1:
            selected_user = st.selectbox("Filter by User:", ["All"] + sorted(df_user_logs['user_email'].unique().tolist()), key="user_log_filter")
        with col_ua2:
            selected_action = st.selectbox("Filter by Action:", ["All"] + sorted(df_user_logs['action'].unique().tolist()), key="action_log_filter")
        with col_ua3:
            # Default date range for the last 7 days
            default_start_date = datetime.now().date() - timedelta(days=7)
            default_end_date = datetime.now().date()
            date_range = st.date_input("Select Date Range:", value=(default_start_date, default_end_date), key="user_log_date_filter")
            
            # Ensure date_range has two elements
            if len(date_range) == 2:
                start_date, end_date = date_range[0], date_range[1]
            else: # If only one date is selected, treat it as both start and end
                start_date = end_date = date_range[0]


        filtered_df_user = df_user_logs[
            (df_user_logs['timestamp'].dt.date >= start_date) &
            (df_user_logs['timestamp'].dt.date <= end_date)
        ]

        if selected_user != "All":
            filtered_df_user = filtered_df_user[filtered_df_user['user_email'] == selected_user]
        if selected_action != "All":
            filtered_df_user = filtered_df_user[filtered_df_user['action'] == selected_action]
        
        # Display DataFrame, optionally adjust text color for dark mode
        st.dataframe(filtered_df_user, use_container_width=True, height=300)
    else:
        st.info("No user activity logs yet.")

    st.markdown("---")

    # --- System Events Log Section ---
    st.subheader("💻 System Events Log")
    system_logs = get_system_events_logs()
    df_system_logs = pd.DataFrame(system_logs)

    if not df_system_logs.empty:
        df_system_logs['timestamp'] = pd.to_datetime(df_system_logs['timestamp'])
        df_system_logs = df_system_logs.sort_values(by='timestamp', ascending=False)

        # Filters for System Events Log
        col_se1, col_se2 = st.columns([1, 3])
        with col_se1:
            selected_level = st.selectbox("Filter by Level:", ["All"] + sorted(df_system_logs['level'].unique().tolist()), key="level_log_filter")
        with col_se2:
            default_start_date_sys = datetime.now().date() - timedelta(days=7)
            default_end_date_sys = datetime.now().date()
            date_range_sys = st.date_input("Select Date Range (System):", value=(default_start_date_sys, default_end_date_sys), key="sys_log_date_filter")
            
            if len(date_range_sys) == 2:
                start_date_sys, end_date_sys = date_range_sys[0], date_range_sys[1]
            else:
                start_date_sys = end_date_sys = date_range_sys[0]


        filtered_df_system = df_system_logs[
            (df_system_logs['timestamp'].dt.date >= start_date_sys) &
            (df_system_logs['timestamp'].dt.date <= end_date_sys)
        ]

        if selected_level != "All":
            filtered_df_system = filtered_df_system[filtered_df_system['level'] == selected_level]

        st.dataframe(filtered_df_system, use_container_width=True, height=300)
    else:
        st.info("No system event logs yet.")

    st.markdown("---")

    # --- Performance Metrics Section ---
    st.subheader("📈 Performance Metrics")

    # Screening Throughput
    st.markdown("##### Resumes Screened Over Time")
    screening_data = metrics.get('total_resumes_screened', {})
    if screening_data:
        # Convert dictionary to DataFrame
        df_screening = pd.DataFrame(list(screening_data.items()), columns=['Date', 'Count'])
        df_screening['Date'] = pd.to_datetime(df_screening['Date'])
        df_screening = df_screening.sort_values(by='Date')
        
        # Plotting - ensure plot colors adapt to dark mode
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=df_screening, x='Date', y='Count', marker='o', ax=ax, color='#2196F3' if not dark_mode else '#00BCD4') # Blue for light, Cyan for dark
        
        ax.set_title("Daily Resumes Screened", color='black' if not dark_mode else 'white')
        ax.set_xlabel("Date", color='black' if not dark_mode else 'white')
        ax.set_ylabel("Count", color='black' if not dark_mode else 'white')
        ax.tick_params(axis='x', colors='black' if not dark_mode else 'white')
        ax.tick_params(axis='y', colors='black' if not dark_mode else 'white')
        ax.grid(True, linestyle='--', alpha=0.6, color='lightgray' if not dark_mode else '#444')
        
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("No screening throughput data available yet.")

    # User Productivity - Resumes Screened
    st.markdown("##### User Productivity: Resumes Screened")
    user_screening_data = metrics.get('user_resumes_screened', {})
    user_resumes_screened_summary = {}
    # Aggregate counts across all dates for each user
    for user_email, dates_data in user_screening_data.items():
        user_resumes_screened_summary[user_email] = sum(dates_data.values())

    if user_resumes_screened_summary:
        df_productivity_screened = pd.DataFrame(list(user_resumes_screened_summary.items()), columns=['User', 'Resumes Screened'])
        df_productivity_screened = df_productivity_screened.sort_values(by='Resumes Screened', ascending=False)
        st.dataframe(df_productivity_screened, use_container_width=True, hide_index=True)
    else:
        st.info("No user-specific resume screening data available yet.")

    # User Productivity - Emails Sent
    st.markdown("##### User Productivity: Emails Sent")
    user_emails_sent_data = metrics.get('user_emails_sent', {})
    user_emails_sent_summary = {}
    for user_email, dates_data in user_emails_sent_data.items():
        user_emails_sent_summary[user_email] = sum(dates_data.values())
    
    if user_emails_sent_summary:
        df_emails_sent = pd.DataFrame(list(user_emails_sent_summary.items()), columns=['User', 'Emails Sent'])
        df_emails_sent = df_emails_sent.sort_values(by='Emails Sent', ascending=False)
        st.dataframe(df_emails_sent, use_container_width=True, hide_index=True)
    else:
        st.info("No user-specific email sending data available yet.")

    # Add more charts/tables for other metrics as you implement them
    # Example: JDs Managed by User
    st.markdown("##### User Productivity: JDs Managed")
    user_jds_managed_data = metrics.get('user_jds_managed', {}) # Make sure you log this in manage_jds.py
    user_jds_managed_summary = {}
    for user_email, dates_data in user_jds_managed_data.items():
        user_jds_managed_summary[user_email] = sum(dates_data.values())
    
    if user_jds_managed_summary:
        df_jds_managed = pd.DataFrame(list(user_jds_managed_summary.items()), columns=['User', 'JDs Managed'])
        df_jds_managed = df_jds_managed.sort_values(by='JDs Managed', ascending=False)
        st.dataframe(df_jds_managed, use_container_width=True, hide_index=True)
    else:
        st.info("No user-specific JD management data available yet.")
