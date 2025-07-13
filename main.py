import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import json

# Import the page functions from their respective files
from login import (
    login_section, load_users, admin_registration_section,
    admin_password_reset_section, admin_disable_enable_user_section,
    is_current_user_admin
)
# Assuming these files exist in your project structure (you'll need to create them)
# from email_sender import send_email_to_candidate
# from screener import resume_screener_page
# from analytics import analytics_dashboard_page


# --- Page Config ---
st.set_page_config(page_title="ScreenerPro – AI Hiring Dashboard", layout="wide", page_icon="🧠")


# --- Dark Mode Toggle ---
# Note: The dark mode toggle will still exist, but without the CSS,
# its visual effect on other elements might be limited to Streamlit's defaults.
dark_mode = st.sidebar.toggle("🌙 Dark Mode", key="dark_mode_main")

# --- Global Fonts & UI Styling ---
# This CSS block is now implemented as requested.
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.main .block-container {
    padding: 2rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 12px 30px rgba(0,0,0,0.1);
    animation: fadeIn 0.8s ease-in-out;
}
@keyframes fadeIn {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
.dashboard-card {
    padding: 2rem;
    text-align: center;
    font-weight: 600;
    border-radius: 16px;
    background: linear-gradient(145deg, #f1f2f6, #ffffff);
    border: 1px solid #e0e0e0;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.3s ease;
    cursor: pointer;
}
.dashboard-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.1);
    background: linear-gradient(145deg, #e0f7fa, #f1f1f1);
}
.dashboard-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #222;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #00cec9;
    display: inline-block;
    margin-bottom: 2rem;
    animation: slideInLeft 0.8s ease-out;
}
@keyframes slideInLeft {
    0% { transform: translateX(-40px); opacity: 0; }
    100% { transform: translateX(0); opacity: 1; }
}
/* New CSS for custom buttons to look like cards */
.custom-dashboard-button {
    width: 100%;
    height: 100%; /* Ensure it takes full height of its column */
    padding: 2rem;
    text-align: center;
    font-weight: 600;
    border-radius: 16px;
    background: linear-gradient(145deg, #f1f2f6, #ffffff);
    border: 1px solid #e0e0e0;
    box-shadow: 0 6px 18px rgba(0,0,0,0.05);
    transition: transform 0.2s ease, box-shadow 0.3s ease;
    cursor: pointer;
    display: flex;
    flex-direction: column; /* Stack icon and text vertically */
    justify-content: center;
    align-items: center;
    color: #333; /* Ensure text color is visible */
    min-height: 120px; /* Ensure a consistent height for the buttons */
}
.custom-dashboard-button:hover {
    transform: translateY(-6px);
    box-shadow: 0 10px 24px rgba(0,0,0,0.1);
    background: linear-gradient(145deg, #e0f7fa, #f1f1f1);
}
.custom-dashboard-button span { /* For the icon */
    font-size: 1.5rem;
    margin-bottom: 0.5rem;
}
.custom-dashboard-button div { /* For the text */
    font-size: 1rem;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# Set Matplotlib style for dark mode if active
if dark_mode:
    plt.style.use('dark_background')
else:
    plt.style.use('default')


# --- Branding ---
try:
    st.image("logo.png", width=300)
except FileNotFoundError:
    st.warning("Logo file 'logo.png' not found. Please ensure it's in the correct directory.")
st.title("🧠 ScreenerPro – AI Hiring Assistant")

# --- Auth ---
if not login_section():
    st.stop()

# Determine if the logged-in user is an admin
is_admin = is_current_user_admin()

# --- Navigation Control ---
navigation_options = [
    "🏠 Dashboard", "🧠 Resume Screener", "📁 Manage JDs", "📊 Screening Analytics",
    "📤 Email Candidates", "🔍 Search Resumes", "📝 Candidate Notes"
]
if is_admin: # Only add Admin Tools if the user is an admin
    navigation_options.append("⚙️ Admin Tools")
navigation_options.append("🚪 Logout") # Always add Logout last

default_tab = st.session_state.get("tab_override", "🏠 Dashboard")
if default_tab not in navigation_options: # Handle cases where default_tab might be Admin Tools for non-admins
    default_tab = "🏠 Dashboard"

tab = st.sidebar.radio("📍 Navigate", navigation_options, index=navigation_options.index(default_tab))

if "tab_override" in st.session_state:
    del st.session_state.tab_override

# ======================
# 🏠 Dashboard Section
# ======================
if tab == "🏠 Dashboard":
    # The div for "dashboard-header" will now have custom styling
    st.markdown('<div class="dashboard-header">📊 Overview Dashboard</div>', unsafe_allow_html=True) 

    # Initialize metrics
    resume_count = 0
    # Create the 'data' directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    jd_count = len([f for f in os.listdir("data") if f.endswith(".txt")])
    shortlisted = 0
    avg_score = 0.0
    df_results = pd.DataFrame()

    # Load results from session state
    if 'screening_results' in st.session_state and st.session_state['screening_results']:
        try:
            df_results = pd.DataFrame(st.session_state['screening_results'])
            resume_count = df_results["File Name"].nunique()
            
            cutoff_score = st.session_state.get('screening_cutoff_score', 75)
            min_exp_required = st.session_state.get('screening_min_experience', 2)

            shortlisted_df = df_results[
                (df_results["Score (%)"] >= cutoff_score) &
                (df_results["Years Experience"] >= min_exp_required)
            ].copy()
            shortlisted = shortlisted_df.shape[0]
            avg_score = df_results["Score (%)"].mean()
        except Exception as e:
            st.error(f"Error processing screening results from session state: {e}")
            df_results = pd.DataFrame()
            shortlisted_df = pd.DataFrame()
    else:
        st.info("No screening results available in this session yet. Please run the Resume Screener.")
        shortlisted_df = pd.DataFrame()

    col1, col2, col3 = st.columns(3)

    with col1:
        # The div for "dashboard-card" will now have custom styling
        st.markdown(f"""<div class="dashboard-card">📂 <br><b>{resume_count}</b><br>Resumes Screened</div>""", unsafe_allow_html=True) 
        if resume_count > 0:
            with st.expander(f"View {resume_count} Screened Names"):
                for idx, row in df_results.iterrows():
                    st.markdown(f"- **{row['Candidate Name']}** (Score: {row['Score (%)']:.1f}%)")
        elif 'screening_results' in st.session_state and st.session_state['screening_results']:
            st.info("No resumes have been screened yet.")
        else:
            st.info("Run the screener to see screened resumes.")

    with col2:
        # The div for "dashboard-card" will now have custom styling
        st.markdown(f"""<div class="dashboard-card">📝 <br><b>{jd_count}</b><br>Job Descriptions</div>""", unsafe_allow_html=True) 

    with col3:
        # The div for "dashboard-card" will now have custom styling
        st.markdown(f"""<div class="dashboard-card">✅ <br><b>{shortlisted}</b><br>Shortlisted Candidates</div>""", unsafe_allow_html=True) 
        if shortlisted > 0:
            with st.expander(f"View {shortlisted} Shortlisted Names"):
                for idx, row in shortlisted_df.iterrows():
                    st.markdown(f"- **{row['Candidate Name']}** (Score: {row['Score (%)']:.1f}%, Exp: {row['Years Experience']:.1f} yrs)")
        elif 'screening_results' in st.session_state and st.session_state['screening_results']:
            st.info("No candidates met the current shortlisting criteria.")
        else:
            st.info("Run the screener to see shortlisted candidates.")

    col4, col5, col6 = st.columns(3)
    # The div for "dashboard-card" will now have custom styling
    col4.markdown(f"""<div class="dashboard-card">📈 <br><b>{avg_score:.1f}%</b><br>Avg Score</div>""", unsafe_allow_html=True) 

    with col5:
        # The div for "custom-dashboard-button" will now have custom styling
        st.markdown("""
        <div class="custom-dashboard-button" onclick="window.parent.postMessage({streamlit: {type: 'setSessionState', args: ['tab_override', '🧠 Resume Screener']}}, '*');">
            <span>🧠</span>
            <div>Resume Screener</div>
        </div>
        """, unsafe_allow_html=True) 
        if st.button("🧠 Resume Screener", key="dashboard_screener_button"):
            st.session_state.tab_override = '🧠 Resume Screener'
            st.rerun()

    with col6:
        # The div for "custom-dashboard-button" will now have custom styling
        st.markdown("""
        <div class="custom-dashboard-button" onclick="window.parent.postMessage({streamlit: {type: 'setSessionState', args: ['tab_override', '📤 Email Candidates']}}, '*');">
            <span>📤</span>
            <div>Email Candidates</div>
        </div>
        """, unsafe_allow_html=True) 
        if st.button("📤 Email Candidates", key="dashboard_email_button"):
            st.session_state.tab_override = '📤 Email Candidates'
            st.rerun()


    # Optional: Dashboard Insights
    if not df_results.empty:
        try:
            df_results['Tag'] = df_results.apply(lambda row:
                "👑 Exceptional Match" if row['Score (%)'] >= 90 and row['Years Experience'] >= 5 and row['Semantic Similarity'] >= 0.85 else (
                "🔥 Strong Candidate" if row['Score (%)'] >= 80 and row['Years Experience'] >= 3 and row['Semantic Similarity'] >= 0.7 else (
                "✨ Promising Fit" if row['Score (%)'] >= 60 and row['Years Experience'] >= 1 else (
                "⚠️ Needs Review" if row['Score (%)'] >= 40 else
                "❌ Limited Match"))), axis=1)

            st.markdown("### 📊 Dashboard Insights")

            col_g1, col_g2 = st.columns(2)

            with col_g1:
                st.markdown("##### 🔥 Candidate Distribution")
                pie_data = df_results['Tag'].value_counts().reset_index()
                pie_data.columns = ['Tag', 'Count']
                fig_pie, ax1 = plt.subplots(figsize=(4.5, 4.5))
                # Colors will revert to default Matplotlib/Seaborn unless specified manually without CSS
                if dark_mode:
                    colors = plt.cm.Dark2.colors
                    text_color = 'white'
                else:
                    colors = plt.cm.Pastel1.colors
                    text_color = 'black'

                wedges, texts, autotexts = ax1.pie(pie_data['Count'], labels=pie_data['Tag'], autopct='%1.1f%%', startangle=90, colors=colors, textprops={'fontsize': 10, 'color': text_color})
                for autotext in autotexts:
                    autotext.set_color(text_color)
                ax1.axis('equal')
                st.pyplot(fig_pie)
                plt.close(fig_pie)

            with col_g2:
                st.markdown("##### 📊 Experience Distribution")
                bins = [0, 2, 5, 10, 20]
                labels = ['0-2 yrs', '3-5 yrs', '6-10 yrs', '10+ yrs']
                df_results['Experience Group'] = pd.cut(df_results['Years Experience'], bins=bins, labels=labels, right=False)
                exp_counts = df_results['Experience Group'].value_counts().sort_index()
                fig_bar, ax2 = plt.subplots(figsize=(5, 4))
                
                if dark_mode:
                    sns.barplot(x=exp_counts.index, y=exp_counts.values, palette="viridis", ax=ax2)
                else:
                    sns.barplot(x=exp_counts.index, y=exp_counts.values, palette="coolwarm", ax=ax2)
                
                # These might need manual color adjustments for dark mode if they don't pick up plt.style.use('dark_background') fully
                ax2.set_ylabel("Candidates", color='white' if dark_mode else 'black')
                ax2.set_xlabel("Experience Range", color='white' if dark_mode else 'black')
                ax2.tick_params(axis='x', labelrotation=0, colors='white' if dark_mode else 'black')
                ax2.tick_params(axis='y', colors='white' if dark_mode else 'black')
                ax2.title.set_color('white' if dark_mode else 'black')
                st.pyplot(fig_bar)
                plt.close(fig_bar)
            
            st.markdown("##### 📋 Candidate Quality Breakdown")
            tag_summary = df_results['Tag'].value_counts().reset_index()
            tag_summary.columns = ['Candidate Tag', 'Count']
            st.dataframe(tag_summary, use_container_width=True, hide_index=True)


            st.markdown("##### 🧠 Top 5 Most Common Skills")

            if 'Matched Keywords' in df_results.columns:
                all_skills = []
                for skills in df_results['Matched Keywords'].dropna():
                    all_skills.extend([s.strip().lower() for s in skills.split(",") if s.strip()])

                skill_counts = pd.Series(all_skills).value_counts().head(5)

                if not skill_counts.empty:
                    fig_skills, ax3 = plt.subplots(figsize=(5.8, 3))
                    
                    if dark_mode:
                        palette = sns.color_palette("magma", len(skill_counts))
                    else:
                        palette = sns.color_palette("cool", len(skill_counts))

                    sns.barplot(
                        x=skill_counts.values,
                        y=skill_counts.index,
                        palette=palette,
                        ax=ax3
                    )
                    ax3.set_title("Top 5 Skills", fontsize=13, fontweight='bold', color='white' if dark_mode else 'black')
                    ax3.set_xlabel("Frequency", fontsize=11, color='white' if dark_mode else 'black')
                    ax3.set_ylabel("Skill", fontsize=11, color='white' if dark_mode else 'black')
                    ax3.tick_params(labelsize=10, colors='white' if dark_mode else 'black')
                    
                    for i, v in enumerate(skill_counts.values):
                        ax3.text(v + 0.3, i, str(v), color='white' if dark_mode else 'black', va='center', fontweight='bold', fontsize=9)

                    fig_skills.tight_layout()
                    st.pyplot(fig_skills)
                    plt.close(fig_skills)
                else:
                    st.info("No skill data available in results for the Top 5 Skills chart.")

            else:
                st.info("No 'Matched Keywords' column found in results for skill analysis.")

        except Exception as e:
            st.warning(f"⚠️ Could not render insights due to data error: {e}")

# ======================
# ⚙️ Admin Tools Section
# ======================
elif tab == "⚙️ Admin Tools":
    # The div for "dashboard-header" will now have custom styling
    st.markdown('<div class="dashboard-header">⚙️ Admin Tools</div>', unsafe_allow_html=True) 
    if is_admin:
        st.write("Welcome, Administrator! Here you can manage user accounts.")
        st.markdown("---")

        admin_registration_section() # Create New User Form
        st.markdown("---")

        admin_password_reset_section() # Reset User Password Form
        st.markdown("---")

        admin_disable_enable_user_section() # Disable/Enable User Form
        st.markdown("---")

        st.subheader("👥 All Registered Users")
        st.warning("⚠️ **SECURITY WARNING:** This table displays usernames (email IDs) and **hashed passwords**. This is for **ADMINISTRATIVE DEBUGGING ONLY IN A SECURE ENVIRONMENT**. **NEVER expose this in a public or production application.**")
        try:
            users_data = load_users()
            if users_data:
                display_users = []
                for user, data in users_data.items():
                    hashed_pass = data.get("password", data) if isinstance(data, dict) else data
                    status = data.get("status", "N/A") if isinstance(data, dict) else "N/A"
                    display_users.append([user, hashed_pass, status])
                st.dataframe(pd.DataFrame(display_users, columns=["Email/Username", "Hashed Password (DO NOT EXPOSE)", "Status"]), use_container_width=True)
            else:
                st.info("No users registered yet.")
        except Exception as e:
            st.error(f"Error loading user data: {e}")
    else:
        st.error("🔒 Access Denied: You must be an administrator to view this page.")

# ======================
# Page Routing via function calls (remaining pages)
# ======================
# Placeholder for screener.py, email_sender.py, analytics.py, manage_jds.py, search.py, notes.py
# You need to ensure these files exist and define the respective page functions.
# For demonstration, these will print a message if the files are not available.

elif tab == "🧠 Resume Screener":
    try:
        from screener import resume_screener_page
        resume_screener_page()
    except ImportError:
        st.info("`screener.py` not found or function not defined. Please create it.")
    except Exception as e:
        st.error(f"Error loading Resume Screener: {e}")


elif tab == "📁 Manage JDs":
    try:
        # Assuming manage_jds.py contains its Streamlit code directly or in a function
        with open("manage_jds.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.info("`manage_jds.py` not found. Please ensure the file exists in the same directory.")
    except Exception as e:
        st.error(f"Error loading Manage JDs: {e}")


elif tab == "📊 Screening Analytics":
    try:
        from analytics import analytics_dashboard_page
        analytics_dashboard_page()
    except ImportError:
        st.info("`analytics.py` not found or function not defined. Please create it.")
    except Exception as e:
        st.error(f"Error loading Screening Analytics: {e}")

elif tab == "📤 Email Candidates":
    try:
        from email_sender import send_email_to_candidate
        send_email_to_candidate()
    except ImportError:
        st.info("`email_sender.py` not found or function not defined. Please create it.")
    except Exception as e:
        st.error(f"Error loading Email Candidates: {e}")


elif tab == "🔍 Search Resumes":
    try:
        # Assuming search.py contains its Streamlit code directly or in a function
        with open("search.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.info("`search.py` not found. Please ensure the file exists in the same directory.")
    except Exception as e:
        st.error(f"Error loading Search Resumes: {e}")

elif tab == "📝 Candidate Notes":
    try:
        # Assuming notes.py contains its Streamlit code directly or in a function
        with open("notes.py", encoding="utf-8") as f:
            exec(f.read())
    except FileNotFoundError:
        st.info("`notes.py` not found. Please ensure the file exists in the same directory.")
    except Exception as e:
        st.error(f"Error loading Candidate Notes: {e}")

elif tab == "🚪 Logout":
    st.session_state.authenticated = False
    st.session_state.pop('username', None)
    st.success("✅ Logged out.")
    st.rerun() # Rerun to redirect to login page
