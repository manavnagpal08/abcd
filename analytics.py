import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import plotly.express as px
import statsmodels.api as sm # Added this import for OLS trendline

# Import logging functions
from utils.logger import log_user_action, update_metrics_summary, log_system_event

# --- Function to encapsulate the Analytics Dashboard logic ---
def analytics_dashboard_page():
    # Log that the analytics dashboard page has been accessed
    log_user_action(st.session_state.user_email, "ANALYTICS_PAGE_ACCESSED")

    # --- Page Styling ---
    st.markdown("""
    <style>
    .analytics-box {
        padding: 2rem;
        background: rgba(255, 255, 255, 0.96);
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        animation: fadeInSlide 0.7s ease-in-out;
        margin-bottom: 2rem;
    }
    @keyframes fadeInSlide {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    h3 {
        color: #00cec9;
        font-weight: 700;
    }
    .stMetric {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="analytics-box">', unsafe_allow_html=True)
    st.markdown("## 📊 Screening Analytics Dashboard")

    # --- Load Data ---
    @st.cache_data(show_spinner=False)
    def load_screening_data():
        """Loads screening results only from session state."""
        if 'screening_results' in st.session_state and not st.session_state['screening_results'].empty:
            try:
                df_loaded = pd.DataFrame(st.session_state['screening_results'])
                st.info("✅ Loaded screening results from current session.")
                log_system_event("INFO", "ANALYTICS_DATA_LOAD_SUCCESS", {"source": "session_state", "rows": len(df_loaded)})
                return df_loaded
            except Exception as e:
                st.error(f"Error loading results from session state: {e}")
                log_system_event("ERROR", "ANALYTICS_DATA_LOAD_FAILED", {"source": "session_state", "error": str(e)})
                return pd.DataFrame() # Return empty DataFrame on error
        else:
            st.warning("⚠️ No screening data found in current session. Please run the screener first.")
            log_system_event("INFO", "ANALYTICS_DATA_NOT_FOUND", {"reason": "screening_results empty or not in session"})
            return pd.DataFrame() # Return empty DataFrame if no session data found

    df = load_screening_data()

    # Check if DataFrame is still empty after loading attempts
    if df.empty:
        st.info("No data available for analytics. Please screen some resumes first.")
        st.stop()

    # --- Essential Column Check ---
    # Adjusted column names to match the output from screener.py based on your previous code
    essential_core_columns = ['Similarity Score', 'Candidate Name', 'Predicted Status'] 
    
    # Check for 'Years Experience' if you truly expect it to be extracted, otherwise adjust or remove
    # Assuming 'Years Experience' is not directly extracted and needs to be added or derived if required for analytics
    # For now, I'm keeping the check but flagging it as something to verify.
    # If 'Years Experience' is not extracted by screener.py, this will cause an error.
    # You might need to add a function in screener.py to extract this or mock it for analytics.
    # For a robust solution, ensure screener.py extracts 'Years Experience'
    # or remove it from essential_core_columns if it's not a direct output.
    if 'Years Experience' not in df.columns:
        # Mocking 'Years Experience' for demonstration if it's not present for now
        # In a real app, you'd extract this during screening or handle its absence
        st.warning(" 'Years Experience' column not found in screening results. Generating dummy data for demonstration.")
        df['Years Experience'] = np.random.randint(1, 20, size=len(df)) # Dummy data
        log_system_event("WARNING", "MISSING_YEARS_EXPERIENCE_COLUMN", {"action": "dummy_data_generated"})


    missing_essential_columns = [col for col in essential_core_columns if col not in df.columns]

    if missing_essential_columns:
        st.error(f"Error: The loaded data is missing essential core columns: **{', '.join(missing_essential_columns)}**."
                 " Please ensure your screening process generates at least these required data fields.")
        st.stop()
    
    # Ensure 'Similarity Score' is numeric for filtering and plotting
    df['Similarity Score'] = pd.to_numeric(df['Similarity Score'], errors='coerce')
    df.dropna(subset=['Similarity Score'], inplace=True) # Remove rows where conversion failed
    df['Score (%)'] = df['Similarity Score'] * 100 # Convert to percentage for display

    # --- Filters Section ---
    st.markdown("### 🔍 Filter Results")
    filter_cols = st.columns(3)

    with filter_cols[0]:
        min_score, max_score = float(df['Score (%)'].min()), float(df['Score (%)'].max())
        score_range = st.slider(
            "Filter by Score (%)",
            min_value=min_score,
            max_value=max_score,
            value=(min_score, max_score),
            step=1.0,
            key="score_filter"
        )

    with filter_cols[1]:
        min_exp, max_exp = float(df['Years Experience'].min()), float(df['Years Experience'].max())
        exp_range = st.slider(
            "Filter by Years Experience",
            min_value=min_exp,
            max_value=max_exp,
            value=(min_exp, max_exp),
            step=0.5,
            key="exp_filter"
        )

    with filter_cols[2]:
        shortlist_threshold = st.slider(
            "Set Shortlisting Cutoff Score (%)",
            min_value=0,
            max_value=100,
            value=80, # Default value for this analytics-specific slider
            step=1,
            key="shortlist_filter"
        )

    # Apply filters
    initial_filtered_rows = len(df)
    filtered_df = df[
        (df['Score (%)'] >= score_range[0]) & (df['Score (%)'] <= score_range[1]) &
        (df['Years Experience'] >= exp_range[0]) & (df['Years Experience'] <= exp_range[1])
    ].copy()

    log_user_action(st.session_state.user_email, "ANALYTICS_FILTERS_APPLIED", {
        "score_range": score_range,
        "experience_range": exp_range,
        "shortlist_threshold": shortlist_threshold,
        "initial_rows": initial_filtered_rows,
        "filtered_rows": len(filtered_df)
    })

    if filtered_df.empty:
        st.warning("No data matches the selected filters. Please adjust your criteria.")
        st.stop()

    # Add Shortlisted/Not Shortlisted column to filtered_df for plotting
    filtered_df['Shortlisted'] = filtered_df['Score (%)'].apply(lambda x: f"Yes (Score >= {shortlist_threshold}%)" if x >= shortlist_threshold else "No")

    # --- Metrics ---
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg. Score", f"{filtered_df['Score (%)'].mean():.2f}%")
    col2.metric("Avg. Experience", f"{filtered_df['Years Experience'].mean():.1f} yrs")
    col3.metric("Total Candidates", f"{len(filtered_df)}")
    shortlisted_count_filtered = (filtered_df['Score (%)'] >= shortlist_threshold).sum()
    col4.metric("Shortlisted", f"{shortlisted_count_filtered}")

    st.divider()

    # --- Detailed Candidate Table ---
    st.markdown("### 📋 Filtered Candidates List")
    display_cols_for_table = ['Resume Name', 'Candidate Name', 'Score (%)', 'Years Experience', 'Shortlisted']

    # Adjusting column names to match screener.py output
    if 'Matched Skills' in filtered_df.columns:
        display_cols_for_table.append('Matched Skills')
    if 'Missing Skills' in filtered_df.columns:
        display_cols_for_table.append('Missing Skills')
    if 'Predicted Status' in filtered_df.columns: # Assuming AI Suggestion might map to Predicted Status
        display_cols_for_table.append('Predicted Status')

    st.dataframe(
        filtered_df[display_cols_for_table].sort_values(by="Score (%)", ascending=False),
        use_container_width=True
    )

    # --- Download Filtered Data ---
    @st.cache_data
    def convert_df_to_csv(df_to_convert):
        return df_to_convert.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(filtered_df)
    if st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_screening_results.csv",
        mime="text/csv",
        help="Download the data currently displayed in the table above."
    ):
        log_user_action(st.session_state.user_email, "ANALYTICS_DATA_DOWNLOADED", {"rows_downloaded": len(filtered_df)})

    st.divider()

    # --- Visualizations ---
    st.markdown("### 📊 Visualizations")
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Score Distribution", "Experience Distribution", "Shortlist Breakdown", "Score vs. Experience", "Skill Clouds"])

    with tab1:
        st.markdown("#### Score Distribution")
        try:
            fig_hist, ax_hist = plt.subplots(figsize=(10, 5))
            sns.histplot(filtered_df['Score (%)'], bins=10, kde=True, color="#00cec9", ax=ax_hist)
            ax_hist.set_xlabel("Score (%)")
            ax_hist.set_ylabel("Number of Candidates")
            st.pyplot(fig_hist)
            plt.close(fig_hist) # Close the figure to free up memory
        except Exception as e:
            st.error("Error generating Score Distribution chart.")
            log_system_event("ERROR", "PLOT_GENERATION_FAILED", {"chart": "Score Distribution", "error": str(e)})

    with tab2:
        st.markdown("#### Experience Distribution")
        try:
            fig_exp, ax_exp = plt.subplots(figsize=(10, 5))
            sns.histplot(filtered_df['Years Experience'], bins=5, kde=True, color="#fab1a0", ax=ax_exp)
            ax_exp.set_xlabel("Years of Experience")
            ax_exp.set_ylabel("Number of Candidates")
            st.pyplot(fig_exp)
            plt.close(fig_exp) # Close the figure to free up memory
        except Exception as e:
            st.error("Error generating Experience Distribution chart.")
            log_system_event("ERROR", "PLOT_GENERATION_FAILED", {"chart": "Experience Distribution", "error": str(e)})

    with tab3:
        st.markdown("#### Shortlist Breakdown")
        try:
            shortlist_counts = filtered_df['Shortlisted'].value_counts()
            if not shortlist_counts.empty:
                fig_pie = px.pie(
                    names=shortlist_counts.index,
                    values=shortlist_counts.values,
                    title=f"Candidates Shortlisted vs. Not Shortlisted (Cutoff: {shortlist_threshold}%)",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("Not enough data to generate Shortlist Breakdown.")
        except Exception as e:
            st.error("Error generating Shortlist Breakdown chart.")
            log_system_event("ERROR", "PLOT_GENERATION_FAILED", {"chart": "Shortlist Breakdown", "error": str(e)})

    with tab4:
        st.markdown("#### Score vs. Years Experience")
        try:
            fig_scatter = px.scatter(
                filtered_df,
                x="Years Experience",
                y="Score (%)",
                hover_name="Candidate Name",
                color="Shortlisted",
                title="Candidate Score vs. Years Experience",
                labels={"Years Experience": "Years of Experience", "Score (%)": "Matching Score (%)"},
                trendline="ols",
                color_discrete_map={f"Yes (Score >= {shortlist_threshold}%)": "green", "No": "red"}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        except Exception as e:
            st.error("Error generating Score vs. Years Experience chart.")
            log_system_event("ERROR", "PLOT_GENERATION_FAILED", {"chart": "Score vs. Experience", "error": str(e)})


    with tab5:
        col_wc1, col_wc2 = st.columns(2)
        with col_wc1:
            st.markdown("#### ☁️ Common Skills WordCloud")
            try:
                # Assuming 'Matched Skills' is the column that contains comma-separated skills
                if 'Matched Skills' in filtered_df.columns and not filtered_df['Matched Skills'].empty:
                    all_keywords = [
                        kw.strip() for kws in filtered_df['Matched Skills'].dropna()
                        for kw in str(kws).split(',') if kw.strip()
                    ]
                    if all_keywords:
                        wc = WordCloud(width=800, height=400, background_color="white").generate(" ".join(all_keywords))
                        fig_wc, ax_wc = plt.subplots(figsize=(10, 4))
                        ax_wc.imshow(wc, interpolation='bilinear')
                        ax_wc.axis('off')
                        st.pyplot(fig_wc)
                        plt.close(fig_wc) # Close the figure
                    else:
                        st.info("No common skills to display in the WordCloud for filtered data.")
                else:
                    st.info("No 'Matched Skills' data available or column not found for WordCloud.")
            except Exception as e:
                st.error("Error generating Common Skills WordCloud.")
                log_system_event("ERROR", "PLOT_GENERATION_FAILED", {"chart": "Common Skills WordCloud", "error": str(e)})
            
        with col_wc2:
            st.markdown("#### ❌ Top Missing Skills")
            try:
                if 'Missing Skills' in filtered_df.columns and not filtered_df['Missing Skills'].empty:
                    all_missing = pd.Series([
                        s.strip() for row in filtered_df['Missing Skills'].dropna()
                        for s in str(row).split(',') if s.strip()
                    ])
                    if not all_missing.empty:
                        sns.set_style("whitegrid") # Apply style before creating figure
                        fig_ms, ax_ms = plt.subplots(figsize=(8, 4))
                        top_missing = all_missing.value_counts().head(10)
                        sns.barplot(x=top_missing.values, y=top_missing.index, ax=ax_ms, palette="coolwarm")
                        ax_ms.set_xlabel("Count")
                        ax_ms.set_ylabel("Missing Skill")
                        st.pyplot(fig_ms)
                        plt.close(fig_ms) # Close the figure
                    else:
                        st.info("No top missing skills to display for filtered data.")
                else:
                    st.info("No 'Missing Skills' data available or column not found.")
            except Exception as e:
                st.error("Error generating Top Missing Skills chart.")
                log_system_event("ERROR", "PLOT_GENERATION_FAILED", {"chart": "Top Missing Skills", "error": str(e)})

    st.markdown("</div>", unsafe_allow_html=True)

# This block is for testing the analytics page in isolation if needed
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Resume Screening Analytics")
    st.title("Analytics Dashboard (Standalone Test)")

    # Mock user session state and screening_results for standalone testing
    if "user_email" not in st.session_state:
        st.session_state.user_email = "test_analytics_user@example.com"
        st.info("Running in standalone mode. Mocking user: test_analytics_user@example.com")
    
    if "screening_results" not in st.session_state or st.session_state["screening_results"].empty:
        st.info("Generating mock screening results for standalone analytics test.")
        # Create a dummy DataFrame to simulate screening_results
        mock_data = {
            'Resume Name': [f'resume_{i}.pdf' for i in range(10)],
            'Candidate Name': [f'Candidate {i}' for i in range(10)],
            'Email': [f'candidate{i}@example.com' for i in range(10)],
            'Phone': [f'555-123-000{i}' for i in range(10)],
            'Similarity Score': np.random.uniform(0.3, 0.95, 10),
            'Predicted Status': np.random.choice(['Shortlisted', 'Interview', 'Rejected'], 10),
            'Match Level': np.random.choice(['High', 'Medium', 'Low'], 10),
            'Matched Skills': [', '.join(np.random.choice(['Python', 'SQL', 'AWS', 'Agile', 'Java'], np.random.randint(1, 4), replace=False)) for _ in range(10)],
            'Missing Skills': [', '.join(np.random.choice(['Docker', 'Kubernetes', 'MLOps', 'React', 'Leadership'], np.random.randint(0, 3), replace=False)) for _ in range(10)],
            'Resume Text': ['This is some dummy resume text.' for _ in range(10)],
            'WordCloudText': ['python sql aws' for _ in range(10)], # Dummy for word cloud
            'Years Experience': np.random.randint(1, 15, 10) # Dummy years experience
        }
        st.session_state.screening_results = pd.DataFrame(mock_data)

    analytics_dashboard_page()
