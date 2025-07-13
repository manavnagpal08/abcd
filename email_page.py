# email_page.py

import streamlit as st
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import json # Assuming you might use this for templates or config
import os # For checking if files exist
import traceback # For detailed error logging

# Import logging functions
from utils.logger import log_user_action, update_metrics_summary, log_system_event

def email_candidates_page(): # Renamed function to match main.py's import style
    if 'user_email' not in st.session_state:
        st.warning("Please log in to send emails.")
        log_user_action("unauthenticated", "EMAIL_PAGE_ACCESS_DENIED", {"reason": "Not logged in"})
        return

    user_email = st.session_state.user_email
    log_user_action(user_email, "EMAIL_PAGE_ACCESSED")

    st.markdown("## 📤 Email Candidates")
    st.info("Prepare and send emails to shortlisted candidates based on screening results.")

    # Check if screening results are available in session state
    if 'screening_results' not in st.session_state or st.session_state['screening_results'].empty:
        st.warning("No screening results found. Please run the '🧠 Resume Screener' first to get candidates to email.")
        log_system_event("INFO", "EMAIL_PAGE_NO_SCREENING_RESULTS", {"user_email": user_email})
        return # Exit the function if no results

    try:
        df_results = pd.DataFrame(st.session_state['screening_results'])

        # Ensure required columns exist before proceeding
        # Adjusted 'AI Suggestion' to 'Predicted Status' based on screener.py output
        required_columns = ['Candidate Name', 'Email', 'Score (%)', 'Years Experience', 'Predicted Status']
        missing_columns = [col for col in required_columns if col not in df_results.columns]

        if missing_columns:
            st.error(f"Missing essential data columns in screening results: {', '.join(missing_columns)}. "
                     "Please ensure the 'Resume Screener' generated these columns.")
            # Display available columns for debugging
            st.write(f"Available columns: {list(df_results.columns)}")
            log_system_event("ERROR", "EMAIL_PAGE_MISSING_COLUMNS", {"user_email": user_email, "missing_columns": missing_columns})
            return

        # Filtering for shortlisted candidates based on criteria (can be adjusted)
        # It's better to get the cutoff values from session_state if they are stored there by screener.py
        cutoff_score = st.session_state.get('screening_cutoff_score', 75)
        min_exp_required = st.session_state.get('screening_min_experience', 2)

        shortlisted_candidates = df_results[
            (df_results["Score (%)"] >= cutoff_score) &
            (df_results["Years Experience"] >= min_exp_required)
        ].copy() # Use .copy() to avoid SettingWithCopyWarning

        if shortlisted_candidates.empty:
            st.warning(f"No candidates meet the current shortlisting criteria (Score >= {cutoff_score}%, Experience >= {min_exp_required} years). Adjust criteria in Screener or review results.")
            st.dataframe(df_results[['Candidate Name', 'Score (%)', 'Years Experience', 'Predicted Status']], use_container_width=True)
            log_system_event("INFO", "EMAIL_PAGE_NO_SHORTLISTED_CANDIDATES", {"user_email": user_email, "cutoff_score": cutoff_score, "min_exp": min_exp_required})
            return

        st.success(f"Found {len(shortlisted_candidates)} shortlisted candidates.")
        st.dataframe(shortlisted_candidates[['Candidate Name', 'Email', 'Score (%)', 'Predicted Status']], use_container_width=True)

        st.markdown("### 📧 Email Configuration")
        sender_email = st.text_input("Your Email (Sender)", key="sender_email")
        sender_password = st.text_input("Your Email Password (App Password)", type="password", key="sender_password")
        smtp_server = st.text_input("SMTP Server", "smtp.gmail.com", key="smtp_server")
        smtp_port = st.number_input("SMTP Port", 587, key="smtp_port")

        st.markdown("### ✍️ Email Content")
        email_subject = st.text_input("Email Subject", "Job Application Update - Your Application to [Job Title]")
        default_body = """
        Dear {candidate_name},

        Thank you for your application for the position of [Job Title] at [Company Name].

        We have reviewed your resume and would like to provide an update. Based on our initial assessment, your profile showed a score of {score_percent:.1f}% and {years_experience:.1f} years of experience.

        Our AI's suggestion for your profile: {predicted_status}

        We will be in touch shortly regarding the next steps in our hiring process.

        Best regards,

        The [Company Name] Hiring Team
        """
        email_body = st.text_area("Email Body (use {candidate_name}, {score_percent}, {years_experience}, {predicted_status})", default_body, height=300)

        if st.button("🚀 Send Emails to Shortlisted Candidates"):
            if not sender_email or not sender_password:
                st.error("Please enter your sender email and password.")
                log_user_action(user_email, "EMAIL_SEND_INITIATED_FAILED", {"reason": "Missing sender credentials"})
                return

            log_user_action(user_email, "EMAIL_SEND_INITIATED_SUCCESS", {"num_candidates": len(shortlisted_candidates), "subject_preview": email_subject[:50]})

            total_sent = 0
            total_failed = 0
            with st.spinner("Sending emails..."):
                try:
                    with smtplib.SMTP(smtp_server, smtp_port) as server:
                        server.starttls() # Secure the connection
                        server.login(sender_email, sender_password)

                        for index, row in shortlisted_candidates.iterrows():
                            candidate_name = row['Candidate Name']
                            candidate_email = row['Email']
                            score_percent = row['Score (%)']
                            years_experience = row['Years Experience']
                            # Changed from ai_suggestion to predicted_status
                            predicted_status = row['Predicted Status'] 

                            # Format the email body with actual candidate data
                            formatted_body = email_body.format(
                                candidate_name=candidate_name,
                                score_percent=score_percent,
                                years_experience=years_experience,
                                predicted_status=predicted_status # Use predicted_status
                            )

                            msg = MIMEMultipart()
                            msg['From'] = sender_email
                            msg['To'] = candidate_email
                            msg['Subject'] = email_subject
                            msg.attach(MIMEText(formatted_body, 'plain'))

                            try:
                                server.send_message(msg)
                                st.success(f"Email sent to {candidate_name} ({candidate_email})")
                                total_sent += 1
                                log_user_action(user_email, "EMAIL_SENT_SUCCESS", {
                                    "recipient_name": candidate_name,
                                    "recipient_email": candidate_email,
                                    "subject": email_subject,
                                    "status_tag": predicted_status
                                })
                                update_metrics_summary("total_emails_sent", 1)
                                update_metrics_summary("user_emails_sent", 1, user_email=user_email)
                            except Exception as e:
                                st.error(f"Failed to send email to {candidate_name} ({candidate_email}): {e}")
                                total_failed += 1
                                log_system_event("ERROR", "EMAIL_SEND_INDIVIDUAL_FAILED", {
                                    "user_email": user_email,
                                    "recipient_name": candidate_name,
                                    "recipient_email": candidate_email,
                                    "subject": email_subject,
                                    "error": str(e),
                                    "traceback": traceback.format_exc()
                                })

                except smtplib.SMTPAuthenticationError:
                    st.error("Email sending failed: Invalid sender email or app password. For Gmail, you might need to use an App Password.")
                    log_system_event("ERROR", "SMTP_AUTH_FAILED", {"user_email": user_email, "smtp_server": smtp_server, "error": "Authentication failed"})
                except smtplib.SMTPConnectError:
                    st.error(f"Email sending failed: Could not connect to SMTP server {smtp_server}:{smtp_port}. Check server address and port.")
                    log_system_event("ERROR", "SMTP_CONNECT_FAILED", {"user_email": user_email, "smtp_server": smtp_server, "smtp_port": smtp_port, "error": "Connection failed"})
                except Exception as e:
                    st.error(f"An unexpected error occurred during email sending setup: {e}")
                    log_system_event("CRITICAL", "EMAIL_SEND_SETUP_ERROR", {"user_email": user_email, "error": str(e), "traceback": traceback.format_exc()})
            
            if total_sent > 0:
                st.success(f"Successfully sent {total_sent} emails!")
            if total_failed > 0:
                st.warning(f"Failed to send {total_failed} emails. Check system logs for details.")

    except Exception as e:
        st.error(f"An error occurred while preparing candidate data: {e}")
        log_system_event("ERROR", "EMAIL_PAGE_DATA_PREP_ERROR", {"user_email": user_email, "error": str(e), "traceback": traceback.format_exc()})


# This ensures the function is called when email_page.py is executed (via exec() or direct import)
if __name__ == "__main__":
    # Mock user session state for standalone testing
    if "user_email" not in st.session_state:
        st.session_state.user_email = "test_email_user@example.com"
        st.info("Running in standalone mode. Mocking user: test_email_user@example.com")
    
    # Mock screening results for standalone testing
    if "screening_results" not in st.session_state or st.session_state["screening_results"].empty:
        st.info("Generating mock screening results for standalone email page test.")
        mock_data = {
            'Candidate Name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown'],
            'Email': ['alice@example.com', 'bob@example.com', 'charlie@example.com'],
            'Score (%)': [85, 92, 60],
            'Years Experience': [5, 8, 1],
            'Predicted Status': ['Shortlisted', 'Shortlisted', 'Rejected'] # Changed from AI Suggestion
        }
        st.session_state.screening_results = pd.DataFrame(mock_data)
        st.session_state['screening_cutoff_score'] = 75
        st.session_state['screening_min_experience'] = 2

    email_candidates_page()
