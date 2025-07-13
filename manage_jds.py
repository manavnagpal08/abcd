import streamlit as st
import os
import traceback # For detailed error logging

# Import logging functions
from utils.logger import log_user_action, update_metrics_summary, log_system_event

# --- JD Folder ---
jd_folder = "data"
os.makedirs(jd_folder, exist_ok=True)

# --- UI Styling ---
st.markdown("""
<style>
.manage-jd-container {
    padding: 2rem;
    background: rgba(255, 255, 255, 0.96);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    animation: fadeSlideUp 0.7s ease-in-out;
    margin-bottom: 2rem;
}
@keyframes fadeSlideUp {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
h3 {
    color: #00cec9;
    font-weight: 700;
}
.upload-box {
    background: #f9f9f9;
    padding: 1rem;
    border-radius: 10px;
    border: 1px dashed #ccc;
}
.select-box, .text-box {
    background: #fff;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

def manage_jds_page(): # Encapsulate logic in a function for better modularity
    if 'user_email' not in st.session_state:
        st.warning("Please log in to manage Job Descriptions.")
        log_user_action("unauthenticated", "JD_MANAGER_ACCESS_DENIED", {"reason": "Not logged in"})
        return

    user_email = st.session_state.user_email
    log_user_action(user_email, "JD_MANAGER_PAGE_ACCESSED")

    # --- Header ---
    st.markdown('<div class="manage-jd-container">', unsafe_allow_html=True)
    st.markdown("### 📁 Job Description Manager")

    # --- JD Upload ---
    with st.container():
        st.markdown('<div class="upload-box">', unsafe_allow_html=True)
        st.markdown("#### 📤 Upload New JD (.txt)")
        uploaded_jd = st.file_uploader("Select file", type="txt", key="upload_jd")
        if uploaded_jd:
            try:
                jd_path = os.path.join(jd_folder, uploaded_jd.name)
                with open(jd_path, "wb") as f:
                    f.write(uploaded_jd.read())
                st.success(f"✅ Uploaded: `{uploaded_jd.name}`")
                log_user_action(user_email, "JD_UPLOAD_SUCCESS", {"file_name": uploaded_jd.name, "file_size": uploaded_jd.size})
                update_metrics_summary("total_jds_uploaded", 1)
                update_metrics_summary("user_jds_uploaded", 1, user_email=user_email)
            except Exception as e:
                st.error(f"❌ Error uploading file: {e}")
                log_system_event("ERROR", "JD_UPLOAD_FAILED", {"user_email": user_email, "file_name": uploaded_jd.name, "error": str(e), "traceback": traceback.format_exc()})
        st.markdown('</div>', unsafe_allow_html=True)

    # --- JD Listing & Viewer ---
    jd_files = [f for f in os.listdir(jd_folder) if f.endswith(".txt")]

    if jd_files:
        st.markdown('<div class="select-box">', unsafe_allow_html=True)
        selected_jd = st.selectbox("📄 Select JD to view or delete", jd_files, key="selected_jd_file")
        st.markdown('</div>', unsafe_allow_html=True)

        if selected_jd:
            try:
                with open(os.path.join(jd_folder, selected_jd), "r", encoding="utf-8") as f:
                    jd_content = f.read()

                st.markdown('<div class="text-box">', unsafe_allow_html=True)
                st.markdown("#### 📜 Job Description Content")
                st.text_area("View or Copy", jd_content, height=300, key="jd_content_display", disabled=True)
                log_user_action(user_email, "JD_VIEWED", {"file_name": selected_jd})

                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🗑️ Delete `{selected_jd}`", key="delete_jd_button"):
                        try:
                            os.remove(os.path.join(jd_folder, selected_jd))
                            st.success(f"🗑️ Deleted: `{selected_jd}`")
                            log_user_action(user_email, "JD_DELETE_SUCCESS", {"file_name": selected_jd})
                            update_metrics_summary("total_jds_deleted", 1)
                            update_metrics_summary("user_jds_deleted", 1, user_email=user_email)
                            st.experimental_rerun() # Rerun to update the list of files
                        except Exception as e:
                            st.error(f"❌ Error deleting file: {e}")
                            log_system_event("ERROR", "JD_DELETE_FAILED", {"user_email": user_email, "file_name": selected_jd, "error": str(e), "traceback": traceback.format_exc()})
                with col2:
                    if st.download_button("⬇️ Download JD", data=jd_content, file_name=selected_jd, mime="text/plain", key="download_jd_button"):
                        log_user_action(user_email, "JD_DOWNLOAD_SUCCESS", {"file_name": selected_jd})

                st.markdown('</div>', unsafe_allow_html=True)
            except FileNotFoundError:
                st.error(f"File not found: `{selected_jd}`. It might have been deleted.")
                log_system_event("WARNING", "JD_FILE_NOT_FOUND", {"user_email": user_email, "file_name": selected_jd, "action": "view_or_download"})
            except Exception as e:
                st.error(f"An error occurred while accessing the JD content: {e}")
                log_system_event("ERROR", "JD_CONTENT_ACCESS_FAILED", {"user_email": user_email, "file_name": selected_jd, "error": str(e), "traceback": traceback.format_exc()})
    else:
        st.warning("📂 No JD files uploaded yet.")
        log_system_event("INFO", "JD_MANAGER_NO_FILES_DISPLAYED", {"user_email": user_email})

    st.markdown('</div>', unsafe_allow_html=True)

# This ensures the function is called when manage_jds.py is executed directly (for testing)
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Job Description Manager")
    st.title("Job Description Manager (Standalone Test)")

    # Mock user session state for standalone testing
    if "user_email" not in st.session_state:
        st.session_state.user_email = "test_jd_manager_user@example.com"
        st.info("Running in standalone mode. Mocking user: test_jd_manager_user@example.com")

    # Call the page function
    manage_jds_page()
