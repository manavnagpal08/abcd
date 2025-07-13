import streamlit as st
import pdfplumber
import re
import pandas as pd
import io
import traceback # For detailed error logging

# Import logging functions
from utils.logger import log_user_action, update_metrics_summary, log_system_event

# --- Styling ---
st.markdown("""
<style>
.search-box {
    padding: 2rem;
    margin-top: 1rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.95);
    box-shadow: 0 8px 30px rgba(0,0,0,0.07);
    animation: slideFade 0.6s ease-in-out;
}
.result-box {
    background: #f7faff;
    padding: 1.2rem;
    margin-bottom: 1.2rem;
    border-radius: 14px;
    border-left: 4px solid #00cec9;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    animation: fadeInResult 0.6s ease;
}
.highlight {
    background-color: #ffeaa7;
    font-weight: 600;
    padding: 2px 6px;
    border-radius: 4px;
}
@keyframes slideFade {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInResult {
    0% { opacity: 0; transform: scale(0.98); }
    100% { opacity: 1; transform: scale(1); }
}
</style>
""", unsafe_allow_html=True)

def search_page(): # Encapsulate logic in a function for better modularity
    if 'user_email' not in st.session_state:
        st.warning("Please log in to use the Resume Search Engine.")
        log_user_action("unauthenticated", "SEARCH_PAGE_ACCESS_DENIED", {"reason": "Not logged in"})
        return

    user_email = st.session_state.user_email
    log_user_action(user_email, "SEARCH_PAGE_ACCESSED")

    # --- UI Header ---
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    st.subheader("🔍 Resume Search Engine")
    st.caption("Upload resumes and search for single or multiple keywords (e.g., `python, sql`).")

    # --- File Upload ---
    resumes = st.file_uploader("📤 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True, key="resume_search_upload")
    resume_texts = {}

    if resumes:
        st.success(f"✅ {len(resumes)} resume(s) uploaded.")
        log_user_action(user_email, "RESUMES_UPLOADED_FOR_SEARCH", {"count": len(resumes)})
        
        for resume in resumes:
            try:
                with pdfplumber.open(resume) as pdf:
                    text = ''.join(page.extract_text() or '' for page in pdf.pages)
                    resume_texts[resume.name] = text
                log_system_event("INFO", "RESUME_PARSED_SUCCESS", {"user_email": user_email, "resume_name": resume.name})
            except Exception as e:
                st.warning(f"⚠️ Error reading {resume.name}. This resume will be skipped.")
                log_system_event("ERROR", "RESUME_PARSE_FAILED", {"user_email": user_email, "resume_name": resume.name, "error": str(e), "traceback": traceback.format_exc()})

        query = st.text_input("🔎 Enter keywords (comma-separated)").strip().lower()
        download_rows = []

        if query:
            keywords = [q.strip() for q in query.split(',') if q.strip()]
            log_user_action(user_email, "RESUME_SEARCH_INITIATED", {"keywords": keywords, "num_resumes_to_search": len(resume_texts)})
            update_metrics_summary("total_searches_performed", 1)
            update_metrics_summary("user_searches_performed", 1, user_email=user_email)

            st.markdown("### 📄 Search Results")
            found = False

            for name, content in resume_texts.items():
                content_lower = content.lower()
                matched_keywords_for_resume = []
                matched_snippets = []
                for keyword in keywords:
                    if keyword in content_lower:
                        found = True
                        matched_keywords_for_resume.append(keyword)
                        
                        # Find all occurrences of the keyword to get multiple snippets
                        for match in re.finditer(re.escape(keyword), content_lower):
                            idx = match.start()
                            snippet_start = max(0, idx - 40)
                            snippet_end = min(len(content), idx + 160)
                            snippet = content[snippet_start:snippet_end]
                            
                            highlighted = re.sub(
                                f"({re.escape(keyword)})",
                                r"<span class='highlight'>\1</span>",
                                snippet,
                                flags=re.IGNORECASE
                            )
                            matched_snippets.append(highlighted)

                if matched_snippets:
                    combined_snippet = " ... ".join(matched_snippets)
                    st.markdown(f"""<div class="result-box">
                    <b>📄 {name}</b><br>{combined_snippet}...
                    </div>""", unsafe_allow_html=True)

                    download_rows.append({
                        "File Name": name,
                        "Matched Keywords": ", ".join(matched_keywords_for_resume),
                        "Snippet": ' '.join(snippet.replace("<span class='highlight'>", "").replace("</span>", "") for snippet in matched_snippets) # Clean snippet for CSV
                    })
            
            if found:
                log_user_action(user_email, "RESUME_SEARCH_RESULTS_FOUND", {"keywords": keywords, "num_results": len(download_rows)})
            else:
                st.error("❌ No matching resumes found.")
                log_user_action(user_email, "RESUME_SEARCH_NO_RESULTS", {"keywords": keywords})

            # --- Export Button ---
            if download_rows:
                df_download = pd.DataFrame(download_rows)
                csv_buffer = io.StringIO()
                df_download.to_csv(csv_buffer, index=False)
                if st.download_button("📥 Download Matched Results (CSV)", data=csv_buffer.getvalue(), file_name="matched_resumes.csv", mime="text/csv"):
                    log_user_action(user_email, "SEARCH_RESULTS_DOWNLOADED", {"keywords": keywords, "num_rows": len(download_rows)})

    else:
        st.info("📁 Please upload resume PDFs to begin searching.")

    st.markdown("</div>", unsafe_allow_html=True)

# This block is for testing the search page in isolation if needed
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Resume Search")
    st.title("Resume Search (Standalone Test)")

    # Mock user session state for standalone testing
    if "user_email" not in st.session_state:
        st.session_state.user_email = "test_search_user@example.com"
        st.info("Running in standalone mode. Mocking user: test_search_user@example.com")

    search_page()
