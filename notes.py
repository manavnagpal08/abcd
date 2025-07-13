import streamlit as st
import json
import os
import traceback # For detailed error logging

# Import logging functions
from utils.logger import log_user_action, update_metrics_summary, log_system_event

# --- Styling ---
st.markdown("""
<style>
.notes-container {
    padding: 2rem;
    border-radius: 20px;
    background: rgba(255, 255, 255, 0.9);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
    animation: fadeInSlide 0.6s ease-in-out;
}
.note-box {
    background: #f0f9ff;
    border-left: 4px solid #00cec9;
    padding: 1rem;
    margin-bottom: 1rem;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.03);
}
@keyframes fadeInSlide {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

def notes_page(): # Encapsulate logic in a function for better modularity
    if 'user_email' not in st.session_state:
        st.warning("Please log in to manage notes.")
        log_user_action("unauthenticated", "NOTES_PAGE_ACCESS_DENIED", {"reason": "Not logged in"})
        return

    user_email = st.session_state.user_email
    log_user_action(user_email, "NOTES_PAGE_ACCESSED")

    st.markdown('<div class="notes-container">', unsafe_allow_html=True)
    st.subheader("📝 Candidate Notes")

    notes_file = "notes.json"
    notes = {}
    try:
        if os.path.exists(notes_file):
            with open(notes_file, "r", encoding="utf-8") as f:
                notes = json.load(f)
            log_system_event("INFO", "NOTES_FILE_LOADED", {"user_email": user_email, "notes_count": len(notes)})
        else:
            # If file doesn't exist, create an empty one
            with open(notes_file, "w", encoding="utf-8") as f:
                json.dump({}, f)
            log_system_event("INFO", "NOTES_FILE_CREATED", {"user_email": user_email})
    except json.JSONDecodeError as e:
        st.error(f"Error reading notes file: {e}. Please check notes.json for valid JSON format.")
        log_system_event("ERROR", "NOTES_FILE_READ_ERROR", {"user_email": user_email, "error": str(e), "traceback": traceback.format_exc()})
        notes = {} # Reset notes to empty dict to prevent further errors
    except Exception as e:
        st.error(f"An unexpected error occurred while loading notes: {e}")
        log_system_event("ERROR", "NOTES_FILE_LOAD_UNEXPECTED_ERROR", {"user_email": user_email, "error": str(e), "traceback": traceback.format_exc()})
        notes = {}


    candidates = sorted(notes.keys())
    selected = st.selectbox("📄 Select Candidate", candidates, key="notes_candidate_select")

    if selected:
        st.markdown(f"#### 🗒️ Notes for {selected}")
        st.markdown('<div class="note-box">', unsafe_allow_html=True)
        text = st.text_area("Edit Note", value=notes.get(selected, ""), height=150, key=f"edit_note_{selected}")
        col1, col2 = st.columns(2)
        if col1.button("💾 Save Note", key=f"save_note_{selected}"):
            old_note = notes.get(selected, "")
            notes[selected] = text
            try:
                with open(notes_file, "w", encoding="utf-8") as f:
                    json.dump(notes, f, indent=2)
                st.success("✅ Note updated.")
                log_user_action(user_email, "NOTE_UPDATED", {
                    "candidate": selected,
                    "old_note_len": len(old_note),
                    "new_note_len": len(text)
                })
                update_metrics_summary("total_notes_updated", 1)
                update_metrics_summary("user_notes_updated", 1, user_email=user_email)
            except Exception as e:
                st.error(f"Error saving note: {e}")
                log_system_event("ERROR", "NOTE_SAVE_FAILED", {"user_email": user_email, "candidate": selected, "error": str(e), "traceback": traceback.format_exc()})

        if col2.button("🗑️ Delete Note", key=f"delete_note_{selected}"):
            if selected in notes:
                deleted_note_len = len(notes[selected])
                notes.pop(selected, None)
                try:
                    with open(notes_file, "w", encoding="utf-8") as f:
                        json.dump(notes, f, indent=2)
                    st.warning("🗑️ Note deleted.")
                    log_user_action(user_email, "NOTE_DELETED", {
                        "candidate": selected,
                        "note_len": deleted_note_len
                    })
                    update_metrics_summary("total_notes_deleted", 1)
                    update_metrics_summary("user_notes_deleted", 1, user_email=user_email)
                    st.rerun()
                except Exception as e:
                    st.error(f"Error deleting note: {e}")
                    log_system_event("ERROR", "NOTE_DELETE_FAILED", {"user_email": user_email, "candidate": selected, "error": str(e), "traceback": traceback.format_exc()})
            else:
                st.info("Note already deleted or does not exist.")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No candidates with notes yet. Add one below!")

    st.divider()
    st.markdown("### ➕ Add New Note")
    new_name = st.text_input("👤 Candidate Name", key="new_candidate_name")
    new_note = st.text_area("📝 Note", height=100, key="new_candidate_note")
    if st.button("➕ Save New Note", key="save_new_note_button"):
        if new_name.strip():
            # Check if note already exists for this candidate
            if new_name.strip() in notes:
                st.warning(f"Note for '{new_name.strip()}' already exists. Please select it from the dropdown to edit.")
                log_user_action(user_email, "NOTE_ADD_ATTEMPT_EXISTING", {"candidate": new_name.strip()})
                return
                
            notes[new_name.strip()] = new_note.strip()
            try:
                with open(notes_file, "w", encoding="utf-8") as f:
                    json.dump(notes, f, indent=2)
                st.success(f"✅ Note added for {new_name.strip()}")
                log_user_action(user_email, "NOTE_ADDED", {
                    "candidate": new_name.strip(),
                    "note_len": len(new_note.strip())
                })
                update_metrics_summary("total_notes_added", 1)
                update_metrics_summary("user_notes_added", 1, user_email=user_email)
                st.rerun()
            except Exception as e:
                st.error(f"Error adding new note: {e}")
                log_system_event("ERROR", "NOTE_ADD_FAILED", {"user_email": user_email, "candidate": new_name.strip(), "error": str(e), "traceback": traceback.format_exc()})
        else:
            st.error("❌ Candidate name cannot be empty.")
            log_user_action(user_email, "NOTE_ADD_FAILED", {"reason": "empty_candidate_name"})

    st.markdown('</div>', unsafe_allow_html=True)

# This block is for testing the notes page in isolation if needed
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Candidate Notes")
    st.title("Candidate Notes (Standalone Test)")

    # Mock user session state for standalone testing
    if "user_email" not in st.session_state:
        st.session_state.user_email = "test_notes_user@example.com"
        st.info("Running in standalone mode. Mocking user: test_notes_user@example.com")

    notes_page()
