import streamlit as st
import json
import bcrypt
import os
from utils.logger import log_user_action, log_system_event # Import logging functions

# File to store user credentials
USER_DB_FILE = "users.json"
ADMIN_USERNAME = "mn@gmail.com" # Define your primary admin username here

def load_users():
    """Loads user data from the JSON file."""
    if not os.path.exists(USER_DB_FILE):
        with open(USER_DB_FILE, "w") as f:
            json.dump({}, f)
    with open(USER_DB_FILE, "r") as f:
        users = json.load(f)
        # Ensure each user has a 'status' key for backward compatibility
        for username, data in users.items():
            if isinstance(data, str): # Old format: "username": "hashed_password"
                users[username] = {"password": data, "status": "active", "role": "recruiter"} # Default role
            elif "status" not in data:
                data["status"] = "active"
            if "role" not in data: # Ensure role is set for existing users
                # If username is the primary ADMIN_USERNAME, set role to 'admin', otherwise 'recruiter'
                data["role"] = "admin" if username == ADMIN_USERNAME else "recruiter"
        return users

def save_users(users):
    """Saves user data to the JSON file."""
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f, indent=4)

def hash_password(password):
    """Hashes a password using bcrypt."""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def check_password(password, hashed_password):
    """Checks a password against its bcrypt hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def register_section():
    """Public self-registration form."""
    st.subheader("📝 Create New Account")
    with st.form("registration_form", clear_on_submit=True):
        new_username = st.text_input("Choose Username (Email address recommended)", key="new_username_reg_public")
        new_password = st.text_input("Choose Password", type="password", key="new_password_reg_public")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password_reg_public")
        register_button = st.form_submit_button("Register New Account")

        if register_button:
            if not new_username or not new_password or not confirm_password:
                st.error("Please fill in all fields.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            else:
                users = load_users()
                if new_username in users:
                    st.error("Username already exists. Please choose a different one.")
                    log_system_event("WARNING", "USER_REGISTRATION_FAILED", {"username": new_username, "reason": "Username already exists"})
                else:
                    users[new_username] = {"password": hash_password(new_password), "status": "active", "role": "recruiter"} # Default role for public registration
                    save_users(users)
                    st.success("✅ Registration successful! You can now switch to the 'Login' option.")
                    log_system_event("INFO", "USER_REGISTERED_PUBLIC", {"user_email": new_username, "role": "recruiter"})
                    st.session_state.active_login_tab_selection = "Login"
                    st.rerun() # Rerun to update the UI and switch tab

def admin_registration_section():
    """Admin-driven user creation form."""
    st.subheader("➕ Create New User Account (Admin Only)")
    with st.form("admin_registration_form", clear_on_submit=True):
        new_username = st.text_input("New User's Username (Email)", key="new_username_admin_reg")
        new_password = st.text_input("New User's Password", type="password", key="new_password_admin_reg")
        new_role = st.selectbox("Assign Role", ["recruiter", "admin"], key="new_user_role") # Allow admin to select role
        admin_register_button = st.form_submit_button("Add New User")

        if admin_register_button:
            if not new_username or not new_password:
                st.error("Please fill in all fields.")
            else:
                users = load_users()
                if new_username in users:
                    st.error(f"User '{new_username}' already exists.")
                    log_user_action(st.session_state.get('username', 'unknown_admin'), "ADMIN_ADD_USER_FAILED", {"target_user": new_username, "reason": "User already exists"})
                else:
                    users[new_username] = {"password": hash_password(new_password), "status": "active", "role": new_role} # Use selected role
                    save_users(users)
                    st.success(f"✅ User '{new_username}' with role '{new_role}' added successfully!")
                    log_user_action(st.session_state.get('username', 'unknown_admin'), "ADMIN_ADD_USER_SUCCESS", {"target_user": new_username, "role": new_role})

def admin_password_reset_section():
    """Admin-driven password reset form."""
    st.subheader("🔑 Reset User Password (Admin Only)")
    users = load_users()
    user_options = [user for user in users.keys() if user != ADMIN_USERNAME] # Cannot reset primary admin's own password here

    if not user_options:
        st.info("No other users to reset passwords for.")
        return

    with st.form("admin_reset_password_form", clear_on_submit=True):
        selected_user = st.selectbox("Select User to Reset Password For", user_options, key="reset_user_select")
        new_password = st.text_input("New Password", type="password", key="new_password_reset")
        reset_button = st.form_submit_button("Reset Password")

        if reset_button:
            if not new_password:
                st.error("Please enter a new password.")
                log_user_action(st.session_state.get('username', 'unknown_admin'), "ADMIN_PASSWORD_RESET_FAILED", {"target_user": selected_user, "reason": "No new password provided"})
            else:
                users[selected_user]["password"] = hash_password(new_password)
                save_users(users)
                st.success(f"✅ Password for '{selected_user}' has been reset.")
                log_user_action(st.session_state.get('username', 'unknown_admin'), "ADMIN_PASSWORD_RESET_SUCCESS", {"target_user": selected_user})

def admin_disable_enable_user_section():
    """Admin-driven user disable/enable form."""
    st.subheader("⛔ Toggle User Status (Admin Only)")
    users = load_users()
    user_options = [user for user in users.keys() if user != ADMIN_USERNAME] # Cannot disable primary admin's own account here

    if not user_options:
        st.info("No other users to manage status for.")
        return
    
    with st.form("admin_toggle_user_status_form", clear_on_submit=False): # Keep values after submit for easier toggling
        selected_user = st.selectbox("Select User to Toggle Status", user_options, key="toggle_user_select")
        
        current_status = users[selected_user]["status"]
        st.info(f"Current status of '{selected_user}': **{current_status.upper()}**")

        if st.form_submit_button(f"Toggle to {'Disable' if current_status == 'active' else 'Enable'} User"):
            new_status = "disabled" if current_status == "active" else "active"
            users[selected_user]["status"] = new_status
            save_users(users)
            st.success(f"✅ User '{selected_user}' status set to **{new_status.upper()}**.")
            log_user_action(st.session_state.get('username', 'unknown_admin'), "ADMIN_USER_STATUS_TOGGLE_SUCCESS", {"target_user": selected_user, "new_status": new_status})
            st.rerun() # Rerun to update the displayed status immediately

def add_admin_user(username, password):
    """
    Programmatically adds a new user with 'admin' role.
    This function can be called for initial setup or by an existing admin.
    """
    users = load_users()
    if username in users:
        st.warning(f"Admin user '{username}' already exists. Skipping creation.")
        return False
    else:
        users[username] = {"password": hash_password(password), "status": "active", "role": "admin"}
        save_users(users)
        st.success(f"Admin user '{username}' created successfully.")
        log_system_event("INFO", "ADMIN_USER_CREATED_PROGRAMMATICALLY", {"user_email": username, "role": "admin"})
        return True


def login_section():
    """Handles user login and public registration."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "username" not in st.session_state:
        st.session_state.username = None
    if "user_role" not in st.session_state: # Store user role
        st.session_state.user_role = None
    
    # Initialize active_login_tab_selection if not present
    if "active_login_tab_selection" not in st.session_state:
        # Default to 'Register' if no users, otherwise 'Login'
        if not os.path.exists(USER_DB_FILE) or len(load_users()) == 0:
            st.session_state.active_login_tab_selection = "Register"
        else:
            st.session_state.active_login_tab_selection = "Login"


    if st.session_state.authenticated:
        return True

    # Use st.radio to simulate tabs if st.tabs() default_index is not supported
    tab_selection = st.radio(
        "Select an option:",
        ("Login", "Register"),
        key="login_register_radio",
        index=0 if st.session_state.active_login_tab_selection == "Login" else 1
    )

    if tab_selection == "Login":
        st.subheader("🔐 HR Login")
        st.info("If you don't have an account, please go to the 'Register' option first.") # Added instructional message
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", key="username_login")
            password = st.text_input("Password", type="password", key="password_login")
            submitted = st.form_submit_button("Login")

            if submitted:
                users = load_users()
                if username not in users:
                    st.error("❌ Invalid username or password. Please register if you don't have an account.")
                    log_user_action(username, "LOGIN_FAILED", {"reason": "Username not found"})
                else:
                    user_data = users[username]
                    if user_data["status"] == "disabled":
                        st.error("❌ Your account has been disabled. Please contact an administrator.")
                        log_user_action(username, "LOGIN_FAILED", {"reason": "Account disabled", "status": "disabled"})
                    elif check_password(password, user_data["password"]):
                        st.session_state.authenticated = True
                        st.session_state.username = username # Store username
                        st.session_state.user_email = username # Consistent with logger
                        st.session_state.user_role = user_data.get("role", "recruiter") # Store user role
                        st.success("✅ Login successful!")
                        log_user_action(username, "LOGIN_SUCCESS", {"role": st.session_state.user_role})
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password.")
                        log_user_action(username, "LOGIN_FAILED", {"reason": "Incorrect password"})
    
    elif tab_selection == "Register": # This will be the initially selected option for new users
        register_section()

    return st.session_state.authenticated

# Helper function to check if the current user is an admin
def is_current_user_admin():
    # This function now correctly checks if the logged-in user's role is 'admin'
    return st.session_state.get("authenticated", False) and \
           st.session_state.get("user_role") == "admin"


# Example of how to use it if running login.py directly for testing
if __name__ == "__main__":
    st.set_page_config(page_title="Login/Register", layout="centered")
    st.title("ScreenerPro Authentication (Test)")
    
    # Ensure primary admin user exists for testing
    add_admin_user(ADMIN_USERNAME, "adminpass") # Automatically create the default admin user
    
    # Add the new admin user as requested
    add_admin_user("mn@gmail.com", "12345")

    if login_section():
        st.write(f"Welcome, {st.session_state.username}!")
        st.write(f"Your role: {st.session_state.user_role}")
        st.write("You are logged in.")
        
        # This condition should check if the current user's role is 'admin'
        if st.session_state.get("user_role") == "admin":
            st.markdown("---")
            st.header("Admin Test Section (You are admin)")
            admin_registration_section()
            admin_password_reset_section()
            admin_disable_enable_user_section()

            st.subheader("All Registered Users (Admin View):")
            # This part requires pandas, which is typically in main.py.
            # For standalone login.py testing, ensure pandas is imported.
            try:
                import pandas as pd
                users_data = load_users()
                if users_data:
                    display_users = []
                    for user, data in users_data.items():
                        hashed_pass = data.get("password", "N/A")
                        status = data.get("status", "N/A")
                        role = data.get("role", "N/A")
                        display_users.append([user, hashed_pass, status, role])
                    st.dataframe(pd.DataFrame(display_users, columns=["Email/Username", "Hashed Password (DO NOT EXPOSE)", "Status", "Role"]), use_container_width=True)
                else:
                    st.info("No users registered yet.")
            except ImportError:
                st.warning("Pandas is not imported in login.py. User table cannot be displayed in standalone test.")
            except Exception as e:
                st.error(f"Error loading user data: {e}")
        else:
            st.info("Log in as an admin (e.g., 'admin@forscreenerpro' or 'mn@gmail.com') to see admin features.")
            
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.pop('username', None)
            st.session_state.pop('user_email', None) # Clear user_email too
            st.session_state.pop('user_role', None) # Clear user_role too
            st.rerun()
    else:
        st.info("Please login or register to continue.")
