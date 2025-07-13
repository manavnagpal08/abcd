import json
import os
from datetime import datetime, timedelta

# Define log file paths
# Assuming 'data' directory exists in your project root for storing persistent files
LOG_DIR = "data"
USER_ACTIVITY_LOG_FILE = os.path.join(LOG_DIR, "user_activity_log.json")
SYSTEM_EVENTS_LOG_FILE = os.path.join(LOG_DIR, "system_events_log.json")
METRICS_SUMMARY_FILE = os.path.join(LOG_DIR, "metrics_summary.json")

def _initialize_log_file(filepath):
    """Initializes a JSON log file if it doesn't exist, ensuring the directory exists."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    if not os.path.exists(filepath):
        with open(filepath, 'w') as f:
            json.dump([], f) # Start with an empty list for logs

def _read_json_file(filepath):
    """Reads content from a JSON file. Initializes if it doesn't exist."""
    _initialize_log_file(filepath) # Ensure file exists before reading
    with open(filepath, 'r') as f:
        try:
            content = json.load(f)
            return content if isinstance(content, list) or isinstance(content, dict) else []
        except json.JSONDecodeError:
            # Handle empty or corrupted JSON files by returning an empty list/dict
            print(f"Warning: {filepath} is empty or corrupted. Initializing with empty data.")
            if filepath in [USER_ACTIVITY_LOG_FILE, SYSTEM_EVENTS_LOG_FILE]:
                return []
            elif filepath == METRICS_SUMMARY_FILE:
                return {}
            return [] # Default to list

def _write_json_file(filepath, data):
    """Writes content to a JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def log_user_action(user_email: str, action: str, details: dict = None, ip_address: str = None):
    """Logs a user's action to the user activity log."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user_email": user_email,
        "action": action,
        "details": details if details is not None else {},
        "ip_address": ip_address
    }
    logs = _read_json_file(USER_ACTIVITY_LOG_FILE)
    logs.append(log_entry)
    _write_json_file(USER_ACTIVITY_LOG_FILE, logs)
    # print(f"Logged user action: {log_entry}") # Uncomment for debugging

def log_system_event(level: str, event: str, details: dict = None, stacktrace: str = None):
    """Logs a system event (INFO, WARNING, ERROR, CRITICAL)."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "event": event,
        "details": details if details is not None else {},
        "stacktrace": stacktrace
    }
    logs = _read_json_file(SYSTEM_EVENTS_LOG_FILE)
    logs.append(log_entry)
    _write_json_file(SYSTEM_EVENTS_LOG_FILE, logs)
    # print(f"Logged system event: {log_entry}") # Uncomment for debugging

def get_user_activity_logs():
    """Retrieves all user activity logs."""
    return _read_json_file(USER_ACTIVITY_LOG_FILE)

def get_system_events_logs():
    """Retrieves all system events logs."""
    return _read_json_file(SYSTEM_EVENTS_LOG_FILE)

def update_metrics_summary(key: str, value: int, user_email: str = None, date: str = None):
    """
    Updates a specific metric in the metrics summary.
    Metrics are stored nested: {metric_key: {user_email (optional): {date: count}}}
    """
    metrics = _read_json_file(METRICS_SUMMARY_FILE)
    if not isinstance(metrics, dict): # Ensure it's a dictionary for metrics
        metrics = {}

    if not date:
        date = datetime.now().strftime("%Y-%m-%d")

    # Ensure the top-level metric key exists
    if key not in metrics:
        metrics[key] = {}

    if user_email: # User-specific metrics (e.g., user_resumes_screened)
        if user_email not in metrics[key]:
            metrics[key][user_email] = {}
        # Increment the count for that user on that specific date
        metrics[key][user_email][date] = metrics[key][user_email].get(date, 0) + value
    else: # Global metrics (e.g., total_resumes_screened)
        # Increment the global count for that specific date
        metrics[key][date] = metrics[key].get(date, 0) + value

    _write_json_file(METRICS_SUMMARY_FILE, metrics)
    # print(f"Updated metric: {key}, {user_email if user_email else 'global'}, {date}, +{value}") # Uncomment for debugging

def get_metrics_summary():
    """Retrieves the full metrics summary."""
    return _read_json_file(METRICS_SUMMARY_FILE)

# Ensure log directories/files exist on import
_initialize_log_file(USER_ACTIVITY_LOG_FILE)
_initialize_log_file(SYSTEM_EVENTS_LOG_FILE)
_initialize_log_file(METRICS_SUMMARY_FILE)
