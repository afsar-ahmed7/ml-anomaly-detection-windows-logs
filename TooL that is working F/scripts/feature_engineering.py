import pandas as pd
import numpy as np

def compute_features(df):
    """
    Given a cleaned and categorized DataFrame, compute user-level behavior features.
    Returns:
        df (DataFrame): Original DataFrame with added feature columns
        user_summary (DataFrame): Aggregated user-level feature summary
    """

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["Created Date/Time - UTC+00:00 (dd/MM/yyyy)"], errors='coerce')

    # ---------------- LOGIN FREQUENCY ----------------
    logon_df = df[df["Category"].str.contains("Logon/Logoff", case=False, na=False)]
    login_freq = logon_df.groupby("Security Identifier").size().reset_index(name="login_frequency")
    df = df.merge(login_freq, on="Security Identifier", how="left")
    df["login_frequency"] = df["login_frequency"].fillna(0).astype(int)

    # ---------------- FAILED LOGIN ATTEMPTS ----------------
    failed_df = df[df["Event ID"] == 4625]
    failed_login_freq = failed_df.groupby("Security Identifier").size().reset_index(name="failed_login_attempts")
    df = df.merge(failed_login_freq, on="Security Identifier", how="left")
    df["failed_login_attempts"] = df["failed_login_attempts"].fillna(0).astype(int)

    # ---------------- FILE ACCESS COUNT ----------------
    file_access_df = df[df["Category"] == "Object Access"]
    file_access_freq = file_access_df.groupby("Security Identifier").size().reset_index(name="file_access_count")
    df = df.merge(file_access_freq, on="Security Identifier", how="left")
    df["file_access_count"] = df["file_access_count"].fillna(0).astype(int)

    # ---------------- SESSION DURATION ----------------
    logons = df[df["Event ID"] == 4624].copy()
    logoffs = df[df["Event ID"] == 4634].copy()

    # Drop rows with missing timestamps
    logons = logons.dropna(subset=["timestamp"])
    logoffs = logoffs.dropna(subset=["timestamp"])

    if not logoffs.empty and not logons.empty:
        logons.rename(columns={"timestamp": "timestamp_logon"}, inplace=True)
        logoffs.rename(columns={"timestamp": "timestamp_logoff"}, inplace=True)

        logons.sort_values(by=["timestamp_logon", "Security Identifier"], inplace=True)
        logoffs.sort_values(by=["timestamp_logoff", "Security Identifier"], inplace=True)

        sessions = pd.merge_asof(
            logons, logoffs,
            by="Security Identifier",
            left_on="timestamp_logon",
            right_on="timestamp_logoff",
            direction="forward"
        )

        sessions["session_duration_min"] = (
            sessions["timestamp_logoff"] - sessions["timestamp_logon"]
        ).dt.total_seconds() / 60

        sessions = sessions[sessions["session_duration_min"] >= 0]

        avg_session = sessions.groupby("Security Identifier")["session_duration_min"] \
                              .mean().reset_index(name="avg_session_duration")

        df = df.merge(avg_session, on="Security Identifier", how="left")
    else:
        df["avg_session_duration"] = None

    # ---------------- TOTAL EVENT COUNT ----------------
    event_volume = df.groupby("Security Identifier").size().reset_index(name="total_event_count")
    df = df.merge(event_volume, on="Security Identifier", how="left")

    # ---------------- APPLICATION ACTIVITY ----------------
    keywords = ["application", "install", "app", "applocker", "process", "executable", ".exe"]
    app_df = df[df["Event Description Summary"].str.lower().fillna("").str.contains('|'.join(keywords))]
    app_activity = app_df.groupby("Security Identifier").size().reset_index(name="application_activity_count")
    df = df.merge(app_activity, on="Security Identifier", how="left")
    df["application_activity_count"] = df["application_activity_count"].fillna(0).astype(int)

    # ---------------- USER SUMMARY TABLE ----------------
    user_summary = df.groupby("Security Identifier")[[ 
        "login_frequency",
        "failed_login_attempts",
        "file_access_count",
        "application_activity_count",
        "avg_session_duration",
        "total_event_count"
    ]].mean().reset_index()

    return df, user_summary
