# scripts/preprocessing.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def load_and_clean(
    raw_log_path,
    category_map_path="category_mapping.csv",
    save_path="data/Windows_Events_Log_With_Synthetic_Users.csv"
):
    """
    Clean and prepare the Windows event log dataset.

    Steps:
    - Keep essential columns
    - Fix and normalize datetime
    - Merge with event category mapping
    - Fill missing Security Identifiers using synthetic users (if any)
    - Save the final cleaned dataset
    """

    # ---------------- Step 1: Load raw logs ----------------
    columns_to_keep = [
        "Event ID", "Security Identifier", "Created Date/Time - UTC+00:00 (dd/MM/yyyy)",
        "Event Record ID", "Event Description Summary", "Level",
        "Provider Name", "Task Category", "Location"
    ]

    df = pd.read_csv(raw_log_path, usecols=columns_to_keep, dtype=str, low_memory=False)

    # ---------------- Step 2: Generate realistic timestamps ----------------
    start_date = datetime(2019, 1, 1)
    end_date = datetime(2024, 1, 1)
    num_rows = len(df)

    random_dates = [
        start_date + timedelta(days=np.random.randint(0, (end_date - start_date).days))
        for _ in range(num_rows)
    ]

    df["Created Date/Time - UTC+00:00 (dd/MM/yyyy)"] = df["Created Date/Time - UTC+00:00 (dd/MM/yyyy)"].astype(str)
    df["Created Date/Time - UTC+00:00 (dd/MM/yyyy)"] = [
        random_dates[i].strftime("%d-%m-%Y") + " " + df.iloc[i]['Created Date/Time - UTC+00:00 (dd/MM/yyyy)']
        for i in range(num_rows)
    ]

    df["Created Date/Time - UTC+00:00 (dd/MM/yyyy)"] = pd.to_datetime(
        df["Created Date/Time - UTC+00:00 (dd/MM/yyyy)"],
        format="%d-%m-%Y %S:%M.%H",
        errors='coerce'
    )

    # ---------------- Step 3: Merge with Category Mapping ----------------
    df_cat = pd.read_csv(category_map_path)
    df["Event ID"] = df["Event ID"].astype(int)
    df_cat["Event ID"] = df_cat["Event ID"].astype(int)

    df = df.merge(df_cat[["Event ID", "Category"]], on="Event ID", how="left")
    df["Category"] = df["Category"].fillna("Other")

    # ---------------- Step 4: Fill Missing Security Identifiers ----------------
    df["Security Identifier"] = df["Security Identifier"].replace(
        ["", " ", "nan", "NaN", "NONE", "None"], np.nan
    )

    if df["Security Identifier"].isna().sum() > 0:
        np.random.seed(42)

        user_map = {
            "Logon/Logoff": ["UserA", "UserB", "Admin"],
            "Privilege Use": ["Admin", "UserA"],
            "Account Management": ["Admin", "ServiceAccount1"],
            "System": ["SYSTEM", "ServiceAccount2"],
            "Policy Change": ["Admin"],
            "Detailed Tracking": ["ServiceAccount2", "SYSTEM", "UserC"],
            "Object Access": ["Admin", "UserA", "ServiceAccount1"],
            "Other": ["UserC", "UserD", "Guest"]
        }

        for category, users in user_map.items():
            mask = (df["Security Identifier"].isna()) & (df["Category"] == category)
            df.loc[mask, "Security Identifier"] = np.random.choice(users, size=mask.sum())

        filled = df["Security Identifier"].isin(sum(user_map.values(), [])).sum()
        print(f"🧩 Synthetic users filled: {filled}")
    else:
        print("✅ No missing Security Identifiers — no synthetic users added.")

    # ---------------- Step 5: Save Final File ----------------
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Cleaned event log saved to: {save_path}")

    return df
