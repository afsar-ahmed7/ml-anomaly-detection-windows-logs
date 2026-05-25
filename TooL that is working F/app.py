import dash
from dash import html, dcc, Input, Output
import pandas as pd
import plotly.express as px
import os

# ✅ Set path to dataset
DATA_PATH = os.path.join("data", "Windows_Event_Log_Feature_Enhanced.csv")

# ✅ Confirm dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ {DATA_PATH} not found. Run the pipeline first.")

# ✅ Load dataset
df = pd.read_csv(DATA_PATH)
print(f"✅ Loaded file from path: {os.path.abspath(DATA_PATH)}")
print("👀 Dash is loading the following Security Identifiers:")
print(df["Security Identifier"].dropna().unique())

# ✅ Rename duplicated feature columns (from merge) if present
df.rename(columns={
    "login_frequency_x": "login_frequency",
    "failed_login_attempts_x": "failed_login_attempts",
    "file_access_count_x": "file_access_count",
    "application_activity_count_x": "application_activity_count",
    "avg_session_duration_x": "avg_session_duration",
    "total_event_count_x": "total_event_count"
}, inplace=True)

# ✅ Ensure expected numeric columns
df["avg_session_duration"] = pd.to_numeric(df.get("avg_session_duration", 0), errors="coerce").fillna(0)
for col in ["login_frequency", "failed_login_attempts"]:
    df[col] = pd.to_numeric(df.get(col, 0), errors="coerce").fillna(0).astype(int)

# ✅ Fill missing anomaly labels (if not generated yet)
for col in ["anomaly_label", "lof_label", "svm_label"]:
    if col not in df.columns:
        df[col] = "N/A"

# ✅ Dropdown options
dropdown_options = [
    {"label": user, "value": user}
    for user in df["Security Identifier"].dropna().unique()
]

# ✅ Prebuild anomaly pie chart
anomaly_counts = {
    "Isolation Forest": df["anomaly_label"].value_counts().get("Anomalous", 0),
    "LOF": df["lof_label"].value_counts().get("Anomalous", 0),
    "SVM": df["svm_label"].value_counts().get("Anomalous", 0)
}
anomaly_df = pd.DataFrame(list(anomaly_counts.items()), columns=["Model", "Anomalous"])
anomaly_fig = px.pie(anomaly_df, names="Model", values="Anomalous", title="Anomalies Detected")

# ✅ App setup
app = dash.Dash(__name__)
app.title = "User Behaviour Dashboard"
server = app.server

# ✅ Layout
app.layout = html.Div([
    html.H1("User Behaviour Anomaly Dashboard", style={"textAlign": "center"}),

    dcc.Dropdown(
        id="user_dropdown",
        options=dropdown_options,
        value=dropdown_options[0]["value"],
        style={"width": "60%", "margin": "auto"}
    ),

    html.Div(id="user_info", style={"padding": "20px"}),

    html.H3("Anomalies Detected by Model", style={"textAlign": "center"}),
    dcc.Graph(id="anomaly_bar", figure=anomaly_fig),

    html.H3("Login Frequency vs Failed Attempts", style={"textAlign": "center"}),
    dcc.Graph(id="login_failed_graph")
])

# ✅ Callback: user info
@app.callback(
    Output("user_info", "children"),
    Input("user_dropdown", "value")
)
def update_user_info(user_id):
    row = df[df["Security Identifier"] == user_id].iloc[0]
    return html.Div([
        html.H4(f"User: {user_id}"),
        html.P(f"Login Frequency: {row['login_frequency']}"),
        html.P(f"Failed Login Attempts: {row['failed_login_attempts']}"),
        html.P(f"File Access Count: {row.get('file_access_count', 'N/A')}"),
        html.P(f"Session Duration: {round(row['avg_session_duration'], 2)} minutes"),
        html.P(f"Anomaly (Isolation Forest): {row['anomaly_label']}"),
        html.P(f"Anomaly (LOF): {row['lof_label']}"),
        html.P(f"Anomaly (SVM): {row['svm_label']}")
    ])

# ✅ Callback: login vs failed attempts chart (for selected user)
@app.callback(
    Output("login_failed_graph", "figure"),
    Input("user_dropdown", "value")
)
def login_vs_failed(user_id):
    row = df[df["Security Identifier"] == user_id]
    if row.empty:
        return px.bar(title="No data available for selected user")

    row = row.iloc[0]
    values = {
        "Login Frequency": row["login_frequency"],
        "Failed Login Attempts": row["failed_login_attempts"]
    }

    fig = px.bar(
        x=list(values.keys()),
        y=list(values.values()),
        title=f"{user_id} - Login vs Failed Attempts",
        labels={"x": "Metric", "y": "Count"}
    )
    fig.update_layout(yaxis_tickformat=',')
    return fig

# ✅ Run app
if __name__ == '__main__':
    app.run(debug=True)
