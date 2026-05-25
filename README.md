

\# Forensic User Behaviour Analysis \& ML-Based Anomaly Detection



\## Overview

A forensic workflow analysing Windows event logs for insider threat patterns using machine learning anomaly detection.



\## Technologies

\- Python, scikit-learn

\- Windows Event Logs

\- Isolation Forest, Random Forest, SVM, LOF



\## Project Structure

\- `Jupyter\_Notebook\_User\_behaviour\_Analysis/` — analysis notebooks and models

\- `pipeline/` — automated pipeline scripts and app



\## Note

Large raw data files are excluded from this repo due to size limits.

Forensic User Behaviour Analysis Tool – Usage Instructions

\----------------------------------------------------------



📁 Folder Structure:

\- run\_pipeline.py       : Preprocesses logs, trains models, generates output

\- app.py                : Launches the dashboard

\- data/                 : Place raw Windows Event Log CSV files here

\- output/               : Contains generated plots and evaluation results

\- models/               : Saved trained machine learning models

\- requirements.txt      : Python dependencies for the project



\----------------------------------------------------------

🔧 Setup Instructions:



1\. Open PowerShell and navigate to your project folder:

&#x20;  cd path\\to\\your\\project\\folder

Or Open the project folder and Open PowerShell from here.



2\. Install all required dependencies:

&#x20;  "pip install -r requirements.txt"



\----------------------------------------------------------

🚀 Running the Tool:



1\. Run the pipeline to preprocess data and train models:

&#x20;  "python run\_pipeline.py"



&#x20;  This will:

&#x20;  - Save feature-enhanced datasets to the 'data/' folder

&#x20;  - Save model files to the 'models/' folder

&#x20;  - Generate output metrics and plots in the 'output/' folder



2\. Launch the dashboard:

&#x20;  "python app.py"



&#x20;  After a few seconds, you should see in PowerShell:

&#x20;  Dash is running on http://127.0.0.1:8050/



3\. Open your web browser and go to:

&#x20;  http://127.0.0.1:8050/



&#x20;  Use the dashboard to:

&#x20;  - Explore anomaly detection results

&#x20;  - Review user-specific behavior patterns



\----------------------------------------------------------⚠️ Notes:

\- Input log files must be in CSV format.

\- Ensure Python 3.8+ is installed and pip is working.



