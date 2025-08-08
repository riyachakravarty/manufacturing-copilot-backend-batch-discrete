from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd
import io
from io import StringIO
import os
import re
import json
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from scipy.stats import zscore
from langchain_community.llms import OpenAI  # updated import


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_df = None
augmented_df = None

class PromptRequest(BaseModel):
    prompt: str

class TreatmentRequest(BaseModel):
    column: str
    intervals: list
    method: str

def summarize_data(_):
    global uploaded_df
    if uploaded_df is not None:
        return uploaded_df.describe().to_string()
    return "No data uploaded."

def plot_variability_analysis_combined(selected_variable):
    global uploaded_df
    if uploaded_df is None:
        return "No data uploaded."

    fig = make_subplots(rows=2, cols=2, subplot_titles=['Box Plot', 'Line Plot', 'Histogram'],
                        row_heights=[0.5, 0.5], column_widths=[0.5, 0.5])
    fig.add_trace(go.Box(y=uploaded_df[selected_variable], name='Box Plot'), row=1, col=1)
    fig.add_trace(go.Scatter(x=uploaded_df['Date_time'], y=uploaded_df[selected_variable], mode='lines', name='Line Plot'), row=1, col=2)
    fig.add_trace(go.Histogram(x=uploaded_df[selected_variable], name='Histogram'), row=2, col=1)

    fig.update_layout(title_text=f'Combined Feature Analysis - {selected_variable}', showlegend=False, height=800, width=1500)
    return fig.to_json()

def plot_variability_tool(input_text):
    # Updated regex to make quotes optional and more flexible
    match = re.search(r"selected variable is ['\"]?(.+?)['\"]?$", input_text, re.IGNORECASE)
    if not match:
        print(f"[VARIABILITY TOOL] Could not parse selected variable from prompt: {input_text}")
        return "Could not find selected variable in prompt."
    selected_variable = match.group(1).strip()
    print(f"[VARIABILITY TOOL] Selected variable parsed: {selected_variable}")
    return plot_variability_analysis_combined(selected_variable)

def get_missing_datetime_intervals(df, datetime_col='Date_time'):
    if datetime_col not in df.columns:
        return []
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    inferred_freq = pd.infer_freq(df[datetime_col])
    if inferred_freq is None:
        diffs = df[datetime_col].diff().dropna()
        most_common_diff = diffs.mode()[0] if not diffs.empty else pd.Timedelta(hours=1)
        inferred_freq = most_common_diff
    full_range = pd.date_range(start=df[datetime_col].min(), end=df[datetime_col].max(), freq=inferred_freq)
    missing_times = full_range.difference(df[datetime_col])

    intervals = []
    if not missing_times.empty:
        start = missing_times[0]
        prev = start
        for t in missing_times[1:]:
            if (t - prev) != pd.Timedelta(inferred_freq):
                intervals.append((start, prev + pd.Timedelta(inferred_freq)))
                start = t
            prev = t
        intervals.append((start, prev + pd.Timedelta(inferred_freq)))
    return intervals

def get_missing_value_intervals(df, column, datetime_col='Date_time'):
    missing_intervals = []
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    in_interval = False
    start = None
    for i in range(len(df)):
        if pd.isna(df[column].iloc[i]):
            if not in_interval:
                start = df[datetime_col].iloc[i]
                in_interval = True
        elif in_interval:
            end = df[datetime_col].iloc[i]
            missing_intervals.append((start, end))
            in_interval = False
    if in_interval and start is not None:
        missing_intervals.append((start, df[datetime_col].iloc[-1]))
    return missing_intervals

def visualize_missing_data(input_text):
    global uploaded_df
    if uploaded_df is None:
        return "No data uploaded."
    match = re.search(r"selected variable is ['\"]?(.+?)['\"]?$", input_text, re.IGNORECASE)
    if not match:
        print(f"[VARIABILITY TOOL] Could not parse selected variable from prompt: {input_text}")
        return "Could not find selected variable in prompt."
    selected_variable = match.group(1).strip()
    print(f"[Vizualize function] Selected variable parsed: {selected_variable}")
    if selected_variable not in uploaded_df.columns:
        return f"Column '{selected_variable}' not in dataframe."
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=uploaded_df['Date_time'], y=uploaded_df[selected_variable], mode='lines+markers', name=selected_variable, line=dict(color='blue')))
    for start, end in get_missing_value_intervals(uploaded_df, selected_variable):
        fig.add_vrect(x0=start, x1=end, fillcolor="orange", opacity=0.3, line_width=0)
    for start, end in get_missing_datetime_intervals(uploaded_df):
        fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.2, line_width=0)
    fig.update_layout(title=f"Missing Data Visualization: '{selected_variable}' Over Time", xaxis_title='Date_time', yaxis_title=selected_variable, hovermode="x unified", height=600)
    return fig.to_json()

def get_outlier_intervals(df, column, datetime_col='Date_time', method='zscore', threshold=3):
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    if method == 'zscore':
        df['zscore'] = zscore(df[column].dropna())
        df['is_outlier'] = df['zscore'].abs() > threshold
    elif method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df['is_outlier'] = (df[column] < lower) | (df[column] > upper)
    else:
        raise ValueError("Unsupported method")

    outlier_intervals = []
    in_interval = False
    start = None
    for i in range(len(df)):
        if df['is_outlier'].iloc[i]:
            if not in_interval:
                start = df[datetime_col].iloc[i]
                in_interval = True
        elif in_interval:
            end = df[datetime_col].iloc[i]
            outlier_intervals.append((start, end))
            in_interval = False
    if in_interval:
        outlier_intervals.append((start, df[datetime_col].iloc[-1]))
    return outlier_intervals

def visualize_outlier_data(prompt):
    global uploaded_df
    if uploaded_df is None:
        return "No data uploaded."
    column_match = re.search(r"selected variable is ['\"](.+?)['\"]", prompt)
    if not column_match:
        return "Could not extract column from prompt."
    column = column_match.group(1)
    method = "zscore"
    if "iqr" in prompt.lower():
        method = "iqr"
    if column not in uploaded_df.columns:
        return f"Column '{column}' not found in data."
    if 'Date_time' not in uploaded_df.columns:
        return "'Date_time' column is missing."

    intervals = get_outlier_intervals(uploaded_df, column, method=method)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=uploaded_df['Date_time'], y=uploaded_df[column], mode='lines+markers', name=column, line=dict(color='blue')))
    for start, end in intervals:
        fig.add_vrect(x0=start, x1=end, fillcolor="purple", opacity=0.3, line_width=0)
    fig.update_layout(title=f"Outlier Analysis ({method.upper()}): '{column}'", xaxis_title='Date_time', yaxis_title=column, hovermode="x unified", height=600)
    return fig.to_json()

@app.get("/missing_datetime_intervals")
def missing_datetime_intervals():
    global uploaded_df
    if uploaded_df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)
    intervals = get_missing_datetime_intervals(uploaded_df, 'Date_time')
    formatted = [{"start": str(start), "end": str(end)} for start, end in intervals]
    return JSONResponse(content={"intervals": formatted})

@app.get("/missing_value_intervals")
def missing_value_intervals(column: str):
    print(f"Column received: {column}")
    
    global augmented_df
    if augmented_df is None:
        return JSONResponse(content={"error": "Missing Date Times treatment must be applied first."}, status_code=400)
    if column not in augmented_df.columns:
        return JSONResponse(content={"error": f"Column '{column}' not found in data."}, status_code=400)

    intervals = get_missing_value_intervals(augmented_df, column)
    formatted = [{"start": str(start), "end": str(end)} for start, end in intervals]
    return JSONResponse(content={"intervals": formatted})


@app.get("/get_columns")
def get_columns():
    global uploaded_df
    if uploaded_df is not None:
        return JSONResponse(content={"columns": uploaded_df.columns.tolist()})
    return JSONResponse(content={"error": "No file uploaded"}, status_code=400)

@app.post("/apply_treatment")
def apply_treatment(payload: dict):
    global uploaded_df, augmented_df

    columns = payload.get("columns", [])
    intervals = payload.get("intervals", [])
    method = payload.get("method")

    if not columns or not intervals or not method:
        return {"message": "Invalid payload: columns, intervals, and method are required."}

    # Always work off current state of augmented_df or build it fresh
    if augmented_df is not None:
        df = augmented_df.copy()
    else:
        df = uploaded_df.sort_values('Date_time').reset_index(drop=True)
        inferred_freq = pd.infer_freq(df['Date_time'])
        if inferred_freq is None:
            diffs = df['Date_time'].diff().dropna()
            most_common_diff = diffs.mode()[0] if not diffs.empty else pd.Timedelta(hours=1)
            inferred_freq = most_common_diff
        full_range = pd.date_range(start=df['Date_time'].min(), end=df['Date_time'].max(), freq=inferred_freq)
        df = pd.DataFrame({'Date_time': full_range}).merge(df, on='Date_time', how='left')

    for interval in intervals:
        start = pd.to_datetime(interval['start'])
        end = pd.to_datetime(interval['end'])
        mask = (df['Date_time'] >= start) & (df['Date_time'] <= end)

        for column in columns:
            if method == "Delete rows":
                df = df[~mask]
            elif method == "Forward fill":
                df.loc[mask, column] = df[column].ffill()
            elif method == "Backward fill":
                df.loc[mask, column] = df[column].bfill()
            elif method == "Mean":
                df.loc[mask, column] = df[column].mean()
            elif method == "Median":
                df.loc[mask, column] = df[column].median()

    # ✅ Reassign updated dataframe to global
    augmented_df = df
    print("Shape of augmented_df after apply_treatment:", augmented_df.shape)
    return {"message": "Treatment applied successfully"}

@app.post("/apply_missing_value_treatment")
def apply_missing_value_treatment(payload: dict):
    global augmented_df
    print("Shape of augmented_df at start of apply_missing_value_treatment:", augmented_df.shape if 'augmented_df' in globals() else "augmented_df not found")

    column = payload.get("column")
    intervals = payload.get("intervals", [])
    method = payload.get("method")

    if not column or not intervals or not method:
        return JSONResponse(content={"error": "Invalid payload: column, intervals, and method are required."}, status_code=400)

    if augmented_df is None:
        return JSONResponse(content={"error": "Missing Date Times treatment must be applied first."}, status_code=400)

    for interval in intervals:
        start = pd.to_datetime(interval['start'])
        end = pd.to_datetime(interval['end'])
        mask = (augmented_df['Date_time'] >= start) & (augmented_df['Date_time'] <= end)

        if method == "Delete rows":
            augmented_df = augmented_df[~mask]
        elif method == "Forward fill":
            augmented_df.loc[mask, column] = augmented_df[column].ffill()
        elif method == "Backward fill":
            augmented_df.loc[mask, column] = augmented_df[column].bfill()
        elif method == "Mean":
            mean_val = augmented_df[column].mean()
            augmented_df.loc[mask, column] = mean_val
        elif method == "Median":
            median_val = augmented_df[column].median()
            augmented_df.loc[mask, column] = median_val
        else:
            return JSONResponse(content={"error": f"Unknown method: {method}"}, status_code=400)

    return JSONResponse(content={"message": f"Missing value treatment applied to column '{column}'."})

    
@app.get("/download")
def download_file():
    global uploaded_df, augmented_df
    df_to_download = augmented_df if augmented_df is not None else uploaded_df
    if df_to_download is None:
        return JSONResponse(content={"message": "No data available for download"}, status_code=400)
    try:
        stream = StringIO()
        df_to_download.to_csv(stream, index=False)
        stream.seek(0)
        return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=treated_data.csv"})
    except Exception as e:
        return JSONResponse(content={"message": f"Download failed: {str(e)}"}, status_code=500)

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global uploaded_df, augmented_df
    try:
        content = await file.read()
        uploaded_df = pd.read_csv(io.BytesIO(content))
        uploaded_df['Date_time'] = pd.to_datetime(uploaded_df['Date_time'])
        augmented_df = None
        return JSONResponse(content={"message": "File uploaded successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
tools = [
    Tool(name="SummarizeData", func=summarize_data, description="Summarizes the uploaded manufacturing dataset"),
    Tool(name="VariabilityAnalysis", func=plot_variability_tool, description="Generates variability analysis plots. Format: selected variable is 'ColumnName'"),
    Tool(name="MissingValueAnalysis", func=visualize_missing_data, description="Generates missing value analysis plots. Format: selected variable is 'ColumnName'")
]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

@app.post("/chat")
async def chat(request: Request):
    global uploaded_df, augmented_df
    if uploaded_df is None:
        print("[CHAT] ERROR: No uploaded DataFrame found.")
    else:
        print(f"[CHAT] DataFrame shape: {uploaded_df.shape}")
        print(f"[CHAT] DataFrame columns: {uploaded_df.columns.tolist()}")

    try:
        body = await request.body()
        data = json.loads(body)
        prompt = data.get("prompt", "")
    except Exception as e:
        print(f"[CHAT] Error parsing request body: {e}")
        return JSONResponse(content={"type": "text", "data": "Invalid request format."}, status_code=400)

    prompt_lower = prompt.lower()
    try:
        if "missing value analysis" in prompt_lower or "anomaly analysis" in prompt_lower:
            result = visualize_missing_data(prompt)
            return JSONResponse(content={"type": "plot", "data": json.loads(result)})

        elif "outlier analysis" in prompt_lower:
            df = augmented_df if augmented_df is not None else uploaded_df
            if df is None:
                raise ValueError("No data uploaded yet.")
            result = visualize_outlier_data(prompt, df)
            return JSONResponse(content={"type": "plot", "data": json.loads(result)})

        elif "variability analysis" in prompt_lower:
            result = plot_variability_tool(prompt)
            print("[CHAT] Result type from plot_variability_tool:", type(result))
            print("[CHAT] Raw result:", result)

            # If the tool returned a plain error string, handle gracefully
            if isinstance(result, str) and result.startswith("Could not find selected variable"):
                return JSONResponse(content={"type": "text", "data": result}, status_code=400)

            return JSONResponse(content={"type": "plot", "data": json.loads(result)})

        else:
            result = agent.run(prompt)
            return JSONResponse(content={"type": "text", "data": str(result)})

    except Exception as e:
        print(f"[CHAT] Exception: {e}")
        return JSONResponse(content={"type": "text", "data": f"Error: {e}"}, status_code=500)
