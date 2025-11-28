from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import io
from io import StringIO
import os
import re
import json
from plotly.subplots import make_subplots
import plotly.graph_objs as go
#from langchain.agents import initialize_agent, Tool
#from langchain.llms import OpenAI
from scipy.stats import zscore
#from langchain_community.llms import   # updated import
from pydantic import BaseModel
import plotly
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from datetime import datetime
import shap


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
last_continuous_ranges = None
features_raw_df = None

class PromptRequest(BaseModel):
    prompt: str

class TreatmentRequest(BaseModel):
    column: str
    intervals: list
    method: str
########################## Data visualization and engineering tab ##########################################

def summarize_data(_):
    global uploaded_df
    if uploaded_df is not None:
        return uploaded_df.describe().to_string()
    return "No data uploaded."


class BatchProfileRequest(BaseModel):
    columns: List[str]
    batch_numbers: List[str]

@app.post("/run_batch_profiles")
def run_batch_profiles(req: BatchProfileRequest):
    global uploaded_df, augmented_df

    df = augmented_df if augmented_df is not None else uploaded_df

    if df is None:
        return JSONResponse(
            content={"error": "No data uploaded"},
            status_code=400
        )

    # Validate columns
    missing = [c for c in req.columns if c not in df.columns]
    if missing:
        return JSONResponse(
            content={"error": f"Columns not found: {missing}"},
            status_code=400
        )

    if "Batch_No" not in df.columns:
        return JSONResponse(
            content={"error": "'Batch_No' column missing"},
            status_code=400
        )

    # Make a copy and add Batch_Counter per batch
    df = df.copy()
    df["Batch_Counter"] = df.groupby("Batch_No").cumcount() + 1

    # Filter batches
    df["Batch_No"] = df["Batch_No"].astype(str)
    req.batch_numbers = [str(b) for b in req.batch_numbers]
    df_filtered = df[df["Batch_No"].isin(req.batch_numbers)]
    max_counter = df_filtered["Batch_Counter"].max()


    if df_filtered.empty:
        return JSONResponse(
            content={"error": "No data for selected batches"},
            status_code=400
        )

    pages = []
    # ====================================================================
    # CASE A: Multiple columns -> One page per batch
    # ====================================================================
    if len(req.columns) > 1:
        for batch in req.batch_numbers:
            batch_df = df_filtered[df_filtered["Batch_No"] == batch]
            if batch_df.empty:
                continue

            for col in req.columns:
                fig = go.Figure()

                fig.add_trace(
                    go.Scatter(
                        x=batch_df["Batch_Counter"],
                        y=batch_df[col],
                        mode="lines",
                        name=col
                    )
                )

            fig.update_layout(
                title=f"Batch Profile – Batch {batch} – {col}",
                xaxis_title="Batch Counter",
                yaxis_title=col,
                template="plotly_white",
                height=400 * len(req.columns)* len(req.batch_numbers), width=600, showlegend=False,
            )

            # Force x-axis to match all batches
            fig.update_xaxes(range=[1, max_counter])

            pages.append({
                "batch": batch,
                "type": "plot",
                "data": json.loads(fig.to_json())
            })

    # ====================================================================
    # CASE B: Single column -> Overlay all batches on one page
    # ====================================================================
    else:
        col = req.columns[0]
        fig = go.Figure()

        for batch in req.batch_numbers:
            batch_df = df_filtered[df_filtered["Batch_No"] == batch]
            if batch_df.empty:
                continue

            fig.add_trace(
                go.Scatter(
                    x=batch_df["Batch_Counter"],
                    y=batch_df[col],
                    mode="lines",
                    name=f"Batch {batch}"
                )
            )

        fig.update_layout(
            title=f"Batch Overlay – {batch} – {col}",
            xaxis_title="Batch Counter",
            yaxis_title=col,
            template="plotly_white",
            height=400 * len(req.columns)* len(req.batch_numbers), width=600, showlegend=False,
        )

        # Force x-axis to match all batches
        fig.update_xaxes(range=[1, max_counter])

        pages.append({
            "column": col,
            "type": "plot",
            "data": json.loads(fig.to_json())
        })

    # ====================================================================
    # RETURN RESULT
    # ====================================================================

    return JSONResponse(
        content={
            "pages": pages,
            "message": "Batch profiling successful",
            "num_pages": len(pages),
        }
    )


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
        print(f"[VISUALIZE_MISSING_DATA] Could not parse selected variable from prompt: {input_text}")
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
    fig.update_layout(title=f"Missing Data Visualization: '{selected_variable}' Over Time", xaxis_title='Date_time', yaxis_title=selected_variable, hovermode="x unified", height=500, width=700)
    return fig.to_json()

def visualize_missing_data_post_treatment(input_text):
    global augmented_df
    df = augmented_df
    if df is None:
        raise ValueError("No data exists for plot")
    match = re.search(r"selected variable is ['\"]?(.+?)['\"]?$", input_text, re.IGNORECASE)
    if not match:
        print(f"[VISUALIZE_MISSING_DATA] Could not parse selected variable from prompt: {input_text}")
        return "Could not find selected variable in prompt."
    selected_variable = match.group(1).strip()
    print(f"[Vizualize function] Selected variable parsed: {selected_variable}")
    if selected_variable not in df.columns:
        return f"Column '{selected_variable}' not in dataframe."
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date_time'], y=df[selected_variable], mode='lines+markers', name=selected_variable, line=dict(color='blue')))
    for start, end in get_missing_value_intervals(df, selected_variable):
        fig.add_vrect(x0=start, x1=end, fillcolor="orange", opacity=0.3, line_width=0)
    for start, end in get_missing_datetime_intervals(df):
        fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.2, line_width=0)
    fig.update_layout(title=f"Missing Data Visualization: '{selected_variable}' Over Time", xaxis_title='Date_time', yaxis_title=selected_variable, hovermode="x unified", height=500, width=700)
    return fig.to_json()

def get_outlier_intervals(df, column, method='z-score', datetime_col='Date_time', threshold=3):
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    method=method.lower()
    if method == 'zscore' or method == 'z-score':
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
    global augmented_df
    df = augmented_df if augmented_df is not None else uploaded_df
    if df is None:
        raise ValueError("No data uploaded yet.")
    match = re.search(r"selected variable is ['\"]?(.+?)['\"]?(?=\s+using method|$)", prompt, re.IGNORECASE)
    if not match:
        print(f"[VISUALIZE_OUTLIER_DATA] Could not parse selected variable from prompt: {prompt}")
        return "Could not find selected variable in prompt."
    column = match.group(1).strip()
    print(f"[Vizualize function] Selected variable parsed: {column}")
    if column not in df.columns:
        return f"Column '{column}' not in dataframe."
    method = "zscore"
    if "iqr" in prompt.lower():
        method = "iqr"
    if column not in df.columns:
        return f"Column '{column}' not found in data."
    if 'Date_time' not in df.columns:
        return "'Date_time' column is missing."

    intervals = get_outlier_intervals(df, column, method=method)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date_time'], y=df[column], mode='lines+markers', name=column, line=dict(color='blue')))
    for start, end in intervals:
        fig.add_vrect(x0=start, x1=end, fillcolor="purple", opacity=0.3, line_width=0)
    fig.update_layout(title=f"Outlier Analysis ({method.upper()}): '{column}'", xaxis_title='Date_time', yaxis_title=column, hovermode="x unified", height=500, width=700)
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

@app.get("/outlier_intervals")
def outlier_intervals(column: str, method: str):
    print(f"Column received: {column}")
    
    global augmented_df
    df = augmented_df if augmented_df is not None else uploaded_df
    if df is None:
        raise ValueError("No data uploaded yet.")
    if column not in df.columns:
        return JSONResponse(content={"error": f"Column '{column}' not found in data."}, status_code=400)

    intervals = get_outlier_intervals(df, column, method)
    formatted = [{"start": str(start), "end": str(end)} for start, end in intervals]
    return JSONResponse(content={"intervals": formatted})


@app.get("/get_columns")
def get_columns():
    global augmented_df, uploaded_df
    df = augmented_df if augmented_df is not None else uploaded_df
    if df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)
    if df is not None:
        return JSONResponse(content={"columns": df.columns.tolist()})
    return JSONResponse(content={"error": "No file uploaded"}, status_code=400)

@app.get("/get_batchnos")
def get_batchnos():
    global augmented_df, uploaded_df
    df = augmented_df if augmented_df is not None else uploaded_df
    if df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)
    if df is not None:
        if "Batch_No" not in df.columns:
            return JSONResponse(content={"error": "'Batch_No' column not found in the uploaded file"},
            status_code=400
        )
        # extract unique batch numbers (sorted, converted to Python types)
        unique_batches = df["Batch_No"].dropna().unique().tolist()
        return JSONResponse(content={"batch_numbers": unique_batches})
    
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
    return {"message": "Treatment applied successfully!", "columns": list(augmented_df.columns)}

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

    return {"message": "Treatment applied successfully!", "columns": list(augmented_df.columns)}

@app.post("/apply_outlier_treatment")
def apply_outlier_treatment(payload: dict):
    global augmented_df
    print("Shape of augmented_df at start of apply_outlier_treatment:", augmented_df.shape if 'augmented_df' in globals() else "augmented_df not found")

    column = payload.get("column")
    intervals = payload.get("intervals", [])
    method = payload.get("method")

    if not column or not intervals or not method:
        return JSONResponse(content={"error": "Invalid payload: column, intervals, and method are required."}, status_code=400)

    if augmented_df is None:
        return JSONResponse(content={"error": "Missing value treatment must be applied first."}, status_code=400)

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

    return {"message": "Treatment applied successfully!", "columns": list(augmented_df.columns)}

    
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
        return StreamingResponse(stream, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=latest_data.csv"})
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

#llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
#tools = [
 #   Tool(name="SummarizeData", func=summarize_data, description="Summarizes the uploaded manufacturing dataset"),
  #  Tool(name="VariabilityAnalysis", func=plot_variability_tool, description="Generates variability analysis plots. Format: selected variable is 'ColumnName'"),
   # Tool(name="MissingValueAnalysis", func=visualize_missing_data, description="Generates missing value analysis plots. Format: selected variable is 'ColumnName'")
#]
#agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

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

        if "post treatment missing value plot" in prompt_lower:
            result = visualize_missing_data_post_treatment(prompt)
            return JSONResponse(content={"type": "plot", "data": json.loads(result)})

        elif "outlier analysis" in prompt_lower:
            df = augmented_df if augmented_df is not None else uploaded_df
            if df is None:
                raise ValueError("No data uploaded yet.")
            result = visualize_outlier_data(prompt)
            print("[CHAT] Result type from visualize_outlier_data:", type(result))
            print("[CHAT] Raw result:", result)
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

########################## Exploratory data analysis tab ##########################################
@app.post("/eda/qcut_boxplot")
def qcut_boxplot(columns: list[str], target: str, quantiles: int ):
    """
    X-axis = target quantile labels (Q1..Qk) repeated per row
    Y-axis = values of each selected column
    One subplot per selected column (stacked)
    """
    global augmented_df, uploaded_df
    df = augmented_df if augmented_df is not None else uploaded_df
    if df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)

    df = df.copy()
    df = df.sort_values(by=target, ascending=True).reset_index(drop=True)

    try:
        # Create quantile bins
        df['quantile_bin'], bins = pd.qcut(df[target], q=quantiles, retbins=True, duplicates="drop")

        # Build labels like Q1: (50.2, 65.4]
        unique_bins = df['quantile_bin'].cat.categories
        bin_labels = [f"Q{i+1}: ({interval.left:.2f}, {interval.right:.2f}]" for i, interval in enumerate(unique_bins)]
        
        # Map each row’s bin to the combined label
        bin_mapping = {interval: label for interval, label in zip(unique_bins, bin_labels)}
        df['quantile_label'] = df['quantile_bin'].map(bin_mapping)
        df['quantile_label'] = pd.Categorical(df['quantile_label'],categories=bin_labels,  ordered=True) # preserves Q1, Q2, … order

        fig = make_subplots(rows=len(columns), cols=1, subplot_titles=columns, 
                            #shared_xaxes=True
                           )

        for i, col in enumerate(columns, start=1):
            fig.add_trace(
                go.Box(
                    x=df['quantile_label'],
                    y=df[col],
                    name=col,
                    boxmean="sd"
                ),
                row=i, col=1
            )

        fig.update_layout(
            title_text=f"Specialized Q-cut Box Plots (Target: {target}, Quantiles: {quantiles})",
            height=400 * len(columns), width=600, showlegend=False,
            xaxis_title=f"Quantile bins of {target}"
        )

        # Add x-axis title only for bottom subplot
        #fig.update_xaxes(title_text=f"Quantile bins of {target}", row=len(columns), col=1

        # Add x-axis title for each subplot
        #for i in range(1, len(columns) + 1):
         #   fig.update_xaxes(title_text=f"Quantile bins of {target}", row=i, col=1)

        for i in range(1, len(columns) + 1):
            fig.update_xaxes(
                tickmode="array",
                tickvals=bin_labels,   # categories in order
                ticktext=bin_labels,   # display labels as-is
                row=i, col=1)

        # ✅ Serialize safely
        #fig_dict = fig.to_plotly_json()

        # ===== Debugging Section =====
        fig_json = fig.to_json()
        fig_dict = json.loads(fig_json)
        #fig_dict = fig.to_dict()
        print("=== Backend Debug: Q-cut Box Plot ===")
        print("Figure type:", type(fig))
        print("Keys in figure dict:", fig_dict.keys())
        print("Number of traces:", len(fig_dict.get("data", [])))
        for idx, trace in enumerate(fig_dict.get("data", [])):
            print(f"Trace {idx}:")
            print("   type:", trace.get("type"))
            print("   name:", trace.get("name"))
            print("   x length:", len(trace.get("x", [])))
            print("   y length:", len(trace.get("y", [])))
        print("=== End of Debug ===")

        return JSONResponse(content={"type": "plot", "data": json.loads(fig.to_json())})
        #return JSONResponse(content={"type": "plot", "data": json.loads(fig.to_json())})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#---------------------------------------------------------------------------------------------#

class DualBoxPlotRequest(BaseModel):
    column_x: str
    column_y: str
    plot_type: str  # "auto" or "quantile"
    num_bins_quantiles: int | None = None

@app.post("/eda/dualaxes_boxplot")
def dualaxes_boxplot(req: DualBoxPlotRequest):
    try:
        global augmented_df, uploaded_df
        df = augmented_df if augmented_df is not None else uploaded_df
        if df is None:
            return JSONResponse(content={"error": "No data uploaded"}, status_code=400)
    
        col_x, col_y = req.column_x, req.column_y
        plot_type, num = req.plot_type, req.num_bins_quantiles
    
        df = df.copy()
        df = df.sort_values(by=col_x, ascending=True).reset_index(drop=True)
        
        if plot_type == "quantile":
            try:
                df["x_bin"] = pd.qcut(df[col_x], q=num, duplicates="drop")
                group_col = "x_bin"
            except Exception as e:
                return {"error": f"Quantile binning failed: {str(e)}"}
        else:
            # auto: use ranges of X (bin into 5 automatically)
            bins = num if num is not None else 5
            df["x_bin"] = pd.cut(df[col_x], bins=num)
            group_col = "x_bin"
        
        # Build labels like Q1: (50.2, 65.4]
        unique_bins = df["x_bin"].cat.categories
        bin_labels = [f"Q{i+1}: ({interval.left:.2f}, {interval.right:.2f}]" for i, interval in enumerate(unique_bins)]
                
        # Map each row’s bin to the combined label
        bin_mapping = {interval: label for interval, label in zip(unique_bins, bin_labels)}
        df["x_bin"] = df["x_bin"].map(bin_mapping)
        df["x_bin"] = pd.Categorical(df["x_bin"],categories=bin_labels,  ordered=True) # preserves Q1, Q2, … order
    
        fig = go.Figure()
        fig.add_trace(go.Box(
            x=df["x_bin"],
            y=df[col_y],
            name=col_y,
            boxmean="sd"))
    
        fig.update_layout(
            title_text=f"Dual Axes Box Plots (X: {col_x}, Quantiles: {num})",
            height=400, width=600, showlegend=False,
            xaxis_title=f"{col_x}"
        )
        # ===== Debugging Section =====
        fig_json = fig.to_json()
        fig_dict = json.loads(fig_json)
        print("=== Backend Debug: Dual Axes Box Plot ===")
        print("Figure type:", type(fig))
        print(fig_json)
        print("Keys in figure dict:", fig_dict.keys())
        print("Number of traces:", len(fig_dict.get("data", [])))
        for idx, trace in enumerate(fig_dict.get("data", [])):
            print(f"Trace {idx}:")
            print("   type:", trace.get("type"))
            print("   name:", trace.get("name"))
            print("   x length:", len(trace.get("x", [])))
            print("   y length:", len(trace.get("y", [])))
        print("=== End of Debug ===")
    
        return JSONResponse(content={"type": "plot", "data": json.loads(fig.to_json())})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#-----------------------------------------------------------------------------------------------------#
class CorrelationRequest(BaseModel):
    columns: list[str]
    method: str  # "pearson", "spearman", "kendall"

@app.post("/eda/correlation_analysis")
def correlation_analysis(req: CorrelationRequest):
    global augmented_df, uploaded_df
    df = augmented_df if augmented_df is not None else uploaded_df

    if df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)

    selected_cols = req.columns
    method = req.method.lower()

    if method not in ["pearson", "spearman", "kendall"]:
        return JSONResponse(content={"error": "Invalid correlation method"}, status_code=400)

    try:
        # Keep only numeric selected cols
        df_selected = df[selected_cols].select_dtypes(include="number")
        corr_matrix = df_selected.corr(method=method)
    except Exception as e:
        return JSONResponse(content={"error": f"Correlation failed: {str(e)}"}, status_code=500)

    # Plotly Heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation"),
        )
    )

    # Add title and layout
    fig.update_layout(
        title=f"{method.capitalize()} Correlation Matrix",
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
        width=600,
        height=600,
    )

    # ===== Debugging Section =====
    fig_json = fig.to_json()
    fig_dict = json.loads(fig_json)
    print("=== Backend Debug: Correlation Analysis ===")
    print("Figure type:", type(fig))
    print(fig_json)
    print("Keys in figure dict:", fig_dict.keys())
    print("Number of traces:", len(fig_dict.get("data", [])))
    for idx, trace in enumerate(fig_dict.get("data", [])):
        print(f"Trace {idx}:")
        print("   type:", trace.get("type"))
        print("   name:", trace.get("name"))
    print("=== End of Debug ===")

    return JSONResponse(
        content={
            "type": "plot",
            "data": json.loads(fig.to_json()),
            "matrix": corr_matrix.round(2).to_dict()
        }
    )
#-----------------------------------------------------------------------------------------------------------#
class ContinuousRangeRequest(BaseModel):
    target: str
    min_duration: int
    lower_pct: float
    upper_pct: float
    max_break: int
    
@app.post("/eda/continuous_range")
def continuous_range_analysis(req: ContinuousRangeRequest):
    """
    Detect continuous ranges where target stays within [start*(1-lower_pct), start*(1+upper_pct)]
    for at least `min_duration` minutes, allowing gaps of up to `max_break` minutes.
    """
    try:
        global augmented_df, uploaded_df, last_continuous_ranges
        df = augmented_df if augmented_df is not None else uploaded_df
        if df is None:
            return JSONResponse(content={"error": "No data uploaded"}, status_code=400)

        target, min_duration = req.target, req.min_duration
        lower_pct, upper_pct = req.lower_pct, req.upper_pct
        max_break=req.max_break

        # Assume default time column
        datetime_col='Date_time'
        if datetime_col not in df.columns:
            return JSONResponse(content={"error": f"Default time column '{datetime_col}' not found in dataset"}, status_code=400)

        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.sort_values(datetime_col).reset_index(drop=True)

        # Estimate base frequency
        diffs = df[datetime_col].diff().dropna()
        inferred_freq = diffs.mode().iloc[0] if not diffs.empty else pd.Timedelta(hours=1)

        continuous_ranges = []
        in_range = False
        start_time, start_val = None, None
        lower, upper = None, None
        out_of_range_duration = pd.Timedelta(0)

        for i in range(len(df)):
            t = df.loc[i, datetime_col]
            val = df.loc[i, target]

            #start a new continuous range
            if not in_range:
                if not pd.isna(val):
                    start_time = t
                    start_val = val
                    lower = start_val * (1 - lower_pct)
                    upper = start_val * (1 + upper_pct)
                    in_range = True
                    out_of_range_duration = pd.Timedelta(0)
                continue

            #If value goes out of range
            if pd.isna(val):
                step = (t - df.loc[i - 1, datetime_col]) if i > 0 else inferred_freq
                out_of_range_duration += step
            elif lower <= val <= upper:
                out_of_range_duration = pd.Timedelta(0)
            else:
                step = (t - df.loc[i - 1, datetime_col]) if i > 0 else freq
                out_of_range_duration += step

            # Break condition
            if out_of_range_duration >= pd.Timedelta(minutes=max_break):
                end_time = df.loc[i - 1, datetime_col]
                if (end_time - start_time) >= pd.Timedelta(minutes=min_duration):
                    continuous_ranges.append({
                        "start": start_time.isoformat(),
                        "end": end_time.isoformat(),
                        "duration_min": float((end_time - start_time).total_seconds() / 60),
                        "start_value": float(start_val),
                        "lower": float(lower),
                        "upper": float(upper),
                    })
                in_range = False
                start_time, start_val, lower, upper = None, None, None, None
                out_of_range_duration = pd.Timedelta(0)

        # Handle unfinished range at end
        if in_range and start_time is not None:
            end_time = df[datetime_col].iloc[-1]
            if (end_time - start_time) >= pd.Timedelta(minutes=min_duration):
                continuous_ranges.append({
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_min": float((end_time - start_time).total_seconds() / 60),
                    "start_value": float(start_val),
                    "lower": float(lower),
                    "upper": float(upper),
                })

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[datetime_col], y=df[target],
            mode="lines", name=target
        ))

        for i, r in enumerate(continuous_ranges):
            fig.add_vrect(
                x0=r["start"], x1=r["end"],
                fillcolor="green", opacity=0.2,
                line_width=0,
                annotation_text=f"Range {i+1}"
            )

        fig.update_layout(title=f"Continuous Range Analysis: '{target}' Over Time", 
                          xaxis_title='Date_time', yaxis_title=target, hovermode="x unified", height=500, width=700)

        last_continuous_ranges = continuous_ranges

        return JSONResponse(content={
            "type": "plot",
            "data": json.loads(fig.to_json()),
            "ranges": continuous_ranges
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#--------------------------------------------------------------------------------------------------------#
class MultivariateRequestWithRanges(BaseModel):
    columns: list[str]              # selected feature columns
    mode: str                       # "Boxplot" or "Timeseries"
    numMultiRanges: int             # number of top and bottom ranges
    performanceDirection: str       # "higher" or "lower"
    target: str                     # target column

@app.post("/eda/multivariate")
def multivariate_analysis_with_ranges(req: MultivariateRequestWithRanges):
    """
    Build multivariate plots using precomputed continuous ranges.
    """
    try:
        global augmented_df, uploaded_df, last_continuous_ranges
        df = augmented_df if augmented_df is not None else uploaded_df
        if df is None:
            return JSONResponse(content={"error": "No data uploaded"}, status_code=400)

        # validations
        if req.target not in df.columns:
            return JSONResponse(content={"error": f"Target column '{req.target}' not found"}, status_code=400)
        if not req.columns:
            return JSONResponse(content={"error": "No columns provided"}, status_code=400)
        missing_cols = [c for c in req.columns if c not in df.columns]
        if missing_cols:
            return JSONResponse(content={"error": f"Selected columns not in dataset: {missing_cols}"}, status_code=400)
        # Assume default time column
        datetime_col='Date_time'
        if datetime_col not in df.columns:
            return JSONResponse(content={"error": f"Default time column '{datetime_col}' not found in dataset"}, status_code=400)

        df = df.copy()
        df[datetime_col] = pd.to_datetime(df[datetime_col])

        if last_continuous_ranges is None:
            return JSONResponse(
                content={"error": "Continuous ranges not found. Please run continuous range analysis first."},
                status_code=400
        )

        # --- compute median target per range ---
        groups = []
        for i, r in enumerate(last_continuous_ranges):
            st = pd.to_datetime(r["start"])
            en = pd.to_datetime(r["end"])
            mask = (df[datetime_col] >= st) & (df[datetime_col] <= en)
            slice_df = df.loc[mask].copy()
            median_target = slice_df[req.target].median(skipna=True) if not slice_df.empty else np.nan
            groups.append({
                "group_id": i,
                "start": st,
                "end": en,
                "duration_min": r.get("duration_min", float((en - st).total_seconds()/60)),
                "median_target": float(median_target) if pd.notna(median_target) else None,
                "df_slice": slice_df
            })

        # filter valid groups
        groups = [g for g in groups if g["median_target"] is not None]
        if not groups:
            return JSONResponse(content={"error": "No valid continuous ranges with median target found"}, status_code=400)

        # sort by median target
        groups_sorted = sorted(groups, key=lambda x: x["median_target"])

        # select top/bottom ranges based on performance direction
        num = max(1, req.numMultiRanges)
        if req.performanceDirection == "higher":
            good_groups = sorted(groups_sorted, key=lambda x: x["median_target"], reverse=True)[:num]
            bad_groups = groups_sorted[:num]
        else:
            good_groups = groups_sorted[:num]
            bad_groups = sorted(groups_sorted, key=lambda x: x["median_target"], reverse=True)[:num]

        good_ids = set([g["group_id"] for g in good_groups])
        bad_ids = set([g["group_id"] for g in bad_groups])
        selected_ids = list(good_ids.union(bad_ids))
        sel_groups_meta = [g for g in groups_sorted if g["group_id"] in selected_ids]
        sel_groups_meta = sorted(sel_groups_meta, key=lambda x: x["median_target"])

        # labels & colors
        def format_label(g):
            st, en = g["start"], g["end"]
            dur = int((en - st).total_seconds())
            if dur >= 3600:
                dur_label = f"{dur//3600}h{(dur%3600)//60}m"
            elif dur >= 60:
                dur_label = f"{dur//60}m"
            else:
                dur_label = f"{dur}s"
            return f"{st.strftime('%Y-%m-%d %H:%M')} / {dur_label}"

        x_labels = [format_label(g) for g in sel_groups_meta]
        x_group_ids = [g["group_id"] for g in sel_groups_meta]
        classification = {}
        for gid in x_group_ids:
            if gid in good_ids and gid in bad_ids:
                classification[gid] = "neutral"
            elif gid in good_ids:
                classification[gid] = "good"
            elif gid in bad_ids:
                classification[gid] = "bad"
            else:
                classification[gid] = "neutral"
        color_map = {"good": "green", "bad": "red", "neutral": "gray"}
        gid_to_slice = {g["group_id"]: g["df_slice"] for g in sel_groups_meta}

        # --- build subplots ---
        n_rows = 1 + len(req.columns)
        subplot_titles = ["Target: " + req.target] + req.columns
        fig = make_subplots(rows=n_rows, cols=1, shared_xaxes=True,
                            vertical_spacing=0.06, subplot_titles=subplot_titles)

        if req.mode.lower().startswith("box"):
            for row_idx, col in enumerate([req.target] + req.columns, start=1):
                for gid, xlab in zip(x_group_ids, x_labels):
                    slice_df = gid_to_slice.get(gid)
                    if slice_df is None or slice_df.empty or col not in slice_df.columns:
                        continue
                    y = slice_df[col].dropna().values
                    if len(y) == 0:
                        continue
                    fig.add_trace(
                        go.Box(
                            y=y,
                            x=[xlab] * len(y),
                            name=str(xlab),
                            marker=dict(color=color_map.get(classification.get(gid, "neutral"))),
                            boxmean=True,
                            showlegend=False
                        ),
                        row=row_idx, col=1
                    )
        elif req.mode.lower().startswith("time"):
            for row_idx, col in enumerate([req.target] + req.columns, start=1):
                for gid in x_group_ids:
                    slice_df = gid_to_slice.get(gid)
                    if slice_df is None or slice_df.empty or col not in slice_df.columns:
                        continue
                    # Ensure sorted by datetime
                    slice_df = slice_df.sort_values(by=datetime_col)
                    x_series = slice_df[datetime_col].tolist()
                    y_series = slice_df[col].tolist()

                    if len(x_series) == 0:
                        continue
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_series,
                            y=y_series,
                            mode="lines+markers",
                            name=col,
                            marker=dict(color=color_map.get(classification.get(gid, "neutral"))),
                            showlegend=False
                        ),
                        row=row_idx, col=1
                    )
        else:
            return JSONResponse(content={"error": f"Unsupported mode '{req.mode}'"}, status_code=400)

        fig.update_layout(height=220*n_rows, margin=dict(t=70, b=120, l=80, r=20))
        fig.update_xaxes(tickangle=-45)

        return JSONResponse(content={
            "type": "plot",
            "data": json.loads(fig.to_json()),
            "groups": [
                {
                    "group_id": g["group_id"],
                    "start": g["start"].isoformat(),
                    "end": g["end"].isoformat(),
                    "duration_min": g["duration_min"],
                    "median_target": g["median_target"]
                } for g in sel_groups_meta
            ]
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#######################################----Feature Engineering tab-------------###########################
@app.get("/get_augmented_df_columns")
def get_augmented_df_columns():
    global augmented_df, uploaded_df
    df = augmented_df if augmented_df is not None else uploaded_df
    if df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)
    if df is not None:
        return JSONResponse(content={"columns": df.columns.tolist()})
    return JSONResponse(content={"error": "No file uploaded"}, status_code=400)


class CustomFeatureRequest(BaseModel):
    column1: str | None = None
    column2: str | None = None
    column3: str | None = None
    featureInputs: dict

@app.post("/eda/custom_feature")
def custom_feature(req: CustomFeatureRequest):
    try:
        global augmented_df, uploaded_df, features_raw_df
        df = augmented_df if augmented_df is not None else uploaded_df
        if df is None:
            return JSONResponse(content={"error": "No data uploaded"}, status_code=400)

        df = df.copy()
        errors = []

        col1, col2, col3 = req.column1, req.column2, req.column3
        f = req.featureInputs

        # --- Build formula string exactly like frontend ---
        formula = ""

        if col1:
            formula += f.get("beforeCol1", "") + f"({col1})" if f.get("beforeCol1") else col1

        if col2:
            formula += f" {f.get('op12','')} " if f.get("op12") else " "
            formula += f"{f.get('between1and2','')}({col2})" if f.get("between1and2") else col2

        if col3:
            formula += f" {f.get('op23','')} " if f.get("op23") else " "
            formula += f"{f.get('between2and3','')}({col3})" if f.get("between2and3") else col3

        formula = formula.strip()

        # --- Apply formula row by row ---
        new_feature_values = []
        for idx, row in df.iterrows():
            try:
                local_env = {
                    col1: row[col1] if col1 else None,
                    col2: row[col2] if col2 else None,
                    col3: row[col3] if col3 else None,
                    "log": np.log,
                    "sqrt": np.sqrt
                }
                val = eval(formula, {"__builtins__": {}}, local_env)
                new_feature_values.append(val)
            except Exception:
                errors.append(idx)
                new_feature_values.append(None)

# --- Generate a unique feature name ---
        op_map = {"+": "plus", "-": "minus", "*": "times", "/": "div"}
        parts = []
        if col1:
            parts.append(col1)
        if col2 and f.get("op12"):
            parts.append(op_map.get(f.get("op12"), f.get("op12")))
            parts.append(col2)
        if col3 and f.get("op23"):
            parts.append(op_map.get(f.get("op23"), f.get("op23")))
            parts.append(col3)

        base_name = "custom_" + "_".join(parts) if parts else "feature_custom"
        # sanitize name
        base_name = re.sub(r'[^0-9a-zA-Z_]', "_", base_name)

        feature_name = base_name
        counter = 1
        while feature_name in df.columns:
            feature_name = f"{base_name}_{counter}"
            counter += 1

        # --- Assign new feature to DataFrame ---
        df[feature_name] = new_feature_values
        #features_raw_df = df  # update global
        augmented_df = df    # keep augmented version updated

        if len(errors) == 0:
            message = f"Feature '{feature_name}' created successfully with no errors."
        elif len(errors) < len(df):
            message = f"Feature '{feature_name}' created with {len(errors)} row errors."
        else:
            message = f"Feature '{feature_name}' could not be created. All rows failed."

        return {
            "success": True if len(errors) ==0 else False,
            "new_column": feature_name,
            "message": message,
            "errors": errors
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


class feature_variabilityRequest(BaseModel):
    selectedFeature: str | None = None

@app.post("/eda/feature_variability")
def feature_variability(req: feature_variabilityRequest):
    try:
        global augmented_df, uploaded_df
        df = augmented_df if augmented_df is not None else uploaded_df
        if df is None:
            return JSONResponse(content={"error": "No data uploaded"}, status_code=400)

        selected_variable=req.selectedFeature

        fig = make_subplots(rows=2, cols=2, subplot_titles=['Box Plot', 'Line Plot', 'Histogram'],
                        row_heights=[0.5, 0.5], column_widths=[0.5, 0.5])
        fig.add_trace(go.Box(y=df[selected_variable], name='Box Plot'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date_time'], y=df[selected_variable], mode='lines', name='Line Plot'), row=1, col=2)
        fig.add_trace(go.Histogram(x=df[selected_variable], name='Histogram'), row=2, col=1)

        fig.update_layout(title_text=f'Feature Variability Analysis - {selected_variable}', showlegend=False, height=800, width=1500)
        # ===== Debugging Section =====
        fig_json = fig.to_json()
        fig_dict = json.loads(fig_json)
        print("=== Backend Debug: Feature Variability Analysis ===")
        print("Figure type:", type(fig))
        print(fig_json)
        print("Keys in figure dict:", fig_dict.keys())
        print("Number of traces:", len(fig_dict.get("data", [])))
        for idx, trace in enumerate(fig_dict.get("data", [])):
            print(f"Trace {idx}:")
            print("   type:", trace.get("type"))
            print("   name:", trace.get("name"))
            print("   x length:", len(trace.get("x", [])))
            print("   y length:", len(trace.get("y", [])))
        print("=== End of Debug ===")
    
        return JSONResponse(content={"type": "plot", "data": json.loads(fig.to_json())})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


class feature_missingRequest(BaseModel):
    selectedFeature: str | None = None

@app.post("/eda/feature_missing")
def feature_missing(req: feature_missingRequest):
    try:
        global augmented_df, uploaded_df
        df = augmented_df if augmented_df is not None else uploaded_df
        if df is None:
            return JSONResponse(content={"error": "No data uploaded"}, status_code=400)

        selected_variable=req.selectedFeature
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date_time'], y=df[selected_variable], mode='lines+markers', name=selected_variable, line=dict(color='blue')))
        for start, end in get_missing_value_intervals(df, selected_variable):
            fig.add_vrect(x0=start, x1=end, fillcolor="orange", opacity=0.3, line_width=0)
        for start, end in get_missing_datetime_intervals(df):
            fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.2, line_width=0)
        fig.update_layout(title=f"Missing Data Visualization: '{selected_variable}' Over Time", xaxis_title='Date_time', yaxis_title=selected_variable, hovermode="x unified", height=500, width=700)
        
        # ===== Debugging Section =====
        fig_json = fig.to_json()
        fig_dict = json.loads(fig_json)
        print("=== Backend Debug: Feature Missing Value Analysis ===")
        print("Figure type:", type(fig))
        print(fig_json)
        print("Keys in figure dict:", fig_dict.keys())
        print("Number of traces:", len(fig_dict.get("data", [])))
        for idx, trace in enumerate(fig_dict.get("data", [])):
            print(f"Trace {idx}:")
            print("   type:", trace.get("type"))
            print("   name:", trace.get("name"))
            print("   x length:", len(trace.get("x", [])))
            print("   y length:", len(trace.get("y", [])))
        print("=== End of Debug ===")
    
        return JSONResponse(content={"type": "plot", "data": json.loads(fig.to_json())})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

class feature_outlierRequest(BaseModel):
    selectedFeature: str | None = None
    method: str | None = None

@app.post("/eda/feature_outlier")
def feature_outlier(req: feature_outlierRequest):
    try:
        global augmented_df, uploaded_df
        df = augmented_df if augmented_df is not None else uploaded_df
        if df is None:
            return JSONResponse(content={"error": "No data uploaded"}, status_code=400)

        selected_variable=req.selectedFeature
        method=req.method
        intervals = get_outlier_intervals(df, selected_variable, method=method)
        print(intervals)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date_time'], y=df[selected_variable], mode='lines+markers', name=selected_variable, line=dict(color='blue')))
        for start, end in intervals:
            fig.add_vrect(x0=start, x1=end, fillcolor="purple", opacity=0.3, line_width=0)
        fig.update_layout(title=f"Outlier Analysis ({method.upper()}): '{selected_variable}'", xaxis_title='Date_time', yaxis_title=column, hovermode="x unified", height=500, width=700)
        print("[CHAT] Result type from visualize_outlier_data:", type(fig.to_json()))
        print("[CHAT] Raw result:", fig.to_json())
        # ===== Debugging Section =====
        fig_json = fig.to_json()
        fig_dict = json.loads(fig_json)
        print("=== Backend Debug: Feature Outlier Analysis ===")
        print("Figure type:", type(fig))
        print(fig_json)
        print("Keys in figure dict:", fig_dict.keys())
        print("Number of traces:", len(fig_dict.get("data", [])))
        for idx, trace in enumerate(fig_dict.get("data", [])):
            print(f"Trace {idx}:")
            print("   type:", trace.get("type"))
            print("   name:", trace.get("name"))
            print("   x length:", len(trace.get("x", [])))
            print("   y length:", len(trace.get("y", [])))
        print("=== End of Debug ===")
    
        return JSONResponse(content={"type": "plot", "data": json.loads(fig.to_json())})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

##############################------Model Development------------##################################
# Global dataframes (already loaded elsewhere in your app)
uploaded_df = None
augmented_df = None
last_trained_model = None
last_X_train = None
last_y_train = None
last_target_name = None
last_features = None

class TrainModelRequest(BaseModel):
    target: str
    performanceDirection: str = "maximize"  # default
    features: list[str]
    trainTestOption: str = "random"  # default
    splitPercent: float = 70.0
    startDate: str | None = None
    endDate: str | None = None
    modelType: str = "DecisionTree"  # default

@app.post("/train_model")
def train_model(req: TrainModelRequest):
    """
    Train ML model based on provided configuration or defaults.
    Supports regression use case (manufacturing KPI prediction).
    """
    try:
        global augmented_df, uploaded_df
        df = augmented_df if augmented_df is not None else uploaded_df
        if df is None:
            return JSONResponse(content={"error": "No dataset uploaded."}, status_code=400)

        # --- Validate columns ---
        missing_cols = [c for c in [req.target] + req.features if c not in df.columns]
        if missing_cols:
            return JSONResponse(
                content={"error": f"Missing columns in dataset: {missing_cols}"}, status_code=400
            )

        # --- Prepare data ---
        df = df.copy()
        df.dropna(subset=[req.target] + req.features, inplace=True)
        X = df[req.features]
        y = df[req.target]

        # --- Handle train-test split ---
        if req.trainTestOption in ["random", None, ""]:
            test_size = (100 - req.splitPercent) / 100 if req.splitPercent else 0.3
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, shuffle=True
            )

        elif req.trainTestOption == "time_percent":
            if "Date_time" not in df.columns:
                return JSONResponse(content={"error": "Date_time column not found for time-based split."}, status_code=400)
            df_sorted = df.sort_values(by="Date_time").reset_index(drop=True)
            split_index = int(len(df_sorted) * (req.splitPercent / 100))
            train_df, test_df = df_sorted.iloc[:split_index], df_sorted.iloc[split_index:]
            X_train, y_train = train_df[req.features], train_df[req.target]
            X_test, y_test = test_df[req.features], test_df[req.target]

        elif req.trainTestOption == "time_custom":
            if "Date_time" not in df.columns:
                return JSONResponse(content={"error": "Date_time column not found for time-based split."}, status_code=400)
            df_sorted = df.sort_values(by="Date_time").reset_index(drop=True)
            start = pd.to_datetime(req.startDate)
            end = pd.to_datetime(req.endDate)
            train_mask = (df_sorted["Date_time"] >= start) & (df_sorted["Date_time"] <= end)
            train_df = df_sorted.loc[train_mask]
            test_df = df_sorted.loc[~train_mask]
            X_train, y_train = train_df[req.features], train_df[req.target]
            X_test, y_test = test_df[req.features], test_df[req.target]

        else:
            return JSONResponse(content={"error": f"Invalid train/test option: {req.trainTestOption}"}, status_code=400)

        # --- Select and train model ---
        model_type = req.modelType.lower()
        if model_type in ["xgboost", "xgb"]:
            model = XGBRegressor(random_state=42)
        elif model_type in ["lightgbm", "lgb", "lgbm"]:
            model = LGBMRegressor(random_state=42)
        elif model_type in ["randomforest", "rf"]:
            model = RandomForestRegressor(random_state=42)
        else:
            model = DecisionTreeRegressor(random_state=42)  # Default

        # --- Train model ---
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # ✅ Save model and training data globally
        global last_trained_model, last_X_train, last_y_train, last_target, last_features, last_performance_direction
        last_trained_model = model
        last_X_train = X_train
        last_y_train = y_train
        last_target = req.target
        last_features = req.features
        last_performance_direction = req.performanceDirection

        # --- Evaluate model ---
        def calc_metrics(y_true, y_pred):
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
            return {
                "R2": round(r2, 3),
                "RMSE": round(rmse, 3),
                "MAE": round(mae, 3),
                "MAPE": round(mape, 2),
            }

        metrics_train = calc_metrics(y_train, y_pred_train)
        metrics_test = calc_metrics(y_test, y_pred_test)
        metrics_train["PerformanceDirection"] = req.performanceDirection
        metrics_test["PerformanceDirection"] = req.performanceDirection

        # --- Plot Predicted vs Actual (Train) ---
        fig_train = go.Figure()
        fig_train.add_trace(go.Scatter(x=y_train, y=y_pred_train, mode="markers", name="Train Predictions"))
        fig_train.add_trace(go.Scatter(x=y_train, y=y_train, mode="lines", name="Ideal Fit", line=dict(color="red")))
        fig_train.update_layout(
            title="Predicted vs Actual (Train Data)",
            xaxis_title="Actual Target",
            yaxis_title="Predicted Target",
            height=500,
            width=700,
        )

        # --- Plot Predicted vs Actual (Test) ---
        fig_test = go.Figure()
        fig_test.add_trace(go.Scatter(x=y_test, y=y_pred_test, mode="markers", name="Test Predictions"))
        fig_test.add_trace(go.Scatter(x=y_test, y=y_test, mode="lines", name="Ideal Fit", line=dict(color="red")))
        fig_test.update_layout(
            title="Predicted vs Actual (Test Data)",
            xaxis_title="Actual Target",
            yaxis_title="Predicted Target",
            height=500,
            width=700,
        )

        return JSONResponse(
            content={
                "success": True,
                "model_name": req.modelType,
                "metrics_train": metrics_train,
                "metrics_test": metrics_test,
                "plot_train": json.loads(fig_train.to_json()),
                "plot_test": json.loads(fig_test.to_json()),
            }
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/feature_importance")
def shap_feature_importance():
    """
    Generate SHAP-based feature importance plot from last trained model.
    """
    try:
        global last_trained_model, last_X_train, last_y_train, last_target_name, last_features
        print("🔍 [INFO] Entered /eda/feature_importance route")

        # Validate preconditions
        if last_trained_model is None or last_X_train is None:
            print("❌ [ERROR] No trained model or X_train found")
            return JSONResponse(
                content={"error": "No trained model found. Please train a model first."},
                status_code=400
            )

        model = last_trained_model
        X = last_X_train
        print(f"✅ Model type: {type(model).__name__}")
        print(f"✅ Training dataset shape: {X.shape}")

        # --- Compute SHAP values ---
        try:
            model_name = type(model).__name__.lower()
            print(f"🧠 Using SHAP for model type: {model_name}")
        
            if "lgbm" in model_name:
                explainer = shap.Explainer(model, X)  # LightGBM safe
                shap_values = explainer(X)
                shap_matrix = shap_values.values if hasattr(shap_values, "values") else shap_values
            elif "xgb" in model_name:
                # XGBoost specific fix: use booster directly
                explainer = shap.TreeExplainer(model.get_booster())
                shap_values = explainer.shap_values(X)
                shap_matrix = shap_values
            else:
                # Default Tree-based models (DecisionTree, RandomForest)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                shap_matrix = shap_values

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            shap_importance = sorted(zip(X.columns, mean_abs_shap),key=lambda x: x[1],reverse=True)

            print(f"✅ SHAP computed successfully with shape {np.shape(shap_matrix)}")
            
        except Exception as e:
            print("⚠️ SHAP computation failed — falling back to model feature_importances_")
            traceback.print_exc()
        
            if hasattr(model, "feature_importances_"):
                shap_matrix = None
                mean_abs_shap = model.feature_importances_
                shap_importance = sorted(
                    zip(X.columns, mean_abs_shap), key=lambda x: x[1], reverse=True
                )
                print("✅ Used model.feature_importances_ for fallback importance")
            else:
                raise RuntimeError(f"SHAP failed and model has no feature_importances_: {str(e)}")


        # --- Plot feature importance ---
        features = [x[0] for x in shap_importance[:15]]
        importances = [x[1] for x in shap_importance[:15]]

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=importances[::-1],
                y=features[::-1],
                orientation="h",
                marker=dict(color="green"),
                name="Mean |SHAP| Value"
            )
        )
        fig.update_layout(
            title="SHAP Feature Importance",
            xaxis_title="Mean |SHAP| Value",
            yaxis_title="Feature",
            height=600,
            margin=dict(t=60, b=40, l=80, r=40)
        )
        print("✅ SHAP feature importance plot created successfully")
        
        return JSONResponse(content={
            "type": "plot",
            "plot": json.loads(fig.to_json()),
            "feature_importance": {feat: float(val) for feat, val in shap_importance},
            "top_features": [feat for feat, _ in shap_importance[:10]]
        })


    except Exception as e:
        print("❌ [EXCEPTION] SHAP Feature Importance route failed")
        return JSONResponse(content={"error": f"Internal error: {str(e)}"}, status_code=500)

# 🔹 Route : SHAP Dependence / Optimal Ranges
# ----------------------------
@app.get("/optimal_ranges")
def shap_dependence_plots():
    """
    Generate SHAP dependence plots for top features (Optimal Operating Ranges).
    """
    try:
        global last_trained_model, last_X_train, last_y_train, last_target_name, last_features

        if last_trained_model is None or last_X_train is None:
            return JSONResponse(
                content={"error": "No trained model found. Please train a model first."},
                status_code=400
            )

        model = last_trained_model
        X = last_X_train

        # --- Compute SHAP values ---
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = sorted(
            zip(X.columns, mean_abs_shap), key=lambda x: x[1], reverse=True
        )

        # --- Take top 4 important features for dependence plots ---
        top_features = [x[0] for x in shap_importance[:4]]

        fig = make_subplots(
            rows=len(top_features),
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.1,
            subplot_titles=[f"SHAP Dependence: {f}" for f in top_features]
        )

        # --- Create SHAP dependence scatter for each top feature ---
        for i, feature in enumerate(top_features):
            fig.add_trace(
                go.Scatter(
                    x=X[feature],
                    y=shap_values[:, i],
                    mode="markers",
                    marker=dict(
                        color=X[feature],
                        colorscale="Viridis",
                        showscale=True,
                        size=6,
                        opacity=0.7
                    ),
                    name=feature
                ),
                row=i + 1, col=1
            )

        fig.update_layout(
            title="SHAP Dependence / Optimal Operating Ranges",
            height=300 * len(top_features),
            showlegend=False,
            margin=dict(t=80, b=60, l=80, r=40)
        )

        return JSONResponse(content={
            "type": "plot",
            "plot": json.loads(fig.to_json())
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

###################--------------Adding LLMs for interpretation-------------#############
from schemas import SHAPSummaryContext, SHAPDependenceContext
from interpreter_services import (
    interpret_shap_summary,
    interpret_shap_dependence
)


@app.post("/interpret_shap_summary")
def interpret_shap_summary_api(ctx: SHAPSummaryContext):
    try:
        result = interpret_shap_summary(ctx.dict())

        return {
            "insight": result.get("insight", result.get("raw_response", "Model did not return 'insight'.")),
            "confidence": result.get("confidence", 0.0),
            "suggested_next_steps": result.get("suggested_next_steps", []),
            "raw_output": result
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP summary interpretation failed: {str(e)}")


@app.post("/interpret_shap_dependence")
def interpret_shap_dependence_api(ctx: SHAPDependenceContext):
    try:
        result = interpret_shap_dependence(ctx.dict())
        return {
            "insight": result.get("insight", result.get("raw_response", "Model did not return 'insight'.")),
            "confidence": result.get("confidence", 0.0),
            "suggested_next_steps": result.get("suggested_next_steps", []),
            "raw_output": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP dependence interpretation failed: {str(e)}")
