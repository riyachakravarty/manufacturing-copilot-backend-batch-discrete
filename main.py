from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import io
import os
import re
import json
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_df = None

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
    match = re.search(r"selected variable is ['\"](.+?)['\"]", input_text)
    if not match:
        return "Could not find selected variable in prompt."
    selected_variable = match.group(1)
    return plot_variability_analysis_combined(selected_variable)

def get_missing_datetime_intervals(df, datetime_col='Date_time'):
    if datetime_col not in df.columns:
        return []
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    inferred_freq = pd.infer_freq(df[datetime_col]) or '1min'
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
    print("[DEBUG] visualize_missing_data() called")

    try:
        if uploaded_df is None:
            print("[ERROR] No uploaded_df found.")
            return "No data uploaded."

        match = re.search(r"selected variable is ['\"](.+?)['\"]", input_text)
        if not match:
            print("[ERROR] Could not extract variable from prompt.")
            return "Could not find selected variable in prompt."

        selected_variable = match.group(1)
        print(f"[DEBUG] Selected variable: {selected_variable}")

        if selected_variable not in uploaded_df.columns:
            print(f"[ERROR] Column '{selected_variable}' not found in dataframe.")
            return f"Column '{selected_variable}' not in dataframe."

        if 'Date_time' not in uploaded_df.columns:
            print("[ERROR] 'Date_time' column not found.")
            return "'Date_time' column missing in uploaded data."

        fig = go.Figure()

        # Plot actual data
        fig.add_trace(go.Scatter(
            x=uploaded_df['Date_time'],
            y=uploaded_df[selected_variable],
            mode='lines+markers',
            name=selected_variable,
            line=dict(color='blue')
        ))

        # Highlight Missing Values in Selected Variable
        #print("[DEBUG] Calculating missing value intervals...")
        #missing_val_intervals = get_missing_value_intervals(uploaded_df, selected_variable)
        #print(f"[DEBUG] Found {len(missing_val_intervals)} missing value intervals.")
        #for start, end in missing_val_intervals:
         #   fig.add_vrect(
          #      x0=start, x1=end,
           #     fillcolor="orange", opacity=0.3, line_width=0,
            #    annotation_text="Missing value", annotation_position="top left"
            #)

        # Highlight Missing Timestamps (Datetime Gaps)
        #print("[DEBUG] Calculating missing datetime intervals...")
        #missing_dt_intervals = get_missing_datetime_intervals(uploaded_df, 'Date_time')
        #print(f"[DEBUG] Found {len(missing_dt_intervals)} missing datetime intervals.")
        #for start, end in missing_dt_intervals:
         #   fig.add_vrect(
          #      x0=start, x1=end,
           #     fillcolor="red", opacity=0.2, line_width=0,
            #    annotation_text="Missing datetime", annotation_position="top right"
            #)

        fig.update_layout(
            title=f"Missing Data Visualization: '{selected_variable}' Over Time",
            xaxis_title='Date_time',
            yaxis_title=selected_variable,
            hovermode="x unified",
            height=600
        )

        print("[DEBUG] Finished creating figure.")
        return fig.to_json()

    except Exception as e:
        print(f"[ERROR] visualize_missing_data() failed: {e}")
        return "Error generating missing data plot."

@app.get("/missing_datetime_intervals")
def missing_datetime_intervals():
    global uploaded_df
    if uploaded_df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)
    intervals = get_missing_datetime_intervals(uploaded_df, 'Date_time')
    formatted = [{"start": str(start), "end": str(end)} for start, end in intervals]
    return JSONResponse(content={"intervals": formatted})

@app.get("/get_columns")
def get_columns():
    global uploaded_df
    if uploaded_df is not None:
        return JSONResponse(content={"columns": uploaded_df.columns.tolist()})
    else:
        return JSONResponse(content={"error": "No file uploaded"}, status_code=400)

@app.post("/apply_treatment")
def apply_treatment(req: TreatmentRequest):
    global uploaded_df
    if uploaded_df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)
    for interval in req.intervals:
        mask = (uploaded_df['Date_time'] >= interval['start']) & (uploaded_df['Date_time'] <= interval['end'])
        if req.method == "Delete rows":
            uploaded_df = uploaded_df[~mask]
        elif req.method in ["Forward fill", "Backward fill"]:
            method = "ffill" if req.method == "Forward fill" else "bfill"
            uploaded_df.loc[mask, req.column] = uploaded_df[req.column].fillna(method=method)
        elif req.method == "Mean":
            uploaded_df.loc[mask, req.column] = uploaded_df.loc[mask, req.column].fillna(uploaded_df[req.column].mean())
        elif req.method == "Median":
            uploaded_df.loc[mask, req.column] = uploaded_df.loc[mask, req.column].fillna(uploaded_df[req.column].median())
    return {"message": "Treatment applied successfully."}

@app.get("/download")
def download():
    global uploaded_df
    if uploaded_df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)
    return JSONResponse(content={"csv": uploaded_df.to_csv(index=False)})

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global uploaded_df
    try:
        content = await file.read()
        uploaded_df = pd.read_csv(io.BytesIO(content))
        uploaded_df['Date_time'] = pd.to_datetime(uploaded_df['Date_time'])
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
    try:
        body = await request.body()
        data = json.loads(body)
        prompt = data.get("prompt", "")
        print(f"[DEBUG] Parsed prompt: {prompt}")
    except Exception as e:
        return JSONResponse(content={"type": "text", "data": "Invalid request format."}, status_code=400)

    prompt_lower = prompt.lower()
    try:
        if "missing value analysis" in prompt_lower or "anomaly analysis" in prompt_lower:
            result = visualize_missing_data(prompt)
            return JSONResponse(content={"type": "plot", "data": json.loads(result)})
        elif "variability analysis" in prompt_lower:
            result = plot_variability_tool(prompt)
            return JSONResponse(content={"type": "plot", "data": json.loads(result)})
        else:
            result = agent.run(prompt)
            return JSONResponse(content={"type": "text", "data": str(result)})
    except Exception as e:
        return JSONResponse(content={"type": "text", "data": f"Error: {e}"}, status_code=500)
