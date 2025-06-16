from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import base64
import re
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import json

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
    intervals: list  # list of {start, end}
    method: str      # e.g., "Mean", "Forward fill", etc.

def summarize_data(_):
    global uploaded_df
    if uploaded_df is not None:
        return uploaded_df.describe().to_string()
    return "No data uploaded."

def plot_variability_analysis_combined(selected_variable):
    global uploaded_df
    if uploaded_df is None:
        return "No data uploaded."

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=['Box Plot', 'Line Plot', 'Histogram'],
        row_heights=[0.5, 0.5],
        column_widths=[0.5, 0.5]
    )

    fig.add_trace(go.Box(y=uploaded_df[selected_variable], name='Box Plot'), row=1, col=1)
    fig.add_trace(go.Scatter(x=uploaded_df['Date_time'], y=uploaded_df[selected_variable], mode='lines', name='Line Plot'), row=1, col=2)
    fig.add_trace(go.Histogram(x=uploaded_df[selected_variable], name='Histogram'), row=2, col=1)

    fig.update_layout(
        title_text=f'Combined Feature Analysis - {selected_variable}',
        showlegend=False,
        height=800,
        width=1500
    )

    return fig.to_json()

def plot_variability_tool(input_text):
    match = re.search(r"selected variable is ['\"](.+?)['\"]", input_text)
    if not match:
        return "Could not find selected variable in prompt."
    return plot_variability_analysis_combined(match.group(1))

def get_all_missing_intervals(df):
    df = df.sort_values(by='Date_time').reset_index(drop=True)
    intervals = set()
    time_series = df['Date_time']
    sensor_cols = [col for col in df.columns if col != 'Date_time']

    for col in sensor_cols:
        in_gap = False
        start = None
        for i in range(len(df)):
            if pd.isna(df[col].iloc[i]):
                if not in_gap:
                    if i > 0:
                        start = time_series.iloc[i - 1]
                    in_gap = True
            elif in_gap:
                end = time_series.iloc[i]
                if pd.notna(start) and pd.notna(end):
                    intervals.add((start, end))
                in_gap = False
        if in_gap and start is not None:
            intervals.add((start, time_series.iloc[-1]))

    sorted_intervals = sorted(intervals)
    return sorted_intervals

def visualize_missing_data(input_text):
    global uploaded_df
    if uploaded_df is None:
        return "No data uploaded."

    match = re.search(r"selected variable is ['\"](.+?)['\"]", input_text)
    if not match:
        return "Could not find selected variable in prompt."
    selected_variable = match.group(1)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=uploaded_df['Date_time'],
        y=uploaded_df[selected_variable],
        mode='lines+markers',
        name=selected_variable,
        line=dict(color='blue')
    ))

    intervals = get_all_missing_intervals(uploaded_df)
    for start, end in intervals:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="orange", opacity=0.3, line_width=0,
            annotation_text=f"Missing", annotation_position="top right"
        )

    fig.update_layout(
        title=f"Missing Data Visualization: '{selected_variable}' Over Time",
        xaxis_title='Date_time',
        yaxis_title=selected_variable,
        hovermode="x unified",
        height=600
    )

    return json.loads(fig.to_json())

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global uploaded_df
    content = await file.read()
    uploaded_df = pd.read_csv(io.BytesIO(content))
    uploaded_df['Date_time'] = pd.to_datetime(uploaded_df['Date_time'])
    return {"message": "File uploaded successfully"}

@app.get("/get_columns")
def get_columns():
    global uploaded_df
    if uploaded_df is not None:
        sensor_columns = [col for col in uploaded_df.columns if col != 'Date_time']
        return JSONResponse(content={"columns": sensor_columns})
    else:
        return JSONResponse(content={"error": "No file uploaded"}, status_code=400)

@app.get("/missing_datetime_intervals")
def missing_datetime_intervals():
    global uploaded_df
    if uploaded_df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)
    intervals = get_all_missing_intervals(uploaded_df)
    formatted = [{"start": str(start), "end": str(end)} for start, end in intervals]
    return JSONResponse(content={"intervals": formatted})

@app.post("/apply_treatment")
def apply_treatment(req: TreatmentRequest):
    global uploaded_df
    if uploaded_df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)

    sensor_cols = [col for col in uploaded_df.columns if col != 'Date_time']

    for interval in req.intervals:
        mask = (uploaded_df['Date_time'] >= interval['start']) & (uploaded_df['Date_time'] <= interval['end'])

        if req.method == "Delete rows":
            uploaded_df = uploaded_df[~mask]
        else:
            for col in sensor_cols:
                if req.method == "Forward fill":
                    uploaded_df.loc[mask, col] = uploaded_df[col].fillna(method="ffill")
                elif req.method == "Backward fill":
                    uploaded_df.loc[mask, col] = uploaded_df[col].fillna(method="bfill")
                elif req.method == "Mean":
                    uploaded_df.loc[mask, col] = uploaded_df.loc[mask, col].fillna(uploaded_df[col].mean())
                elif req.method == "Median":
                    uploaded_df.loc[mask, col] = uploaded_df.loc[mask, col].fillna(uploaded_df[col].median())

    return {"message": "Treatment applied successfully."}

@app.get("/download")
def download():
    global uploaded_df
    if uploaded_df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)
    return JSONResponse(content={"csv": uploaded_df.to_csv(index=False)})

@app.post("/plot_missing")
async def plot_missing(req: PromptRequest):
    result = visualize_missing_data(req.prompt)
    return {"type": "plot", "data": result}

llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
tools = [
    Tool(name="SummarizeData", func=summarize_data, description="Summarizes the uploaded dataset"),
    Tool(name="VariabilityAnalysis", func=plot_variability_tool, description="Generates variability plots. Format: selected variable is 'ColumnName'"),
    Tool(name="MissingValueAnalysis", func=visualize_missing_data, description="Generates missing value plots. Format: selected variable is 'ColumnName'")
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

@app.post("/chat")
async def chat(request: PromptRequest):
    prompt_lower = request.prompt.lower()
    if "missing value analysis" in prompt_lower and "selected variable" in prompt_lower:
        result = visualize_missing_data(request.prompt)
        return {"type": "plot", "data": result}
    elif "variability analysis" in prompt_lower and "selected variable" in prompt_lower:
        result = plot_variability_tool(request.prompt)
        try:
            return {"type": "plot", "data": json.loads(result)}
        except Exception:
            return {"type": "text", "data": str(result)}
    else:
        result = agent.run(request.prompt)
        return {"type": "text", "data": str(result)}
