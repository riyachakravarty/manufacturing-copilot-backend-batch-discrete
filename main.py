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
    fig.add_trace(go.Scatter(x=uploaded_df.index, y=uploaded_df[selected_variable], mode='lines', name='Line Plot'), row=1, col=2)
    fig.add_trace(go.Histogram(x=uploaded_df[selected_variable], name='Histogram'), row=2, col=1)

    fig.update_layout(
        title_text=f'Combined Feature Analysis - {selected_variable}',
        showlegend=False,
        height=800,
        width=1500
    )

    img_bytes = fig.to_image(format="png")
    encoded = base64.b64encode(img_bytes).decode('utf-8')
    return f"[Image] data:image/png;base64,{encoded}"

def plot_variability_tool(input_text):
    match = re.search(r"selected variable is ['\"](.+?)['\"]", input_text)
    if not match:
        return "Could not find selected variable in prompt."

    selected_variable = match.group(1)
    return plot_variability_analysis_combined(selected_variable)


# Assume these are already defined
#global_df = df_with_sensor1_missing.copy()
#time_col = 'Date_time'


def get_missing_intervals(df, col):
    """
    Groups consecutive missing values in a column and returns intervals (start_time, end_time)
    """
    missing_intervals = []
    df = df.reset_index(drop=True)
    in_interval = False
    start = None

    for i in range(len(df)):
        if pd.isna(df[col].iloc[i]):
            if not in_interval:
                # Start of new missing interval
                if i > 0:
                    start = df[time_col].iloc[i - 1]
                in_interval = True
        elif in_interval:
            # End of missing interval
            end = df[time_col].iloc[i]
            if pd.notna(start) and pd.notna(end):
                missing_intervals.append((start, end))
            in_interval = False

    # If it ends in a missing streak
    if in_interval and start is not None and pd.notna(df[time_col].iloc[-1]):
        missing_intervals.append((start, df[time_col].iloc[-1]))

    return missing_intervals
    

def visualize_missing_data(selected_variable):
    global uploaded_df
    time_col = 'Date_time'

    if uploaded_df is None:
        return "No data uploaded."
        
    uploaded_df[time_col] = pd.to_datetime(uploaded_df[time_col], errors='coerce')

    fig = go.Figure()

    # Plot sensor values
    fig.add_trace(go.Scatter(
        x=uploaded_df[time_col],
        y=uploaded_df[selected_variable],
        mode='lines+markers',
        name=selected_variable,
        line=dict(color='blue')
    ))

    # Red bands: missing Date_time intervals
    time_missing_intervals = get_missing_intervals(uploaded_df, time_col)
    for start, end in time_missing_intervals:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="red", opacity=0.3, line_width=0,
            annotation_text="Missing Date_time", annotation_position="top left"
        )

    # Orange bands: missing selected_variable intervals
    value_missing_intervals = get_missing_intervals(uploaded_df, selected_variable)
    for start, end in value_missing_intervals:
        fig.add_vrect(
            x0=start, x1=end,
            fillcolor="orange", opacity=0.3, line_width=0,
            annotation_text=f"Missing {selected_variable}", annotation_position="top right"
        )

    fig.update_layout(
        title=f"Missing Data Visualization: '{selected_variable}' Over Time",
        xaxis_title=time_col,
        yaxis_title=selected_variable,
        hovermode="x unified",
        height=600
    )

    fig.show()


llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
tools = [
    Tool(
        name="SummarizeData",
        func=summarize_data,
        description="Summarizes the uploaded manufacturing dataset"
    ),
    Tool(
        name="VariabilityAnalysis",
        func=plot_variability_tool,
        description="Generates variability analysis plots. Format: selected_variable is 'ColumnName'"
    ),
    Tool(
        name="MissingValueAnalysis",
        func=visualize_missing_data,
        description="Generates missing value analysis plots. Format: selected_variable is 'ColumnName'"
    )
]

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global uploaded_df
    content = await file.read()
    uploaded_df = pd.read_csv(io.BytesIO(content))
    uploaded_df['Date_time'] = pd.to_datetime(uploaded_df['Date_time'])
    uploaded_df.set_index('Date_time', inplace=True)
    uploaded_df.sort_index(inplace=True)
    return {"message": "File uploaded successfully"}

@app.post("/chat")
async def chat(request: PromptRequest):
    if "selected variable" in request.prompt:
        result = plot_variability_tool(request.prompt)
    else:
        result = agent.run(request.prompt)
    return {"response": result}

