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
    match = re.search(r"selected_variable is ['\"](.+?)['\"]", input_text)
    if not match:
        return "Could not find selected variable in prompt."

    selected_variable = match.group(1)
    return plot_variability_analysis_combined(selected_variable)

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
    if "selected_variable" in request.prompt:
        result = plot_variability_tool(request.prompt)
    else:
        result = agent.run(request.prompt)
    return {"response": result}

