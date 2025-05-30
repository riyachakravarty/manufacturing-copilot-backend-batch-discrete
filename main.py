from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
import pandas as pd
import io
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_df = None

def summarize_data(_):
    if uploaded_df is not None:
        return uploaded_df.describe().to_string()
    return "No data uploaded."

def plot_variability_analysis_combined(selected_variable):
    """
    Plot box plot, line plot, and histogram for a given feature column in a combined subplot.

    Parameters:
    - df (pd.DataFrame) : The original dataframe
    - feature_data (pd.Series or pd.DataFrame): The feature data to analyze.

    Returns:
    - fig : figure containing plots for variability analysis
    """
    uploaded_df.set_index('Date_time')
    uploaded_df.sort_index(inplace=True)
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=['Box Plot', 'Line Plot', 'Histogram'],
        row_heights=[0.5, 0.5],  # Adjust the heights as needed
        column_widths=[0.5, 0.5]  # Adjust the widths as needed
    )

    # Box Plot
    box_trace = go.Box(y=uploaded_df[selected_variable], name='Box Plot')
    fig.add_trace(box_trace, row=1, col=1)
    
    # Line Plot
    line_trace = go.Scatter(x=uploaded_df.index, y=uploaded_df[selected_variable], mode='lines', name='Line Plot')
    fig.add_trace(line_trace, row=1, col=2)
    
    # Histogram
    hist_trace = go.Histogram(x=uploaded_df[selected_variable], name='Histogram')
    fig.add_trace(hist_trace, row=2, col=1)
    
    fig.update_layout(
        title_text=f'Combined Feature Analysis - {selected_variable}',
        showlegend=False,
        height=800,  # Adjust the height as needed
        width=1500,  # Adjust the width as needed
    )
    
    # Update layout
    fig.update_layout(title_text='Combined Feature Analysis', showlegend=False)
    fig.update_xaxes(title_text='Index', row=2, col=1)
    fig.update_yaxes(title_text='Values', row=1, col=1)
    fig.update_yaxes(title_text='Values', row=2, col=2)
    
    return fig
    

llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
tools = [
    Tool(
        name="SummarizeData",
        func=summarize_data,
        description="Summarizes the uploaded manufacturing dataset"
    ),
    Tool(
        name="VariabilityAnalysis",
        func=plot_variability_analysis_combined,
        description="Generates a box/line/histogram plot of a selected variable in the uploaded data. Use format: selected_variable is 'ColumnName'"
    )
]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global uploaded_df
    content = await file.read()
    uploaded_df = pd.read_csv(io.BytesIO(content))
    return {"message": "File uploaded successfully"}


from pydantic import BaseModel

class PromptRequest(BaseModel):
    prompt: str

@app.post("/chat")
async def chat(request: PromptRequest):
    result = agent.run(request.prompt)
    return {"response": result}

