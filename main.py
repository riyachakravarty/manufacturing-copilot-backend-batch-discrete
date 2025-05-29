from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from langchain.agents import initialize_agent, Tool
from langchain.llms import OpenAI
import pandas as pd
import io
import os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_df = None

def summarize_data():
    if uploaded_df is not None:
        return uploaded_df.describe().to_string()
    return "No data uploaded."

llm = OpenAI(temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY"))
tools = [
    Tool(
        name="SummarizeData",
        func=summarize_data,
        description="Summarizes the uploaded manufacturing dataset"
    )
]
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global uploaded_df
    content = await file.read()
    uploaded_df = pd.read_csv(io.BytesIO(content))
    return {"message": "File uploaded successfully"}

@app.post("/chat")
async def chat(prompt: str):
    result = agent.run(prompt)
    return {"response": result}
