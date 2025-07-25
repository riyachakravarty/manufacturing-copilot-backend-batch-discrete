# ✅ Changes applied to make `/upload`, `/get_columns`, and `/apply_treatment` fully compatible with frontend multiselect + select-all

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import pandas as pd
import io
from io import StringIO
from typing import List, Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

uploaded_df: Optional[pd.DataFrame] = None
augmented_df: Optional[pd.DataFrame] = None

class Interval(BaseModel):
    start: str
    end: str

class TreatmentRequest(BaseModel):
    columns: List[str]
    intervals: List[Interval]
    method: str

@app.post("/upload")
async def upload_csv(file: UploadFile = File(...)):
    global uploaded_df, augmented_df
    try:
        content = await file.read()
        uploaded_df = pd.read_csv(io.BytesIO(content))
        uploaded_df['Date_time'] = pd.to_datetime(uploaded_df['Date_time'])
        augmented_df = None  # Reset augmented data
        return JSONResponse(content={"message": "File uploaded successfully"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/get_columns")
def get_columns():
    global uploaded_df
    if uploaded_df is not None:
        return {"columns": uploaded_df.columns.tolist()}
    else:
        return {"columns": []}

@app.get("/missing_datetime_intervals")
def get_missing_datetime_intervals():
    global uploaded_df
    if uploaded_df is None:
        return {"intervals": []}

    df = uploaded_df.sort_values('Date_time')
    inferred_freq = pd.infer_freq(df['Date_time'])
    if inferred_freq is None:
        diffs = df['Date_time'].diff().dropna()
        most_common_diff = diffs.mode()[0] if not diffs.empty else pd.Timedelta(minutes=1)
        inferred_freq = most_common_diff
    full_range = pd.date_range(start=df['Date_time'].min(), end=df['Date_time'].max(), freq=inferred_freq)
    missing_times = full_range.difference(df['Date_time'])

    intervals = []
    if not missing_times.empty:
        start = missing_times[0]
        prev = start
        for t in missing_times[1:]:
            if (t - prev) != pd.Timedelta(inferred_freq):
                intervals.append({"start": start.isoformat(), "end": (prev + pd.Timedelta(inferred_freq)).isoformat()})
                start = t
            prev = t
        intervals.append({"start": start.isoformat(), "end": (prev + pd.Timedelta(inferred_freq)).isoformat()})
    return {"intervals": intervals}

@app.post("/apply_treatment")
def apply_treatment(request: TreatmentRequest):
    global uploaded_df, augmented_df

    if uploaded_df is None:
        return JSONResponse(content={"message": "No data uploaded"}, status_code=400)

    columns = request.columns
    intervals = request.intervals
    method = request.method

    # Create augmented_df if not exists
    if augmented_df is None:
        df = uploaded_df.sort_values('Date_time').reset_index(drop=True)
        inferred_freq = pd.infer_freq(df['Date_time'])
        if inferred_freq is None:
            diffs = df['Date_time'].diff().dropna()
            inferred_freq = diffs.mode()[0] if not diffs.empty else pd.Timedelta(minutes=1)
        full_range = pd.date_range(start=df['Date_time'].min(), end=df['Date_time'].max(), freq=inferred_freq)
        full_df = pd.DataFrame({'Date_time': full_range})
        augmented_df = pd.merge(full_df, df, on='Date_time', how='left')

    for interval in intervals:
        start = pd.to_datetime(interval.start)
        end = pd.to_datetime(interval.end)
        mask = (augmented_df['Date_time'] >= start) & (augmented_df['Date_time'] <= end)

        for col in columns:
            if col not in augmented_df.columns:
                continue
            if method == "Delete rows":
                augmented_df = augmented_df[~mask]
            elif method == "Forward fill":
                augmented_df.loc[mask, col] = augmented_df[col].ffill()
            elif method == "Backward fill":
                augmented_df.loc[mask, col] = augmented_df[col].bfill()
            elif method == "Mean":
                augmented_df.loc[mask, col] = augmented_df[col].mean()
            elif method == "Median":
                augmented_df.loc[mask, col] = augmented_df[col].median()

    return {"message": "Treatment applied successfully"}

@app.get("/download")
def download_file():
    global uploaded_df, augmented_df
    df = augmented_df if augmented_df is not None else uploaded_df

    if df is None:
        return JSONResponse(content={"message": "No data available for download"}, status_code=400)

    stream = StringIO()
    df.to_csv(stream, index=False)
    stream.seek(0)
    return StreamingResponse(
        stream,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=treated_data.csv"}
    )
