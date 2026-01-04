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
from typing import List, Any
import traceback


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


###########################-------Generate batch profiles-----------###################

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

class BatchProfileRequest(BaseModel):
    columns: List[str]
    batch_numbers: List[str]



@app.post("/run_batch_profiles")
def run_batch_profiles(req: BatchProfileRequest):
    global uploaded_df, augmented_df

    df = augmented_df if augmented_df is not None else uploaded_df

    if df is None:
        return JSONResponse({"error": "No data uploaded"}, status_code=400)

    # Validate columns
    missing = [c for c in req.columns if c not in df.columns]
    if missing:
        return JSONResponse({"error": f"Columns not found: {missing}"}, status_code=400)

    if "Batch_No" not in df.columns:
        return JSONResponse({"error": "'Batch_No' column missing"}, status_code=400)

    # Work on a copy
    df = df.copy()

    # Ensure Date_time exists and is datetime
    if "Date_time" in df.columns:
        try:
            df["Date_time"] = pd.to_datetime(df["Date_time"])
        except Exception as e:
            print("[run_batch_profiles] Warning: failed to convert Date_time to datetime:", e)
            # continue, but mapping may fail
    else:
        print("[run_batch_profiles] Warning: Date_time column not found. Missing-value rectangles will not be computed robustly.")

    # Prepare batch counter
    df["Batch_Counter"] = df.groupby("Batch_No").cumcount() + 1

    # Normalize Batch_No and requested batch numbers to strings
    df["Batch_No"] = df["Batch_No"].astype(str).str.replace(".0", "", regex=False)
    
    req.batch_numbers = [str(b).replace(".0", "") for b in req.batch_numbers]

    # Filter data
    df_filtered = df[df["Batch_No"].isin(req.batch_numbers)].copy()
    if df_filtered.empty:
        return JSONResponse({"error": "No data for selected batches"}, status_code=400)

    # Global x-axis range (based on batch counters across selected batches)
    max_counter = int(df_filtered["Batch_Counter"].max())

    # Total number of subplots = batches × columns
    total_rows = len(req.batch_numbers) * len(req.columns)

    # Create subplot layout
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            f"Batch {batch} – {col}"
            for batch in req.batch_numbers
            for col in req.columns
        ]
    )

    # Build subplots row by row
    current_row = 1
    for batch in req.batch_numbers:
        batch_df = df_filtered[df_filtered["Batch_No"] == batch].copy()

        # Log batch info
        print(f"[run_batch_profiles] Processing batch: {batch}, rows: {len(batch_df)}")

        # ensure Date_time sorted for this batch
        if "Date_time" in batch_df.columns:
            batch_df = batch_df.sort_values("Date_time").reset_index(drop=True)
        else:
            batch_df = batch_df.reset_index(drop=True)

        for col in req.columns:
            # add main trace
            try:
                fig.add_trace(
                    go.Scatter(
                        x=batch_df["Batch_Counter"],
                        y=batch_df[col],
                        mode="lines",
                        name=f"{batch} – {col}"
                    ),
                    row=current_row,
                    col=1
                )
            except Exception as e:
                print(f"[run_batch_profiles] ERROR adding trace for batch {batch} col {col}: {e}")
                traceback.print_exc()

            # Fix x-axis range for this subplot
            fig.update_xaxes(range=[1, max_counter], row=current_row, col=1)

            # For debugging: log missing-value intervals for this batch/column
            try:
                missing_intervals = get_missing_value_intervals(batch_df, col)
            except Exception as e:
                missing_intervals = []
                print(f"[run_batch_profiles] ERROR computing missing intervals for batch {batch} col {col}: {e}")
                traceback.print_exc()

            print(f"[run_batch_profiles] batch={batch} col={col} missing_intervals_count={len(missing_intervals)}")
            for mi_idx, (start, end) in enumerate(missing_intervals):
                try:
                    # convert to timestamps
                    start_ts = pd.to_datetime(start)
                    end_ts = pd.to_datetime(end)

                    # robustly find batch counters matching these timestamps
                    if "Date_time" not in batch_df.columns or batch_df["Date_time"].isna().all():
                        # fallback: if no datetime in batch_df, skip mapping
                        print(f"[run_batch_profiles] batch={batch} col={col} missing interval {mi_idx} skipped: Date_time missing")
                        continue

                    # searchsorted gives insertion index for start_ts
                    # we want the index of the last timestamp <= start_ts (so choose pos-1 if not exact)
                    times = batch_df["Date_time"]
                    pos_start = times.searchsorted(start_ts)
                    if pos_start < len(times) and times.iloc[pos_start] == start_ts:
                        start_idx = pos_start
                        start_counter = int(batch_df["Batch_Counter"].iloc[start_idx])
                        match_type_start = "exact"
                    else:
                        # choose previous index (last <= start_ts) if exists, else first
                        if pos_start == 0:
                            start_idx = 0
                        else:
                            start_idx = pos_start - 1
                        start_counter = int(batch_df["Batch_Counter"].iloc[start_idx])
                        match_type_start = "nearest_le"

                    pos_end = times.searchsorted(end_ts)
                    # for end, we want last index <= end_ts (so pos_end-1 if pos_end not exact)
                    if pos_end < len(times) and times.iloc[pos_end] == end_ts:
                        end_idx = pos_end
                        end_counter = int(batch_df["Batch_Counter"].iloc[end_idx])
                        match_type_end = "exact"
                    else:
                        if pos_end == 0:
                            end_idx = 0
                        else:
                            end_idx = min(pos_end - 1, len(times) - 1)
                        end_counter = int(batch_df["Batch_Counter"].iloc[end_idx])
                        match_type_end = "nearest_le"

                    print(f"[run_batch_profiles] batch={batch} col={col} interval#{mi_idx} start_ts={start_ts} end_ts={end_ts} -> start_idx={start_idx} end_idx={end_idx} start_counter={start_counter} end_counter={end_counter} (match:{match_type_start}/{match_type_end})")

                    # Only add rectangle if end_counter >= start_counter (safety)
                    if end_counter >= start_counter:
                        fig.add_vrect(
                            x0=start_counter, x1=end_counter,
                            fillcolor="orange", opacity=0.3, line_width=0,
                            row=current_row, col=1
                        )
                        print(f"[run_batch_profiles] Added orange vrect for batch={batch} col={col} x0={start_counter} x1={end_counter}")
                    else:
                        print(f"[run_batch_profiles] SKIPPED vrect for batch={batch} col={col} interval#{mi_idx}: end_counter < start_counter ({end_counter} < {start_counter})")

                except Exception as e:
                    print(f"[run_batch_profiles] ERROR mapping interval #{mi_idx} for batch={batch} col={col}: {e}")
                    traceback.print_exc()
                    # continue to next interval

            current_row += 1

    # Layout settings
    fig.update_layout(
        height=max(400, total_rows * 250),    # make rows smaller so more are visible and page scrolls
        width=1000,
        showlegend=False,
        template="plotly_white",
        title="Batch Profiles"
    )

    return JSONResponse(
        content={
            "type": "plot",
            "data": json.loads(fig.to_json()),
            "message": "Batch profiling successful",
            "num_subplots": total_rows,
        }
    )

###########################-------Generate phase wise batch profiles-----------###################
class Condition(BaseModel):
    column: str
    operator: str         # "<", ">", "==", "<=", ">=", "!="
    value: Any            # string or numeric; for n_times use integer N
    logic: str  # "AND" or "OR" - note: logic to be applied with previous condition
    conditionType: str = "first_time"  # "first_time" or "n_times"

class PhaseDef(BaseModel):
    phaseName: str
    startConditions: List[Condition]
    endConditions: List[Condition]

class DefinePhasesRequest(BaseModel):
    phases: List[PhaseDef]
    # optional: allow client to request which batches to produce plots for immediately
    batch_numbers: List[str] = None
    plot_columns: List[str] = None   # which columns to plot for batch profiles (if omitted use all selected earlier or all numeric)

# -------------------------
# Helper utilities
# -------------------------
def _to_number_if_possible(x):
    try:
        return float(x)
    except Exception:
        return x

def _apply_operator(series: pd.Series, operator: str, value: Any) -> pd.Series:
    """Return boolean mask applying operator between series and value.
       value might be numeric or string; cast series to numeric where appropriate.
    """
    op = operator.strip()
    # Attempt numeric comparison if both sides numeric-like
    # We'll try to coerce value to float
    try:
        numeric_value = float(value)
        # coerce series to numeric (errors -> NaN), comparisons with NaN are False
        s_num = pd.to_numeric(series, errors="coerce")
        if op == ">":
            return s_num > numeric_value
        elif op == "<":
            return s_num < numeric_value
        elif op == ">=":
            return s_num >= numeric_value
        elif op == "<=":
            return s_num <= numeric_value
        elif op in ("==", "="):
            return s_num == numeric_value
        elif op == "!=":
            return s_num != numeric_value
        else:
            # unknown operator: return all False
            return pd.Series(False, index=series.index)
    except Exception:
        # fallback to string comparison
        s_str = series.astype(str)
        val_str = str(value)
        if op in ("==", "="):
            return s_str == val_str
        elif op == "!=":
            return s_str != val_str
        else:
            # cannot do >,< on strings reliably -> return False mask
            return pd.Series(False, index=series.index)

def _mask_for_condition(batch_df: pd.DataFrame, cond: Condition) -> pd.Series:
    """
    Evaluate a single condition and return boolean mask indexed like batch_df.index.
    Supports conditionType "first_time" (simple per-row boolean) and "n_times" (consecutive count).
    """
    series = batch_df[cond.column]

    base_mask = _apply_operator(series, cond.operator, cond.value)

    if (cond.conditionType or "first_time") == "first_time":
        return base_mask.fillna(False)
    elif cond.conditionType == "n_times":
        # value must be integer N
        try:
            N = int(float(cond.value))
            if N <= 0:
                return pd.Series(False, index=batch_df.index)
        except Exception:
            return pd.Series(False, index=batch_df.index)

        # We want positions where condition holds for N consecutive rows.
        # Compute rolling sum on boolean-as-int and find windows with sum == N.
        mask_int = base_mask.astype(int).reindex(batch_df.index, fill_value=0)
        # rolling on mask_int (center=False) -> value at window end
        # Use min_periods=N to ensure full window
        rolling_sum = mask_int.rolling(window=N, min_periods=N).sum()
        # Where rolling_sum == N indicates the last row of a consecutive block of length N satisfying condition.
        # For start detection we will convert this to True at the first index of that run (end_idx - N + 1).
        hits = rolling_sum == N
        hit_indices = hits[hits].index.tolist()
        if not hit_indices:
            return pd.Series(False, index=batch_df.index)
        # create a mask that marks the *first* index of each N-run (end_idx - N +1)
        mask = pd.Series(False, index=batch_df.index)
        for end_idx in hit_indices:
            try:
                # compute position N-window start index
                # end_idx may be label index, get integer position
                pos = batch_df.index.get_loc(end_idx)
                start_pos = pos - N + 1
                if start_pos >= 0:
                    start_idx = batch_df.index[start_pos]
                    mask.loc[start_idx] = True
            except Exception:
                # index lookup issues: skip
                continue
        return mask
    else:
        # unknown condition type
        return pd.Series(False, index=batch_df.index)

def _combine_conditions(batch_df: pd.DataFrame, conditions: List[Condition]) -> pd.Series:
    """
    Combine multiple condition masks into one boolean mask.
    IMPORTANT: combine using the logic of the *previous* condition for cond_i.
    - For the 1st condition (i==0) we simply start with its mask.
    - For cond_i where i>0, get prev_logic = conditions[i-1].logic (default "AND"),
      and combine combined_mask (so far) with current mask using prev_logic.
    """
    if not conditions:
        return pd.Series(False, index=batch_df.index)

    combined = None
    for i, cond in enumerate(conditions):
        mask = _mask_for_condition(batch_df, cond)
        if i == 0:
            combined = mask.copy()
        else:
            prev_logic = (conditions[i - 1].logic or "AND").upper()
            if prev_logic == "AND":
                combined = combined & mask
            elif prev_logic == "OR":
                combined = combined | mask
            else:
                combined = combined & mask
    return combined.fillna(False)

# -------------------------
# /define_phases endpoint
# -------------------------
@app.post("/define_phases")
def define_phases(req: DefinePhasesRequest):
    """
    Accept phase definitions, compute start/end per batch, update augmented_df,
    and return batch profiles with vertical lines for phase starts/ends.
    """
    global uploaded_df, augmented_df

    if uploaded_df is None and augmented_df is None:
        return JSONResponse({"error": "No data uploaded"}, status_code=400)

    # start from uploaded_df if augmented_df absent
    base_df = augmented_df.copy() if augmented_df is not None else uploaded_df.copy()

    df = base_df.copy()

    # validate Batch_No
    if "Batch_No" not in df.columns:
        return JSONResponse({"error": "'Batch_No' column missing"}, status_code=400)

    # ensure Batch_No string and compute batch counter
    df["Batch_No"] = df["Batch_No"].astype(str).str.replace(".0", "", regex=False)
    df["Batch_Counter"] = df.groupby("Batch_No").cumcount() + 1
    df["Date_time"] = pd.to_datetime(df["Date_time"])

    # ensure columns used in conditions exist
    # collect all columns referenced
    referenced_cols = set()
    for phase in req.phases:
        for c in phase.startConditions + phase.endConditions:
            referenced_cols.add(c.column)
    missing_cols = [c for c in referenced_cols if c not in df.columns]
    if missing_cols:
        return JSONResponse({"error": f"Referenced columns not found: {missing_cols}"}, status_code=400)

    # prepare augmented columns for each phase's start and end (boolean)
    for phase in req.phases:
        safe_name = phase.phaseName.strip().replace(" ", "_")
        start_col = f"{safe_name}_start"
        end_col = f"{safe_name}_end"
        df[start_col] = False
        df[end_col] = False

    # process each batch separately and mark starts/ends
    for batch, batch_df in df.groupby("Batch_No"):
        # batch_df is a view — copy to ensure indexing stable
        batch_df = batch_df.copy()
        # We'll track previous phase's end batch_counter to enforce phase ordering
        prev_phase_end_counter = None  # integer batch_counter

        for phase in req.phases:
            safe_name = phase.phaseName.strip().replace(" ", "_")
            start_col = f"{safe_name}_start"
            end_col = f"{safe_name}_end"

            # compute start mask & candidate start points (batch_counter values)
            start_mask = _combine_conditions(batch_df, phase.startConditions)
            # For combined start_mask, we want candidate batch_counters where mask True.
            # If the mask marks first rows of N-run (n_times), it will have True at that row.
            start_candidates = list(batch_df.loc[start_mask, "Batch_Counter"].astype(int).tolist())

            # compute end mask & candidate end points
            end_mask = _combine_conditions(batch_df, phase.endConditions)
            end_candidates = list(batch_df.loc[end_mask, "Batch_Counter"].astype(int).tolist())

            # choose start_point and end_point ensuring ordering constraints
            chosen_start = None
            chosen_end = None

            # iterate through start_candidates in ascending order and find a valid pair with an end > start
            for s in sorted(start_candidates):
                # enforce start must be after prev_phase_end_counter (if present)
                if prev_phase_end_counter is not None and s <= prev_phase_end_counter:
                    continue

                # find the smallest end candidate > s
                larger_ends = [e for e in sorted(end_candidates) if e > s]
                if not larger_ends:
                    # no valid end for this start; try next start
                    continue

                e = larger_ends[0]

                # sanity check: enforce e > s (should be by construction)
                if e <= s:
                    continue

                # Also ensure that if there is a next phase, its start will be > e.
                # We cannot foresee next phase now, so we will accept (but next phase processing will enforce)
                chosen_start = s
                chosen_end = e
                break

            # If we couldn't find pair, skip marking for this batch-phase
            if chosen_start is None or chosen_end is None:
                # leave default False flags
                prev_phase_end_counter = prev_phase_end_counter  # unchanged
                continue

            # mark df rows where Batch_Counter == chosen_start / chosen_end for this batch
            # find indices in original df for this batch and these counters
            batch_mask_total = (df["Batch_No"] == batch)
            start_idx_mask = batch_mask_total & (df["Batch_Counter"] == chosen_start)
            end_idx_mask = batch_mask_total & (df["Batch_Counter"] == chosen_end)

            df.loc[start_idx_mask, start_col] = True
            df.loc[end_idx_mask, end_col] = True

            # update prev_phase_end_counter for next phase ordering
            prev_phase_end_counter = chosen_end

    # Save augmented_df globally
    augmented_df = df.copy()

    # Optionally return batch profiles with vertical lines for each phase start/end
    # If client provided batch_numbers and plot_columns, use them; otherwise plot all batches and all req-referenced numeric columns
    plot_batches = [str(b).replace(".0", "") for b in req.batch_numbers] if req.batch_numbers else sorted(df["Batch_No"].unique())
    if req.plot_columns:
        plot_columns = req.plot_columns
    else:
        # choose numeric columns among referenced or all numeric
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        # prefer referenced columns if numeric; else use first few numeric columns
        plot_columns = [c for c in numeric_cols if c in set([cond.column for ph in req.phases for cond in (ph.startConditions + ph.endConditions)])]
        if not plot_columns:
            # fallback: use first 2 numeric cols
            plot_columns = numeric_cols[:2] if numeric_cols else []

    # If no plot_columns found, return augmented_df saved and a message
    if not plot_columns:
        return JSONResponse(
            content={
                "message": "Phase definitions saved. No numeric columns available for plotting.",
                "saved_phases": [p.phaseName for p in req.phases],
            }
        )

    # Prepare df_filtered for plotting
    df_plot = augmented_df.copy()
    df_plot["Batch_No"] = df_plot["Batch_No"].astype(str)
    df_plot["Batch_Counter"] = df_plot.groupby("Batch_No").cumcount() + 1
    df_plot_filtered = df_plot[df_plot["Batch_No"].isin(plot_batches)]

    if df_plot_filtered.empty:
        return JSONResponse({"error": "No data for plotting after phase definition"}, status_code=400)

    max_counter = int(df_plot_filtered["Batch_Counter"].max())
    total_rows = len(plot_batches) * len(plot_columns)

    # Create subplot grid: each row = one (batch, column) pair
    subplot_titles = [f"Batch {b} – {c}" for b in plot_batches for c in plot_columns]
    fig = make_subplots(rows=total_rows, cols=1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=subplot_titles)

    current_row = 1
    for batch in plot_batches:
        batch_df = df_plot_filtered[df_plot_filtered["Batch_No"] == batch]
        if batch_df.empty:
            # still increment rows for consistency
            current_row += len(plot_columns)
            continue

        for col in plot_columns:
            # main trace
            fig.add_trace(
                go.Scatter(
                    x=batch_df["Batch_Counter"],
                    y=batch_df[col],
                    mode="lines",
                    name=f"{batch} – {col}",
                    hoverinfo="x+y"
                ),
                row=current_row, col=1
            )

            # find phase vertical lines for this batch
            for phase in req.phases:
                safe_name = phase.phaseName.strip().replace(" ", "_")
                start_col = f"{safe_name}_start"
                end_col = f"{safe_name}_end"

                # get start and end counters for this batch
                starts = batch_df.loc[batch_df[start_col] == True, "Batch_Counter"].tolist()
                ends = batch_df.loc[batch_df[end_col] == True, "Batch_Counter"].tolist()

                # add start verticals (green dashed)
                for s in starts:
                    fig.add_vline(x=s, line=dict(color="green", dash="dash"), row=current_row, col=1)

                # add end verticals (red dashed)
                for e in ends:
                    fig.add_vline(x=e, line=dict(color="red", dash="dash"), row=current_row, col=1)

            # fix x range to global max_counter
            fig.update_xaxes(range=[1, max_counter], row=current_row, col=1)
            #current_row += 1

            try:
                missing_intervals = get_missing_value_intervals(batch_df, col)
            except Exception as e:
                missing_intervals = []
                print(f"[run_batch_profiles] ERROR computing missing intervals for batch {batch} col {col}: {e}")
                traceback.print_exc()

            print(f"[run_batch_profiles] batch={batch} col={col} missing_intervals_count={len(missing_intervals)}")
            for mi_idx, (start, end) in enumerate(missing_intervals):
                try:
                    # convert to timestamps
                    start_ts = pd.to_datetime(start)
                    end_ts = pd.to_datetime(end)

                    # robustly find batch counters matching these timestamps
                    if "Date_time" not in batch_df.columns or batch_df["Date_time"].isna().all():
                        # fallback: if no datetime in batch_df, skip mapping
                        print(f"[run_batch_profiles] batch={batch} col={col} missing interval {mi_idx} skipped: Date_time missing")
                        continue

                    # searchsorted gives insertion index for start_ts
                    # we want the index of the last timestamp <= start_ts (so choose pos-1 if not exact)
                    times = batch_df["Date_time"]
                    pos_start = times.searchsorted(start_ts)
                    if pos_start < len(times) and times.iloc[pos_start] == start_ts:
                        start_idx = pos_start
                        start_counter = int(batch_df["Batch_Counter"].iloc[start_idx])
                        match_type_start = "exact"
                    else:
                        # choose previous index (last <= start_ts) if exists, else first
                        if pos_start == 0:
                            start_idx = 0
                        else:
                            start_idx = pos_start - 1
                        start_counter = int(batch_df["Batch_Counter"].iloc[start_idx])
                        match_type_start = "nearest_le"

                    pos_end = times.searchsorted(end_ts)
                    # for end, we want last index <= end_ts (so pos_end-1 if pos_end not exact)
                    if pos_end < len(times) and times.iloc[pos_end] == end_ts:
                        end_idx = pos_end
                        end_counter = int(batch_df["Batch_Counter"].iloc[end_idx])
                        match_type_end = "exact"
                    else:
                        if pos_end == 0:
                            end_idx = 0
                        else:
                            end_idx = min(pos_end - 1, len(times) - 1)
                        end_counter = int(batch_df["Batch_Counter"].iloc[end_idx])
                        match_type_end = "nearest_le"

                    print(f"[run_batch_profiles] batch={batch} col={col} interval#{mi_idx} start_ts={start_ts} end_ts={end_ts} -> start_idx={start_idx} end_idx={end_idx} start_counter={start_counter} end_counter={end_counter} (match:{match_type_start}/{match_type_end})")

                    # Only add rectangle if end_counter >= start_counter (safety)
                    if end_counter >= start_counter:
                        fig.add_vrect(
                            x0=start_counter, x1=end_counter,
                            fillcolor="orange", opacity=0.3, line_width=0,
                            row=current_row, col=1
                        )
                        print(f"[run_batch_profiles] Added orange vrect for batch={batch} col={col} x0={start_counter} x1={end_counter}")
                    else:
                        print(f"[run_batch_profiles] SKIPPED vrect for batch={batch} col={col} interval#{mi_idx}: end_counter < start_counter ({end_counter} < {start_counter})")

                except Exception as e:
                    print(f"[run_batch_profiles] ERROR mapping interval #{mi_idx} for batch={batch} col={col}: {e}")
                    traceback.print_exc()
                    # continue to next interval

            current_row += 1


    fig.update_layout(
        height=max(400, total_rows * 220),
        width=1000,
        showlegend=False,
        template="plotly_white",
        title="Batch Profiles with Phase Markers"
    )

    return JSONResponse(
        content={
            "type": "plot",
            "data": json.loads(fig.to_json()),
            "message": "Phases applied, augmented_df updated and batch profiles returned",
            "num_subplots": total_rows,
        }
    )


# -------------------------
# Helper: safe inferred frequency -> Timedelta
# -------------------------
def _safe_inferred_timedelta(df, datetime_col):
    """
    Return a safe, non-zero pd.Timedelta representing the typical sampling interval.
    Logic:
     - Ensure datetime_col is datetime dtype.
     - Compute diffs; if diffs empty => fallback to 1 hour
     - Use diffs.mode()[0] if available
     - If mode is zero or negative, fall back to median positive diff
     - Ensure final value > 0 seconds
    """
    # convert and sort
    s = pd.to_datetime(df[datetime_col], errors="coerce").dropna().sort_values().reset_index(drop=True)
    if s.empty or len(s) == 1:
        return pd.Timedelta(hours=1)   # safe default

    diffs = s.diff().dropna()
    if diffs.empty:
        return pd.Timedelta(hours=1)

    # pick the most common non-zero diff
    try:
        mode = diffs.mode()
        if not mode.empty:
            candidate = pd.to_timedelta(mode.iloc[0])
        else:
            candidate = pd.to_timedelta(diffs.median())
    except Exception:
        # fallback to median
        candidate = pd.to_timedelta(diffs.median())

    # If candidate is zero or negative, pick smallest positive diff or default
    try:
        if candidate <= pd.Timedelta(0):
            positive = diffs[diffs > pd.Timedelta(0)]
            if not positive.empty:
                candidate = pd.to_timedelta(positive.min())
            else:
                candidate = pd.Timedelta(minutes=1)
    except Exception:
        candidate = pd.Timedelta(minutes=1)

    # final guard
    if candidate <= pd.Timedelta(0):
        candidate = pd.Timedelta(minutes=1)

    return candidate


def get_missing_datetime_intervals(df, datetime_col='Date_time'):
    if datetime_col not in df.columns:
        return []
    df = df.sort_values(by=datetime_col).reset_index(drop=True)
    freq_td = _safe_inferred_timedelta(df, datetime_col)
    full_range = pd.date_range(start=df[datetime_col].min(), end=df[datetime_col].max(), freq=freq_td)
    missing_times = full_range.difference(df[datetime_col])

    intervals = []
    if not missing_times.empty:
        start = missing_times[0]
        prev = start
        for t in missing_times[1:]:
            if (t - prev) != pd.Timedelta(freq_td):
                intervals.append((start, prev + pd.Timedelta(freq_td)))
                start = t
            prev = t
        intervals.append((start, prev + pd.Timedelta(freq_td)))
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
    
    annotation_y_offset = 1.02  # first row for annotations (in paper coordinates)
    annotation_row_gap = 0.04   # vertical gap between annotation rows
    for idx, (start, end) in enumerate(get_missing_value_intervals(uploaded_df, selected_variable)):
        # ensure timestamps
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        xmid = start_ts + (end_ts - start_ts) / 2

        # find unique Batch_No values in this interval
        mask = (uploaded_df['Date_time'] >= start_ts) & (uploaded_df['Date_time'] <= end_ts)
        batches = uploaded_df.loc[mask, 'Batch_No'].astype(str).unique().tolist()
        batches_text = ", ".join(batches) if batches else "None"

        # draw rectangle
        fig.add_vrect(
            x0=start_ts, x1=end_ts,
            fillcolor="orange", opacity=0.25, line_width=0
        )

        # add annotation with batch numbers (placed above plot area)
        fig.add_annotation(
            x=xmid,
            y=annotation_y_offset - idx * annotation_row_gap,
            xref="x",
            yref="paper",
            text=f"Missing vals batches: {batches_text}",
            showarrow=False,
            font=dict(size=10),
            align="center",
            bgcolor="rgba(255,165,0,0.15)",
            bordercolor="rgba(255,165,0,0.5)",
            borderwidth=0.5,
            opacity=0.9
        )
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
    
    annotation_y_offset = 1.02  # first row for annotations (in paper coordinates)
    annotation_row_gap = 0.04   # vertical gap between annotation rows
    for idx, (start, end) in enumerate(get_missing_value_intervals(df, selected_variable)):
        # ensure timestamps
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        xmid = start_ts + (end_ts - start_ts) / 2

        # find unique Batch_No values in this interval
        mask = (df['Date_time'] >= start_ts) & (df['Date_time'] <= end_ts)
        batches = df.loc[mask, 'Batch_No'].astype(str).unique().tolist()
        batches_text = ", ".join(batches) if batches else "None"

        # draw rectangle
        fig.add_vrect(
            x0=start_ts, x1=end_ts,
            fillcolor="orange", opacity=0.25, line_width=0
        )

        # add annotation with batch numbers (placed above plot area)
        fig.add_annotation(
            x=xmid,
            y=annotation_y_offset - idx * annotation_row_gap,
            xref="x",
            yref="paper",
            text=f"Missing vals batches: {batches_text}",
            showarrow=False,
            font=dict(size=10),
            align="center",
            bgcolor="rgba(255,165,0,0.15)",
            bordercolor="rgba(255,165,0,0.5)",
            borderwidth=0.5,
            opacity=0.9
        )
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


@app.post("/outlier_analysis")
def outlier_analysis(req: dict):
    """
    Performs outlier analysis in two modes:

    OVERALL  → aligns batches by Batch_Counter  
    PHASEWISE → aligns batches by Phase_Counter for a selected phase

    Fix Applied:
    ❌ Removed incorrect looping over each batch first
    ✔ Outliers now computed once per counter position
    """

    global uploaded_df, augmented_df
    df = augmented_df if augmented_df is not None else uploaded_df
    if df is None:
        return JSONResponse({"error": "No data uploaded"}, status_code=400)

    # -------------------------------
    # Extract inputs
    # -------------------------------
    analysis_type = req.get("analysis_type")
    column = req.get("column")
    method = req.get("method")
    perc_threshold = req.get("percOutliersInBatch")
    selected_phase = req.get("phase")

    if column not in df.columns:
        return JSONResponse({"error": f"Column '{column}' not found"}, status_code=400)

    df = df.copy()
    df["Batch_No"] = df["Batch_No"].astype(str)
    df["Date_time"] = pd.to_datetime(df["Date_time"])
    df["Batch_Counter"] = df.groupby("Batch_No").cumcount() + 1

    # ----------------------------------------
    # Outlier detection helper
    # ----------------------------------------
    def detect_outliers(values):
        values = pd.Series(values).dropna()
        if len(values) < 2:
            return [False] * len(values)

        if method.lower() == "zscore":
            std = values.std()
            if std == 0:
                return [False] * len(values)
            z = (values - values.mean()) / std
            mask = (abs(z) > 3)
            return mask.reindex(values.index, fill_value=False).tolist()

        if method.lower() == "iqr":
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            mask = (values < lower) | (values > upper)
            return mask.reindex(values.index, fill_value=False).tolist()

        return [False] * len(values)

    # Track outlier counts
    outlier_counts = {b: 0 for b in df["Batch_No"].unique()}

    # ======================================================
    # 🔥 OVERALL ANALYSIS — FIXED
    # ======================================================
    if analysis_type == "overall":

        global_max_counter = df["Batch_Counter"].max()

        for pos in range(1, global_max_counter + 1):
            subset = df[df["Batch_Counter"] == pos]

            if len(subset) < 2:
                continue

            mask = detect_outliers(subset[column])

            for idx, is_out in zip(subset.index, mask):
                if is_out:
                    b = df.loc[idx, "Batch_No"]
                    outlier_counts[b] += 1

    # ======================================================
    # 🔥 PHASEWISE ANALYSIS — FIXED
    # ======================================================
    elif analysis_type == "phasewise":

        if not selected_phase:
            return JSONResponse({"error": "Phase is required for phasewise mode"}, status_code=400)

        df = add_phase_name_column(df)
        phase_df = df[df["Phase_name"] == selected_phase].copy()

        if phase_df.empty:
            return JSONResponse(
                {"error": f"No rows found for phase '{selected_phase}'"},
                status_code=400
            )

        phase_df["Phase_Counter"] = phase_df.groupby("Batch_No").cumcount() + 1
        global_max_pc = phase_df["Phase_Counter"].max()

        for pos in range(1, global_max_pc + 1):
            subset = phase_df[phase_df["Phase_Counter"] == pos]

            if len(subset) < 2:
                continue

            mask = detect_outliers(subset[column])

            for idx, is_out in zip(subset.index, mask):
                if is_out:
                    b = df.loc[idx, "Batch_No"]
                    outlier_counts[b] += 1

    else:
        return JSONResponse({"error": "Invalid analysis type"}, status_code=400)

    # ======================================================
    # 🔥 Determine OUTLIER Batches
    # ======================================================
    outlier_batches = []

    for batch in df["Batch_No"].unique():

        if analysis_type == "overall":
            denom = df[df["Batch_No"] == batch]["Batch_Counter"].max()

        else:
            if batch not in phase_df["Batch_No"].values:
                continue
            denom = phase_df[phase_df["Batch_No"] == batch]["Phase_Counter"].max()

        if denom == 0:
            continue

        perc = (outlier_counts[batch] / denom) * 100

        if perc >= perc_threshold:
            outlier_batches.append(batch)

    if not outlier_batches:
        return JSONResponse({"message": "No batches exceed threshold"})

    # ======================================================
    # 🔥 Build Plot
    # ======================================================
    df_plot = df[df["Batch_No"].isin(outlier_batches)]
    counter_name = "Batch_Counter" if analysis_type == "overall" else "Phase_Counter"
    x_max = df_plot[counter_name].max()

    # Precompute outlier positions
    outlier_positions = {b: [] for b in outlier_batches}
    ref = df if analysis_type == "overall" else phase_df

    for pos in range(1, x_max + 1):
        subset = ref[ref[counter_name] == pos]
        if len(subset) < 2:
            continue
        mask = detect_outliers(subset[column])
        for idx, is_out in zip(subset.index, mask):
            if is_out:
                b = ref.loc[idx, "Batch_No"]
                if b in outlier_positions:
                    outlier_positions[b].append(pos)

    # Plotting
    fig = make_subplots(rows=len(outlier_batches), cols=1, shared_xaxes=True)

    row = 1
    for batch in outlier_batches:
        bdf = df_plot[df_plot["Batch_No"] == batch]

        fig.add_trace(go.Scatter(
            x=bdf[counter_name],
            y=bdf[column],
            mode="lines",
            name=f"Batch {batch}"
        ), row=row, col=1)

        for pos in outlier_positions[batch]:
            val = bdf[bdf[counter_name] == pos][column].values
            if len(val) > 0:
                fig.add_vline(
                    x=pos,
                    line=dict(color="red", dash="dot", width=2),
                    row=row, col=1
                )
                fig.add_trace(go.Scatter(
                    x=[pos],
                    y=[val[0]],
                    mode="markers",
                    marker=dict(color="red", size=8)
                ), row=row, col=1)

        fig.update_xaxes(range=[1, x_max], row=row, col=1)
        row += 1

    fig.update_layout(
        height=350 * len(outlier_batches),
        width=1000,
        template="plotly_white",
        title="Outlier Batches"
    )

    return JSONResponse({
        "type": "plot",
        "data": json.loads(fig.to_json()),
        "outlier_batches": outlier_batches,
        "message": "Outlier analysis complete"
    })




@app.get("/missing_datetime_intervals")
def missing_datetime_intervals():
    global uploaded_df
    if uploaded_df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)
    intervals = get_missing_datetime_intervals(uploaded_df, 'Date_time')
    formatted = [{"start": str(start), "end": str(end)} for start, end in intervals]
    return JSONResponse(content={"intervals": formatted})


#Helper for adding phase name column
def add_phase_name_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a copy of df with a new column 'Phase_name' filled
    between phase_start and phase_end for each phase in each batch.
    Assumptions:
      - For each phase, there is a '{phase}_start' and '{phase}_end' column (boolean).
      - For each batch, exactly 1 start and 1 end are True per phase.
      - end index > start index.
    """
    if "Batch_No" not in df.columns:
        # If no batch column, we can't do phase-by-batch logic
        print("[add_phase_name_column] 'Batch_No' not found in df.columns")
        return df

    df = df.copy()
    # Normalize datetime
    if "Date_time" in df.columns:
        df["Date_time"] = pd.to_datetime(df["Date_time"])

    # Initialize / reset Phase_name
    if "Phase_name" not in df.columns:
        df["Phase_name"] = None
    else:
        df["Phase_name"] = None

    # Detect phase start/end columns
    start_cols = [c for c in df.columns if c.endswith("_start")]
    print(f"[add_phase_name_column] Found phase start columns: {start_cols}")

    for start_col in start_cols:
        phase = start_col[:-6]  # remove "_start"
        end_col = f"{phase}_end"
        if end_col not in df.columns:
            print(f"[add_phase_name_column] No matching end column for phase '{phase}'")
            continue

        print(f"[add_phase_name_column] Processing phase '{phase}'")

        # For each batch, fill Phase_name between start and end
        for batch, batch_df in df.groupby("Batch_No"):
            start_idxs = batch_df.index[batch_df[start_col] == True].tolist()
            end_idxs = batch_df.index[batch_df[end_col] == True].tolist()

            if not start_idxs or not end_idxs:
                # No valid markers in this batch for this phase
                continue

            start_idx = start_idxs[0]
            end_idx = end_idxs[0]

            if end_idx <= start_idx:
                print(
                    f"[add_phase_name_column] Skipping phase '{phase}' in batch {batch}: "
                    f"end_idx ({end_idx}) <= start_idx ({start_idx})"
                )
                continue

            print(
                f"[add_phase_name_column] Setting Phase_name='{phase}' "
                f"for batch={batch}, rows {start_idx} to {end_idx}"
            )
            df.loc[start_idx:end_idx, "Phase_name"] = phase

    return df


@app.get("/missing_value_intervals")
def missing_value_intervals(column: str):
    global uploaded_df, augmented_df

    df = augmented_df if augmented_df is not None else uploaded_df

    if df is None:
        print("[MVI] ERROR: No data uploaded")
        return JSONResponse({"error": "No data uploaded"}, status_code=400)

    if column not in df.columns:
        print(f"[MVI] ERROR: Column {column} not found in DF")
        return JSONResponse({"error": f"Column '{column}' not found"}, status_code=400)

    # ----------------------------
    # Add Phase Name Column
    # ----------------------------
    try:
        print("[MVI] Adding Phase_name column using add_phase_name_column()")
        df = add_phase_name_column(df.copy())
    except Exception as e:
        print(f"[MVI] ERROR constructing Phase_name column: {e}")
        return JSONResponse(
            {"error": f"Failed to construct Phase_name column: {e}"},
            status_code=500
        )

    df["Date_time"] = pd.to_datetime(df["Date_time"])

    # ----------------------------
    # Compute missing intervals
    # ----------------------------
    intervals = get_missing_value_intervals(df, column)
    print(f"[MVI] Found {len(intervals)} missing-value intervals for column={column}")

    result_rows = []

    # ----------------------------
    # Iterate through each interval
    # ----------------------------
    for interval_idx, (interval_start, interval_end) in enumerate(intervals):
        print(f"\n[MVI] ===== Interval #{interval_idx} =====")
        print(f"[MVI] Start: {interval_start}, End: {interval_end}")

        # Extract only rows inside this missing interval
        sub = df[
            (df["Date_time"] >= interval_start)
            & (df["Date_time"] <= interval_end)
        ].copy()

        print(f"[MVI] Rows inside interval: {len(sub)}")

        if sub.empty:
            print("[MVI] Interval contains NO timestamps, skipping.")
            continue

        sub = sub.sort_values("Date_time")

        # Print first few rows for debugging
        print("[MVI] Sample interval rows:")
        print(sub[["Date_time", "Batch_No", "Phase_name"]].head())

        # Detect break in batch OR phase
        sub["group_break"] = (
            (sub["Batch_No"] != sub["Batch_No"].shift(1)) |
            (sub["Phase_name"] != sub["Phase_name"].shift(1))
        ).astype(int)

        sub["group_id"] = sub["group_break"].cumsum()

        print("[MVI] Unique groups found:", sub["group_id"].nunique())

        # ----------------------------
        # Build grouped summary rows
        # ----------------------------
        for i, (gid, g) in enumerate(sub.groupby("group_id")):
            row = {
                "Row_ID": f"{interval_start}_{interval_end}_{i}",
                "Interval_Start": str(interval_start),
                "Interval_End": str(interval_end),
                "Timestamp_From": str(g["Date_time"].min()),
                "Timestamp_To": str(g["Date_time"].max()),
                "Batch_No": str(g["Batch_No"].iloc[0]),
                "Phase_Name": str(g["Phase_name"].iloc[0]),
            }

            print(f"[MVI] Summary row (group {gid}): {row}")

            result_rows.append(row)

    print("\n[MVI] ===== FINAL SUMMARY TABLE =====")
    for r in result_rows:
        print(r)

    return JSONResponse(content={"table": result_rows})




@app.get("/outlier_intervals_summary")
def outlier_intervals_summary(
    column: str,
    method: str,
    analysis_type: str,
    phase: str | None = None
):
    """
    Returns summary rows of outliers grouped by Batch, Phase, and timestamp range.
    Output fields:
        Row_ID, Batch_No, Phase_Name, Timestamp, Counter
    """

    global uploaded_df, augmented_df
    df = augmented_df if augmented_df is not None else uploaded_df

    if df is None:
        return JSONResponse({"error": "No data uploaded"}, status_code=400)

    if column not in df.columns:
        return JSONResponse({"error": f"Column '{column}' not found"}, status_code=400)

    df = df.copy()
    df["Batch_No"] = df["Batch_No"].astype(str)
    df["Date_time"] = pd.to_datetime(df["Date_time"])

    # Always create phase names so phasewise works
    df = add_phase_name_column(df)

    # -------------------------
    # Helper: outlier detector
    # -------------------------
    def detect_outliers(values_series):
        values = values_series.dropna()
        if len(values) < 2:
            return [False] * len(values_series)

        if method.lower() == "zscore":
            mean = values.mean()
            std = values.std()
            if std == 0:
                return [False] * len(values_series)
            z = (values - mean) / std
            mask = (abs(z) > 3)

        elif method.lower() == "iqr":
            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            mask = (values < lower) | (values > upper)
        else:
            mask = pd.Series([False] * len(values_series))

        # Map mask back to original index
        full_mask = values_series.apply(lambda x: False)
        full_mask.loc[values.index] = mask
        return full_mask.tolist()

    # ----------------------------------------------------------------------------
    # CASE 1 — OVERALL OUTLIER ANALYSIS (batch-counter alignment)
    # ----------------------------------------------------------------------------
    if analysis_type == "overall":
        df["Batch_Counter"] = df.groupby("Batch_No").cumcount() + 1
        max_counter = df["Batch_Counter"].max()

        summary_rows = []

        for pos in range(1, max_counter + 1):
            subset = df[df["Batch_Counter"] == pos]
            if len(subset) < 2:
                continue

            mask = detect_outliers(subset[column])

            for (idx, is_out) in zip(subset.index, mask):
                if is_out:
                    row = df.loc[idx]
                    summary_rows.append({
                        "Row_ID": f"{row['Batch_No']}_{pos}_{row['Date_time']}",
                        "Batch_No": row["Batch_No"],
                        "Phase_Name": row["Phase_name"],
                        "Counter": int(pos),
                        "Timestamp": str(row["Date_time"])
                    })

        return JSONResponse({"table": summary_rows})

    # ----------------------------------------------------------------------------
    # CASE 2 — PHASEWISE OUTLIER ANALYSIS
    # ----------------------------------------------------------------------------
    elif analysis_type == "phasewise":

        if not phase or phase == "None":
            return JSONResponse(
                {"error": "Phase name must be provided for phasewise analysis"},
                status_code=400
            )

        phase_df = df[df["Phase_name"] == phase].copy()
        if phase_df.empty:
            return JSONResponse({"error": f"No rows found for phase '{phase}'"}, status_code=400)

        # Create phase counter within each batch
        phase_df["Phase_Counter"] = phase_df.groupby("Batch_No").cumcount() + 1
        max_ctr = phase_df["Phase_Counter"].max()

        summary_rows = []

        for pos in range(1, max_ctr + 1):
            subset = phase_df[phase_df["Phase_Counter"] == pos]
            if len(subset) < 2:
                continue

            mask = detect_outliers(subset[column])

            for (idx, is_out) in zip(subset.index, mask):
                if is_out:
                    row = df.loc[idx]
                    summary_rows.append({
                        "Row_ID": f"{row['Batch_No']}_{phase}_{pos}_{row['Date_time']}",
                        "Batch_No": row["Batch_No"],
                        "Phase_Name": row["Phase_name"],
                        "Counter": int(pos),
                        "Timestamp": str(row["Date_time"])
                    })

        return JSONResponse({"table": summary_rows})

    # ----------------------------------------------------------------------------

    else:
        return JSONResponse({"error": "Invalid analysis type"}, status_code=400)



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

@app.get("/get_phases")
def get_phases():
    global augmented_df
    df = augmented_df.copy()
    df = add_phase_name_column(df)
    phases = sorted(df["Phase_name"].dropna().unique().tolist())
    return {"phases": phases}


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
def apply_missing_value_treatment(req: dict):
    """
    Apply missing-value treatment to selected missing-interval summary rows.
    Works for:
       - Mean / Median per batch + phase
       - Forward Fill inside masked block
       - Backward Fill inside masked block
    """

    global uploaded_df, augmented_df

    # Decide which DF to modify
    df = augmented_df if augmented_df is not None else uploaded_df
    df = df.copy()

    if df is None:
        return JSONResponse({"error": "No data uploaded"}, status_code=400)

    # ---- Extract request fields ----
    column = req.get("column")
    selected_row_ids = req.get("intervals", [])
    method = req.get("method", "").lower()

    print("\n[AMVT] ================= APPLY MISSING VALUE TREATMENT ==================")
    print(f"[AMVT] Column: {column}")
    print(f"[AMVT] Selected Row IDs: {selected_row_ids}")
    print(f"[AMVT] Method: {method}")

    if column not in df.columns:
        return JSONResponse({"error": f"Column '{column}' not in dataset"}, status_code=400)

    # Ensure datetime consistency
    df["Date_time"] = pd.to_datetime(df["Date_time"])
    df["Batch_No"] = df["Batch_No"].astype(str)

    # Add Phase_name column (required for grouping)
    try:
        df = add_phase_name_column(df)
    except Exception as e:
        print("[AMVT] ERROR building Phase_name:", e)
        return JSONResponse({"error": f"Failed to build Phase_name: {e}"}, status_code=500)

    # ---- Reconstruct summary_rows (same logic as missing_value_intervals) ----
    summary_rows = []
    intervals = get_missing_value_intervals(df, column)

    for interval_start, interval_end in intervals:
        sub = df[
            (df["Date_time"] >= interval_start)
            & (df["Date_time"] <= interval_end)
        ].copy()

        if sub.empty:
            continue

        sub = sub.sort_values("Date_time")

        sub["group_break"] = (
            (sub["Batch_No"] != sub["Batch_No"].shift(1)) |
            (sub["Phase_name"] != sub["Phase_name"].shift(1))
        ).astype(int)

        sub["group_id"] = sub["group_break"].cumsum()

        for i, (gid, g) in enumerate(sub.groupby("group_id")):
            summary_rows.append({
                "Row_ID": f"{interval_start}_{interval_end}_{i}",
                "Interval_Start": interval_start,
                "Interval_End": interval_end,
                "Timestamp_From": g["Date_time"].min(),
                "Timestamp_To": g["Date_time"].max(),
                "Batch_No": g["Batch_No"].iloc[0],
                "Phase_Name": g["Phase_name"].iloc[0],
            })

    # ---- Treat only selected summary rows ----
    for row in summary_rows:
        if row["Row_ID"] not in selected_row_ids:
            continue

        batch = str(row["Batch_No"])
        phase = row["Phase_Name"]
        ts_from = pd.to_datetime(row["Timestamp_From"])
        ts_to   = pd.to_datetime(row["Timestamp_To"])

        print(f"\n[AMVT] --- Treating Row_ID={row['Row_ID']} batch={batch}, phase={phase} ---")
        print(f"       block: {ts_from} → {ts_to}")

        # Mask for missing segment
        mask = (
            (df["Batch_No"] == batch) &
            (df["Date_time"] >= ts_from) &
            (df["Date_time"] <= ts_to)
        )

        segment = df.loc[mask, column]
        print(f"[AMVT] Missing block size: {segment.isna().sum()} rows")

        # ---------- MEAN ----------
        if method == "mean":
            # Compute mean only for SAME batch + SAME phase
            ref_mask = (df["Batch_No"] == batch)
            if phase and phase != "None":
                ref_mask &= (df["Phase_name"] == phase)

            mean_val = df.loc[ref_mask, column].dropna().mean()
            print(f"[AMVT] Mean for batch={batch}, phase={phase}: {mean_val}")

            df.loc[mask, column] = df.loc[mask, column].fillna(mean_val)

        # ---------- MEDIAN ----------
        elif method == "median":
            ref_mask = (df["Batch_No"] == batch)
            if phase and phase != "None":
                ref_mask &= (df["Phase_name"] == phase)

            median_val = df.loc[ref_mask, column].dropna().median()
            print(f"[AMVT] Median for batch={batch}, phase={phase}: {median_val}")

            df.loc[mask, column] = df.loc[mask, column].fillna(median_val)

        # ---------- FORWARD FILL ----------
        elif method == "forward fill":
            prev_mask = (df["Batch_No"] == batch) & (df["Date_time"] < ts_from)
            if phase and phase != "None":
                prev_mask &= (df["Phase_name"] == phase)

            prev_val = df.loc[prev_mask, column].dropna().iloc[-1] if prev_mask.any() else None
            print(f"[AMVT] Previous value for ffill: {prev_val}")

            filled_segment = segment.copy().fillna(prev_val)
            df.loc[mask, column] = filled_segment

        # ---------- BACKWARD FILL ----------
        elif method == "backward fill":
            next_mask = (df["Batch_No"] == batch) & (df["Date_time"] > ts_to)
            if phase and phase != "None":
                next_mask &= (df["Phase_name"] == phase)

            next_val = df.loc[next_mask, column].dropna().iloc[0] if next_mask.any() else None
            print(f"[AMVT] Next value for bfill: {next_val}")

            filled_segment = segment.copy().fillna(next_val)
            df.loc[mask, column] = filled_segment

        else:
            return JSONResponse({"error": f"Unknown method '{method}'"}, status_code=400)

    # Save updated df
    augmented_df = df.copy()


    return JSONResponse({
        "message": "Missing value treatment applied successfully",
        "updated_rows": len(selected_row_ids),
        "method": method,
        "columns": list(augmented_df.columns)
    })


@app.post("/apply_outlier_treatment")
def apply_outlier_treatment(req: dict):
    """
    Applies outlier treatment based on selected outlier summary rows.
    - Works for BOTH overall and phasewise analysis
    """

    global uploaded_df, augmented_df
    df = augmented_df if augmented_df is not None else uploaded_df

    if df is None:
        return JSONResponse({"error": "No data uploaded"}, status_code=400)

    df = df.copy()
    df["Batch_No"] = df["Batch_No"].astype(str)
    df["Date_time"] = pd.to_datetime(df["Date_time"])

    # --------------- INPUTS ----------------
    column = req.get("column")
    method = req.get("method")                       # Mean, Median, Forward fill, Backward fill, Delete rows, Delete batch
    analysis_type = req.get("analysis_type")         # overall / phasewise
    selected_row_ids = req.get("selected_rows", [])
    selected_phase = req.get("phase")                # only phasewise

    if column not in df.columns:
        return JSONResponse({"error": f"Column {column} not found"}, status_code=400)

    # If phasewise, ensure phase name exists
    if analysis_type == "phasewise":
        df = add_phase_name_column(df.copy())
        if selected_phase is None:
            return JSONResponse({"error": "Phase must be provided for phasewise treatment"}, status_code=400)


    # ----------------------------
    # PARSE ROW_IDs INTO COMPONENTS
    # ----------------------------
    parsed_rows = []

    for rid in selected_row_ids:
        parts = rid.split("_")

        if analysis_type == "overall":
            # Format: Batch_No_pos_timestamp
            if len(parts) < 3:
                continue
            batch = parts[0]
            pos = int(parts[1])
            timestamp = "_".join(parts[2:])     # In case timestamp contains underscores

            parsed_rows.append({
                "Batch_No": batch,
                "Phase": None,
                "Pos": pos,
                "Timestamp": pd.to_datetime(timestamp)
            })

        else:  # phasewise
            # Format: Batch_No_phase_pos_timestamp
            if len(parts) < 4:
                continue
            batch = parts[0]
            phase = parts[1]
            pos = int(parts[2])
            timestamp = "_".join(parts[3:])

            parsed_rows.append({
                "Batch_No": batch,
                "Phase": phase,
                "Pos": pos,
                "Timestamp": pd.to_datetime(timestamp)
            })

    # ----------------------------
    # APPLY TREATMENTS
    # ----------------------------

    # Utility to select mask for a row needing treatment
    def get_mask(row):
        mask = (df["Batch_No"] == row["Batch_No"]) & (df["Date_time"] == row["Timestamp"])
        if analysis_type == "phasewise" and row["Phase"]:
            mask &= (df["Phase_name"] == row["Phase"])
        return mask

    for row in parsed_rows:

        batch = row["Batch_No"]
        ts = row["Timestamp"]
        phase = row["Phase"]

        mask = get_mask(row)
        if mask.sum() == 0:
            continue

        # ----------------------------
        # APPLY TREATMENT METHODS
        # ----------------------------

        # === DELETE BATCH ===
        if method.lower() == "delete batch":
            df = df[df["Batch_No"] != batch]
            continue

        # === DELETE ROWS ===
        if method.lower() == "delete rows":
            df = df[~mask]
            continue

        # === MEAN / MEDIAN IMPUTATION ===
        if method.lower() in ["mean", "median"]:

            if analysis_type == "overall":
                series = df[df["Batch_No"] == batch][column]
            else:
                subset = df[(df["Batch_No"] == batch) & (df["Phase_name"] == phase)]
                series = subset[column]

            if series.dropna().empty:
                continue

            replace_val = series.mean() if method.lower() == "mean" else series.median()

            df.loc[mask, column] = replace_val
            continue

        # === FORWARD FILL ===
        if method.lower() == "forward fill":

            if analysis_type == "overall":
                subset = df[df["Batch_No"] == batch]
            else:
                subset = df[(df["Batch_No"] == batch) & (df["Phase_name"] == phase)]

            subset = subset.sort_values("Date_time")

            # value before the point
            prev_val = subset[subset["Date_time"] < ts][column].dropna()
            prev_val = prev_val.iloc[-1] if len(prev_val) else None

            df.loc[mask, column] = prev_val
            continue

        # === BACKWARD FILL ===
        if method.lower() == "backward fill":

            if analysis_type == "overall":
                subset = df[df["Batch_No"] == batch]
            else:
                subset = df[(df["Batch_No"] == batch) & (df["Phase_name"] == phase)]

            subset = subset.sort_values("Date_time")

            next_val = subset[subset["Date_time"] > ts][column].dropna()
            next_val = next_val.iloc[0] if len(next_val) else None

            df.loc[mask, column] = next_val
            continue

    # Update global store
    augmented_df = df.copy()

    return JSONResponse({
        "message": "Outlier treatment applied successfully",
        "updated_rows": selected_row_ids,
    })


    
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
@app.post("/eda/sensor_overlays")
def sensor_overlays(
    target: str,
    sensors: list[str],
    num_ranges: float,
    performance_direction: str,
    aggregation: str,
    display_mode: str
):
    """
    Sensor overlays comparing good vs bad batches.

    Handles:
    - BCT (Batch Cycle Time)
    - Phase Duration (e.g., Heating Duration)
    - Normal numeric target columns
    - Display overlays over Batch or over any Phase
    - Min–Max shaded envelopes
    """

    global augmented_df, uploaded_df
    df = augmented_df if augmented_df is not None else uploaded_df
    if df is None:
        return JSONResponse(content={"error": "No data uploaded"}, status_code=400)

    df = df.copy()

    try:
        print("\n=== /eda/sensor_overlays Debug ===")
        print("Target:", target)
        print("Sensors:", sensors)
        print("Range %:", num_ranges)
        print("Perf dir:", performance_direction)
        print("Aggregation:", aggregation)
        print("Display:", display_mode)

        # ---------------------------------------------------
        # 1️⃣ Ensure Phase_name exists
        # ---------------------------------------------------
        if "Phase_name" not in df.columns:
            print("[sensor_overlays] Creating Phase_name column...")
            df = add_phase_name_column(df)

        # ---------------------------------------------------
        # 2️⃣ Create Phase_counter (Option A: resets each time phase changes)
        # ---------------------------------------------------
        if "Phase_counter" not in df.columns:
            print("[sensor_overlays] Creating Phase_counter column (Option A).")

            df["Phase_counter"] = 0

            for batch, batch_df in df.groupby("Batch_No"):
                last_phase = None
                counter = 0
                for idx, row in batch_df.iterrows():
                    phase = row["Phase_name"]
                    if phase != last_phase:
                        counter = 1
                        last_phase = phase
                    else:
                        counter += 1
                    df.at[idx, "Phase_counter"] = counter

        # ---------------------------------------------------
        # 3️⃣ Classify Good/Bad batches
        # ---------------------------------------------------
        pct = num_ranges / 100.0
        target_lower = target.lower()

        is_bct = (target_lower == "bct")
        is_phase_duration = ("duration" in target_lower and not is_bct)

        # ---------------------------
        # CASE A: Target = BCT
        # ---------------------------
        if is_bct:
            print("Classifying batches using BCT (lower = better).")

            batch_stats = (
                df.groupby("Batch_No")
                .size()
                .reset_index(name="duration")
            )

            good_batches = (
                batch_stats.sort_values("duration", ascending=True)
                .head(int(len(batch_stats) * pct))["Batch_No"].tolist()
            )
            bad_batches = (
                batch_stats.sort_values("duration", ascending=False)
                .head(int(len(batch_stats) * pct))["Batch_No"].tolist()
            )

        # ---------------------------
        # CASE B: Phase Duration target (e.g., Heating Duration)
        # ---------------------------
        elif is_phase_duration:
            phase = target.replace("Duration", "").strip()
            print(f"Classifying batches using duration of phase '{phase}'.")

            phase_stats = (
                df[df["Phase_name"] == phase]
                .groupby("Batch_No").size()
                .reset_index(name="duration")
            )

            all_batches = df["Batch_No"].unique()
            phase_stats = (
                phase_stats.set_index("Batch_No")
                .reindex(all_batches, fill_value=0)
                .reset_index()
            )

            good_batches = (
                phase_stats.sort_values("duration", ascending=True)
                .head(int(len(phase_stats) * pct))["Batch_No"].tolist()
            )
            bad_batches = (
                phase_stats.sort_values("duration", ascending=False)
                .head(int(len(phase_stats) * pct))["Batch_No"].tolist()
            )

        # ---------------------------
        # CASE C: Normal numeric target
        # ---------------------------
        else:
            print("Classifying using numeric target mean per batch.")

            batch_stats = df.groupby("Batch_No")[target].mean().reset_index()

            if performance_direction == "higher":
                good_batches = (
                    batch_stats.sort_values(target, ascending=False)
                    .head(int(len(batch_stats) * pct))["Batch_No"].tolist()
                )
                bad_batches = (
                    batch_stats.sort_values(target, ascending=True)
                    .head(int(len(batch_stats) * pct))["Batch_No"].tolist()
                )
            else:
                good_batches = (
                    batch_stats.sort_values(target, ascending=True)
                    .head(int(len(batch_stats) * pct))["Batch_No"].tolist()
                )
                bad_batches = (
                    batch_stats.sort_values(target, ascending=False)
                    .head(int(len(batch_stats) * pct))["Batch_No"].tolist()
                )

        print("GOOD batches:", good_batches[:10])
        print("BAD batches:", bad_batches[:10])

        # ---------------------------------------------------
        # 4️⃣ Aggregation function
        # ---------------------------------------------------
        agg_func = np.mean if aggregation.lower() == "mean" else np.median

        # ---------------------------------------------------
        # 5️⃣ Determine overlay length (Batch or Phase)
        # ---------------------------------------------------
        if display_mode == "Batch":
            counter_col = "batch_counter"

            max_good = df[df.Batch_No.isin(good_batches)].groupby("Batch_No").size().max()
            max_bad  = df[df.Batch_No.isin(bad_batches)].groupby("Batch_No").size().max()

        else:
            counter_col = "Phase_counter"
            phase = display_mode

            max_good = (
                df[(df.Batch_No.isin(good_batches)) & (df.Phase_name == phase)]
                .groupby("Batch_No").size().max()
            )
            max_bad = (
                df[(df.Batch_No.isin(bad_batches)) & (df.Phase_name == phase)]
                .groupby("Batch_No").size().max()
            )

        max_len = int(max(max_good, max_bad))
        print("Overlay length =", max_len)

        # ---------------------------------------------------
        # 6️⃣ Build base Plotly figure
        # ---------------------------------------------------
        fig = make_subplots(
            rows=len(sensors),
            cols=1,
            subplot_titles=sensors,
            shared_xaxes=False
        )

        # ---------------------------------------------------
        # 7️⃣ Build overlays with min–max shading
        # ---------------------------------------------------
        for idx, sensor in enumerate(sensors, start=1):

            if display_mode == "Batch":
                df_good_sub = df[df.Batch_No.isin(good_batches)]
                df_bad_sub  = df[df.Batch_No.isin(bad_batches)]
            else:
                df_good_sub = df[
                    (df.Batch_No.isin(good_batches)) &
                    (df.Phase_name == phase)
                ]
                df_bad_sub = df[
                    (df.Batch_No.isin(bad_batches)) &
                    (df.Phase_name == phase)
                ]

            x_vals = list(range(1, max_len + 1))

            good_profile, bad_profile = [], []
            good_min, good_max = [], []
            bad_min, bad_max = [], []

            # Aggregate at each counter index
            for i in x_vals:
                good_vals = df_good_sub[df_good_sub[counter_col] == i][sensor].values
                bad_vals  = df_bad_sub[df_bad_sub[counter_col] == i][sensor].values

                # GOOD batches
                if len(good_vals) > 0:
                    good_profile.append(agg_func(good_vals))
                    good_min.append(np.nanmin(good_vals))
                    good_max.append(np.nanmax(good_vals))
                else:
                    good_profile.append(np.nan)
                    good_min.append(np.nan)
                    good_max.append(np.nan)

                # BAD batches
                if len(bad_vals) > 0:
                    bad_profile.append(agg_func(bad_vals))
                    bad_min.append(np.nanmin(bad_vals))
                    bad_max.append(np.nanmax(bad_vals))
                else:
                    bad_profile.append(np.nan)
                    bad_min.append(np.nan)
                    bad_max.append(np.nan)

            # ---------------------------
            # Plot GOOD (green)
            # ---------------------------
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=good_profile,
                    mode="lines",
                    line=dict(color="green"),
                    name=f"{sensor} — Good"
                ),
                row=idx, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=x_vals + x_vals[::-1],
                    y=good_max + good_min[::-1],
                    fill="toself",
                    fillcolor="rgba(0,255,0,0.25)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False
                ),
                row=idx, col=1
            )

            # ---------------------------
            # Plot BAD (red)
            # ---------------------------
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=bad_profile,
                    mode="lines",
                    line=dict(color="red"),
                    name=f"{sensor} — Bad"
                ),
                row=idx, col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=x_vals + x_vals[::-1],
                    y=bad_max + bad_min[::-1],
                    fill="toself",
                    fillcolor="rgba(255,0,0,0.25)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False
                ),
                row=idx, col=1
            )

        # ---------------------------------------------------
        # 8️⃣ Final layout + return
        # ---------------------------------------------------
        fig.update_layout(
            title_text=f"Sensor Overlays — Target: {target}, {num_ranges}% Range",
            height=400 * len(sensors),
            width=800,
            showlegend=True
        )

        fig_dict = json.loads(fig.to_json())
        print("Total traces:", len(fig_dict.get("data", [])))
        print("=== End Debug ===")

        return JSONResponse(content={"type": "plot", "data": json.loads(fig.to_json())})

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
