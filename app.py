import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="TPR-H2 Analysis", layout="wide")

st.title("TPR-H₂ Data Analysis")
st.write("Upload your AutoChem II Excel/CSV raw export file to parse and plot the TPR profile.")

uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "xls", "csv"])

def parse_autochem_file(uploaded_file):
    """
    Scans raw AutoChem Excel/CSV files to locate the 'TCD Signal (a.u.) vs. Temperature'
    data table dynamically, regardless of row/column metadata offset.
    """
    if uploaded_file.name.endswith('.csv'):
        raw_df = pd.read_csv(uploaded_file, header=None, dtype=str)
    else:
        # Load headerless as string grid to prevent openpyxl/pandas float-string type errors
        raw_df = pd.read_excel(uploaded_file, header=None, dtype=str, engine="openpyxl")

    # Look for the section header "TCD Signal (a.u.) vs. Temperature"
    matched_cells = []
    for r_idx, row in raw_df.iterrows():
        for c_idx, val in enumerate(row):
            if pd.notna(val) and "TCD Signal" in str(val) and "Temperature" in str(val):
                matched_cells.append((r_idx, c_idx))

    if matched_cells:
        # Get location of the specific block
        start_row, start_col = matched_cells[0]
        
        # Look for headers ("Temperature", "TCD Signal") within 5 rows below section title
        header_row_idx = None
        for r in range(start_row, min(start_row + 6, len(raw_df))):
            row_vals = [str(v).strip() for v in raw_df.iloc[r, start_col:start_col+3] if pd.notna(v)]
            if any("Temperature" in v for v in row_vals):
                header_row_idx = r
                break
        
        if header_row_idx is not None:
            # Extract numerical data below header row
            data_sub = raw_df.iloc[header_row_idx+1:, start_col:start_col+2].copy()
            data_sub.columns = ["Temperature", "TCD Signal"]
            
            # Convert to numeric values, ignoring non-numeric junk
            data_sub["Temperature"] = pd.to_numeric(data_sub["Temperature"], errors='coerce')
            data_sub["TCD Signal"] = pd.to_numeric(data_sub["TCD Signal"], errors='coerce')
            
            clean_df = data_sub.dropna().reset_index(drop=True)
            if not clean_df.empty:
                return clean_df, "AutoChem Table (TCD Signal vs. Temperature)"

    # Fallback: standard column reading if no specific block pattern was matched
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df, "Standard File Layout"


if uploaded_file is not None:
    try:
        parsed_df, layout_type = parse_autochem_file(uploaded_file)
        
        st.success(f"File parsed successfully using: **{layout_type}**")
        
        st.subheader("Data Preview")
        st.dataframe(parsed_df.head(10))

        cols = parsed_df.columns.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            x_col = st.selectbox("Select X-Axis (Temperature):", cols, index=0)
        with col2:
            y_col = st.selectbox("Select Y-Axis (TCD Signal):", cols, index=1 if len(cols) > 1 else 0)

        # Plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(parsed_df[x_col], parsed_df[y_col], color="#d9534f", linewidth=1.5)
        ax.set_xlabel(f"{x_col} (°C)" if "temp" in x_col.lower() else x_col)
        ax.set_ylabel(y_col)
        ax.set_title("Temperature-Programmed Reduction (TPR) Profile")
        ax.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing file: {e}")
