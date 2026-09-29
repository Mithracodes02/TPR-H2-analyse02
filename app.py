import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

st.set_page_config(page_title="TPR-H2 Advanced Multi-Data Suite", layout="wide")

st.title("TPR-H₂ Advanced Analysis & Multi-Profile Comparison")

# Sidebar Configuration
st.sidebar.header("1. Calibration & Global Settings")
calib_factor = st.sidebar.number_input(
    "TCD Calibration Factor (Area units per mmol H₂):", 
    min_value=1e-9, value=1.0e-4, format="%.6e",
    help="Calibration constant mapping peak area to consumed hydrogen."
)

st.sidebar.header("2. Graph Display Customization")
custom_title = st.sidebar.text_input("Plot Title:", "Temperature-Programmed Reduction (TPR)")
custom_xlabel = st.sidebar.text_input("X-Axis Label:", "Temperature (°C)")
custom_ylabel = st.sidebar.text_input("Y-Axis Label:", "TCD Signal (a.u.)")
show_legend = st.sidebar.checkbox("Show Legend", value=True)
show_grid = st.sidebar.checkbox("Show Grid Lines", value=True)

# Main File Upload
st.subheader("Data Upload")
uploaded_files = st.file_uploader(
    "Upload AutoChem Excel/CSV files (Multiple allowed for comparison):", 
    type=["xlsx", "xls", "csv"], 
    accept_multiple_files=True
)

def parse_autochem_file(file):
    """
    Parses raw AutoChem files by scanning for the TCD Signal vs Temperature table
    and standardizes column names to 'Temperature' and 'TCD Signal'.
    """
    try:
        # Load raw file as string grid
        if file.name.endswith('.csv'):
            raw_df = pd.read_csv(file, header=None, dtype=str)
        else:
            raw_df = pd.read_excel(file, header=None, dtype=str, engine="openpyxl")

        matched_cells = []
        for r_idx, row in raw_df.iterrows():
            for c_idx, val in enumerate(row):
                if pd.notna(val) and "TCD Signal" in str(val) and "Temperature" in str(val):
                    matched_cells.append((r_idx, c_idx))

        if matched_cells:
            start_row, start_col = matched_cells[0]
            header_row_idx = None
            for r in range(start_row, min(start_row + 6, len(raw_df))):
                row_vals = [str(v).strip() for v in raw_df.iloc[r, start_col:start_col+3] if pd.notna(v)]
                if any("Temperature" in v for v in row_vals):
                    header_row_idx = r
                    break
            
            if header_row_idx is not None:
                data_sub = raw_df.iloc[header_row_idx+1:, start_col:start_col+2].copy()
                data_sub.columns = ["Temperature", "TCD Signal"]
                data_sub["Temperature"] = pd.to_numeric(data_sub["Temperature"], errors='coerce')
                data_sub["TCD Signal"] = pd.to_numeric(data_sub["TCD Signal"], errors='coerce')
                clean_df = data_sub.dropna().reset_index(drop=True)
                if not clean_df.empty:
                    return clean_df

        # Fallback reading for standard files
        file.seek(0)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        # Standardize column naming in fallback mode
        temp_col = None
        tcd_col = None

        for col in df.columns:
            col_str = str(col).lower()
            if "temp" in col_str and temp_col is None:
                temp_col = col
            elif ("tcd" in col_str or "signal" in col_str) and tcd_col is None:
                tcd_col = col

        # If no matched names, assign by position (Col 0 = Temp, Col 1 = TCD)
        if temp_col is None:
            temp_col = df.columns[0]
        if tcd_col is None:
            tcd_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        df_out = pd.DataFrame({
            "Temperature": pd.to_numeric(df[temp_col], errors='coerce'),
            "TCD Signal": pd.to_numeric(df[tcd_col], errors='coerce')
        }).dropna().reset_index(drop=True)

        return df_out

    except Exception as e:
        st.error(f"Error parsing file {file.name}: {e}")
        return None


if uploaded_files:
    datasets = {}
    for f in uploaded_files:
        parsed = parse_autochem_file(f)
        if parsed is not None and not parsed.empty:
            datasets[f.name] = parsed

    if datasets:
        st.success(f"Successfully parsed {len(datasets)} dataset(s).")
        
        # Metadata Setup for each sample
        st.subheader("Sample Parameters & Mass Details")
        sample_params = {}
        
        for idx, name in enumerate(datasets.keys()):
            with st.expander(f"Sample Settings: {name}", expanded=True):
                col1, col2, col3 = st.columns(3)
                with col1:
                    mass = st.number_input(f"Catalyst Mass (g):", min_value=0.0001, value=0.0500, format="%.4f", key=f"mass_{name}")
                with col2:
                    components = st.text_input(f"Components (e.g., CuO: 10%, ZnO: 15%):", value="CuO: 10%", key=f"comp_{name}")
                with col3:
                    default_colors = ["#d9534f", "#0275d8", "#5cb85c", "#f0ad4e", "#6f42c1"]
                    line_color = st.color_picker(f"Plot Color:", value=default_colors[idx % len(default_colors)], key=f"col_{name}")
                
                sample_params[name] = {
                    "mass": mass,
                    "components": components,
                    "color": line_color
                }

        # Safe Global Axis Range Setup
        all_min_temp = min([df["Temperature"].min() for df in datasets.values()])
        all_max_temp = max([df["Temperature"].max() for df in datasets.values()])
        
        st.subheader("Axis Limits & Integration Range")
        x_min, x_max = st.slider(
            "Temperature Range (°C):", 
            float(all_min_temp), 
            float(all_max_temp), 
            (float(all_min_temp), float(all_max_temp))
        )

        # Process calculations and plotting
        fig, ax = plt.subplots(figsize=(10, 5))
        report_data = []

        for name, df in datasets.items():
            # Filter Range
            df_sub = df[(df["Temperature"] >= x_min) & (df["Temperature"] <= x_max)].sort_values(by="Temperature")
            
            x_vals = df_sub["Temperature"].values
            y_raw = df_sub["TCD Signal"].values
            
            if len(x_vals) < 2:
                continue

            # Baseline subtraction
            baseline = np.linspace(y_raw[0], y_raw[-1], len(y_raw))
            y_corr = np.maximum(y_raw - baseline, 0)

            # Quantification
            integrated_area = trapezoid(y_corr, x_vals)
            mass = sample_params[name]["mass"]
            
            total_mmol = integrated_area / calib_factor if calib_factor > 0 else 0
            mmol_per_g = total_mmol / mass if mass > 0 else 0
            
            t_max_val = x_vals[np.argmax(y_corr)] if len(y_corr) > 0 else 0

            # Plot Profile
            color = sample_params[name]["color"]
            ax.plot(x_vals, y_corr, label=f"{name} (T_max = {t_max_val:.1f}°C)", color=color, linewidth=2)
            ax.fill_between(x_vals, 0, y_corr, color=color, alpha=0.15)

            report_data.append({
                "Filename": name,
                "Mass (g)": mass,
                "Components": sample_params[name]["components"],
                "Peak T_max (°C)": round(t_max_val, 2),
                "Integrated Area": f"{integrated_area:.3e}",
                "H2 Consumption (mmol/g)": round(mmol_per_g, 4)
            })

        # Apply Graph Styling
        ax.set_title(custom_title)
        ax.set_xlabel(custom_xlabel)
        ax.set_ylabel(custom_ylabel)
        ax.set_xlim(x_min, x_max)
        
        if show_grid:
            ax.grid(True, linestyle="--", alpha=0.5)
        if show_legend:
            ax.legend()

        st.pyplot(fig)

        # Graph Download Option
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=300, bbox_inches="tight")
        img_buf.seek(0)
        
        st.download_button(
            label="Download High-Res Graph (PNG)",
            data=img_buf,
            file_name="tpr_profile_comparison.png",
            mime="image/png"
        )

        # Quantitative Results Table
        st.subheader("Calculated Results Summary")
        results_df = pd.DataFrame(report_data)
        st.dataframe(results_df)

        # Text Report Generation
        report_text = "=========================================\n"
        report_text += "     TPR-H2 QUANTITATIVE ANALYSIS REPORT \n"
        report_text += "=========================================\n\n"
        report_text += f"Calibration Factor: {calib_factor:.6e}\n"
        report_text += f"Temperature Integration Range: {x_min:.1f} °C to {x_max:.1f} °C\n\n"
        
        for r in report_data:
            report_text += f"--- Sample: {r['Filename']} ---\n"
            report_text += f"  - Mass of Catalyst: {r['Mass (g)']} g\n"
            report_text += f"  - Components: {r['Components']}\n"
            report_text += f"  - Peak Temperature (T_max): {r['Peak T_max (°C)']} °C\n"
            report_text += f"  - Integrated Peak Area: {r['Integrated Area']}\n"
            report_text += f"  - Specific H2 Consumption: {r['H2 Consumption (mmol/g)']} mmol/g_cat\n\n"

        if len(report_data) > 1:
            report_text += "--- COMPARATIVE SUMMARY ---\n"
            sorted_by_reducibility = sorted(report_data, key=lambda x: x['H2 Consumption (mmol/g)'], reverse=True)
            report_text += f"Highest H2 Consumption: {sorted_by_reducibility[0]['Filename']} ({sorted_by_reducibility[0]['H2 Consumption (mmol/g)']} mmol/g)\n"
            sorted_by_tmax = sorted(report_data, key=lambda x: x['Peak T_max (°C)'])
            report_text += f"Most Easily Reducible (Lowest T_max): {sorted_by_tmax[0]['Filename']} ({sorted_by_tmax[0]['Peak T_max (°C)']} °C)\n"

        st.download_button(
            label="Download Full Analysis Report (TXT)",
            data=report_text,
            file_name="TPR_Analysis_Report.txt",
            mime="text/plain"
        )
