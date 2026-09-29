import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson, trapezoid

st.set_page_config(page_title="TPR-H2 Advanced Analysis", layout="wide")

st.title("TPR-H₂ Advanced Quantitative Analysis")
st.write("Upload AutoChem raw data to extract TPR profiles and calculate quantitative reduction parameters.")

# Sidebar - Calibration & Sample Metadata
st.sidebar.header("Sample & Calibration Setup")

sample_mass = st.sidebar.number_input("Sample Mass (g):", min_value=0.0001, value=0.0554, format="%.4f")
calib_factor = st.sidebar.number_input(
    "TCD Calibration Factor (Area units per mmol H₂):", 
    min_value=1e-9, value=1.0e-4, format="%.6e",
    help="Area under TCD curve per mmol of H2 consumed."
)

st.sidebar.subheader("Stoichiometry & Metal Content")
metal_wt_pct = st.sidebar.number_input("Metal Content (wt%):", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
metal_mw = st.sidebar.number_input("Metal Molecular Weight (g/mol):", min_value=1.0, value=63.55, step=0.1, help="Default: Cu (63.55 g/mol)")
stoich_factor = st.sidebar.number_input("Theoretical H₂ / Metal Stoichiometry:", min_value=0.1, value=1.0, step=0.1, help="e.g., CuO -> Cu requires 1.0 H2/Cu")

uploaded_file = st.file_uploader("Choose an Excel or CSV file", type=["xlsx", "xls", "csv"])

def parse_autochem_file(uploaded_file):
    """
    Scans raw AutoChem Excel/CSV files to locate the 'TCD Signal (a.u.) vs. Temperature'
    data table dynamically, regardless of row/column metadata offset.
    """
    if uploaded_file.name.endswith('.csv'):
        raw_df = pd.read_csv(uploaded_file, header=None, dtype=str)
    else:
        raw_df = pd.read_excel(uploaded_file, header=None, dtype=str, engine="openpyxl")

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
                return clean_df, "AutoChem Table (TCD Signal vs. Temperature)"

    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    return df, "Standard File Layout"


if uploaded_file is not None:
    try:
        parsed_df, layout_type = parse_autochem_file(uploaded_file)
        st.success(f"File parsed using layout: **{layout_type}**")
        
        cols = parsed_df.columns.tolist()
        c1, c2 = st.columns(2)
        with c1:
            x_col = st.selectbox("Select Temperature Axis (X):", cols, index=0)
        with c2:
            y_col = st.selectbox("Select TCD Signal Axis (Y):", cols, index=1 if len(cols) > 1 else 0)

        # Baseline Correction & Range Selection
        st.subheader("1. Profile Baseline & Integration Range")
        
        t_min, t_max = float(parsed_df[x_col].min()), float(parsed_df[x_col].max())
        selected_range = st.slider("Select Integration Temperature Range (°C):", t_min, t_max, (t_min, t_max))
        
        # Filter data based on range
        mask = (parsed_df[x_col] >= selected_range[0]) & (parsed_df[x_col] <= selected_range[1])
        df_sub = parsed_df[mask].copy().sort_values(by=x_col)

        # Subtract Linear Baseline
        y_raw = df_sub[y_col].values
        x_vals = df_sub[x_col].values
        
        baseline = np.linspace(y_raw[0], y_raw[-1], len(y_raw))
        y_corrected = y_raw - baseline
        y_corrected = np.maximum(y_corrected, 0) # Floor negative noise after baseline correction

        # Area Integration
        integrated_area = trapezoid(y_corrected, x_vals)
        
        # Quantitative Calculations
        total_mmol_h2 = integrated_area / calib_factor if calib_factor > 0 else 0
        h2_consumption_per_g = total_mmol_h2 / sample_mass if sample_mass > 0 else 0 # mmol/g_cat
        
        # Metal Stoichiometry & Reductibility Calculations
        mmol_metal_per_g = (metal_wt_pct / 100.0) * (1000.0 / metal_mw)
        h2_to_metal_ratio = h2_consumption_per_g / mmol_metal_per_g if mmol_metal_per_g > 0 else 0
        
        theoretical_h2_req = mmol_metal_per_g * stoich_factor
        degree_of_reductibility = (h2_consumption_per_g / theoretical_h2_req * 100.0) if theoretical_h2_req > 0 else 0

        # Peak Analysis
        t_max_peak = x_vals[np.argmax(y_corrected)]
        max_signal = np.max(y_corrected)

        # Display Metrics
        st.subheader("2. Quantitative Reduction Results")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Peak Maximum (T_max)", f"{t_max_peak:.1f} °C")
        m2.metric("Total H₂ Consumption", f"{h2_consumption_per_g:.3f} mmol/g")
        m3.metric("H₂ / Metal Molar Ratio", f"{h2_to_metal_ratio:.3f}")
        m4.metric("Degree of Reductibility (DR)", f"{degree_of_reductibility:.1f} %")

        # Visualization
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(x_vals, y_raw, label="Raw TCD Signal", color="gray", linestyle="--", alpha=0.7)
        ax.plot(x_vals, baseline, label="Linear Baseline", color="black", linestyle=":")
        ax.plot(x_vals, y_corrected, label="Corrected TCD Signal", color="#d9534f", linewidth=2)
        ax.fill_between(x_vals, 0, y_corrected, color="#d9534f", alpha=0.2, label=f"Integrated Area: {integrated_area:.2e}")
        
        ax.axvline(t_max_peak, color="blue", linestyle="--", alpha=0.6, label=f"T_max = {t_max_peak:.1f} °C")
        
        ax.set_xlabel("Temperature (°C)")
        ax.set_ylabel("TCD Signal (a.u.)")
        ax.set_title("Quantified Temperature-Programmed Reduction (TPR) Profile")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error executing calculations: {e}")

# Kissinger Kinetics Section
st.markdown("---")
st.subheader("3. Kissinger Kinetic Analysis (Activation Energy $E_a$)")
st.write("Enter data from multiple heating rate experiments ($\beta$) to calculate activation energy:")

num_runs = st.number_input("Number of Heating Rate Runs:", min_value=2, max_value=6, value=3, step=1)

k_cols = st.columns(num_runs)
beta_vals = []
t_max_vals = []

for i, col in enumerate(k_cols):
    with col:
        b = st.number_input(f"Run {i+1} β (K/min):", min_value=0.1, value=5.0 * (i+1), key=f"b_{i}")
        tm = st.number_input(f"Run {i+1} T_max (°C):", min_value=0.0, value=150.0 + (i*15.0), key=f"tm_{i}")
        beta_vals.append(b)
        t_max_vals.append(tm + 273.15) # convert to Kelvin

if len(beta_vals) >= 2:
    R = 8.314 # J/(mol*K)
    x_kissinger = 1.0 / np.array(t_max_vals) # 1/Tm (1/K)
    y_kissinger = np.log(np.array(beta_vals) / (np.array(t_max_vals)**2)) # ln(beta / Tm^2)

    slope, intercept = np.polyfit(x_kissinger, y_kissinger, 1)
    Ea_kJ = (-slope * R) / 1000.0 # kJ/mol

    st.success(f"Calculated Activation Energy ($E_a$): **{Ea_kJ:.2f} kJ/mol**")

    fig_k, ax_k = plt.subplots(figsize=(6, 3))
    ax_k.scatter(x_kissinger, y_kissinger, color="darkblue", zorder=3, label="Data points")
    ax_k.plot(x_kissinger, slope * x_kissinger + intercept, color="crimson", label=f"Fit (Ea = {Ea_kJ:.1f} kJ/mol)")
    ax_k.set_xlabel("1 / T_max (1/K)")
    ax_k.set_ylabel("ln(β / T_max²)")
    ax_k.set_title("Kissinger Plot")
    ax_k.legend()
    ax_k.grid(True, linestyle="--", alpha=0.5)
    st.pyplot(fig_k)
    
