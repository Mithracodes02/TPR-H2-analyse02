import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

st.set_page_config(page_title="TPR-H2 Publication & Analysis Suite", layout="wide")

st.title("TPR-H₂ Publication-Ready Analysis & Multi-Profile Suite")

# ---------------------------------------------------------
# Sidebar: Global Settings & Customization
# ---------------------------------------------------------
st.sidebar.header("1. Calibration Settings")
calib_factor = st.sidebar.number_input(
    "TCD Calibration Factor (Area units per mmol H₂):", 
    min_value=1e-9, value=1.0e-4, format="%.6e",
    help="Calibration constant mapping peak area to consumed hydrogen."
)

st.sidebar.header("2. Layout & Display Options")
plot_mode = st.sidebar.radio("Plot Mode:", ["Overlay", "Stacked / Offset"])
y_offset = 0.0
if plot_mode == "Stacked / Offset":
    y_offset = st.sidebar.number_input("Vertical Y-Offset between curves:", min_value=0.0, value=0.05, step=0.01, format="%.3f")

# Peak Filling & Peak Annotations
fill_peaks = st.sidebar.checkbox("Fill / Shade Peak Area", value=True)
fill_alpha = st.sidebar.slider("Fill Transparency (Alpha):", 0.05, 1.0, 0.15) if fill_peaks else 0.0
show_peak_labels = st.sidebar.checkbox("Annotate Peak T_max Values on Graph", value=True)

st.sidebar.header("3. Publication Typography & Styling")
font_family = st.sidebar.selectbox("Font Style:", ["DejaVu Sans", "Arial", "Times New Roman", "Courier New", "Helvetica", "Calibri"])
font_size_labels = st.sidebar.slider("Axis Label Font Size:", 8, 20, 12)
font_size_ticks = st.sidebar.slider("Tick Label Font Size:", 6, 16, 10)

bold_labels = st.sidebar.checkbox("Bold Axis Titles", value=True)
label_weight = "bold" if bold_labels else "normal"

custom_title = st.sidebar.text_input("Plot Title:", "Temperature-Programmed Reduction (TPR)")
custom_xlabel = st.sidebar.text_input("X-Axis Label:", "Temperature (°C)")
custom_ylabel = st.sidebar.text_input("Y-Axis Label:", "TCD Signal (a.u.)")

st.sidebar.header("4. Ticks, Grid & Spine Aesthetics")
# Tick Controls
show_x_ticks = st.sidebar.checkbox("Show X-Axis Ticks & Labels", value=True)
show_y_ticks = st.sidebar.checkbox("Show Y-Axis Ticks & Labels", value=True)

# Grid Controls
show_grid = st.sidebar.checkbox("Show Grid Lines", value=True)
grid_style = st.sidebar.selectbox("Grid Line Style:", ["--", "-", ":", "-."])
grid_alpha = st.sidebar.slider("Grid Transparency:", 0.1, 1.0, 0.5)

# Spine / Outline Controls
spine_color = st.sidebar.color_picker("Graph Outline / Spine Color:", value="#000000")
spine_width = st.sidebar.slider("Graph Outline Thickness:", 0.5, 3.0, 1.2)

# Legend Customization Controls
st.sidebar.header("5. Legend Settings")
show_legend = st.sidebar.checkbox("Show Legend", value=True)
legend_title = st.sidebar.text_input("Legend Title (Optional):", "")
legend_loc = st.sidebar.selectbox("Legend Location:", ["best", "upper right", "upper left", "lower right", "lower left", "outside"])
legend_fontsize = st.sidebar.slider("Legend Font Size:", 6, 16, 10)
legend_ncols = st.sidebar.number_input("Legend Columns:", min_value=1, max_value=5, value=1)
legend_frame = st.sidebar.checkbox("Draw Legend Box Frame", value=True)

# Apply Global Font Styling
plt.rcParams['font.sans-serif'] = font_family
plt.rcParams['font.serif'] = font_family
plt.rcParams['font.family'] = "sans-serif" if font_family in ["DejaVu Sans", "Arial", "Helvetica", "Calibri"] else "serif"

# ---------------------------------------------------------
# Main File Upload
# ---------------------------------------------------------
st.subheader("Data Upload")
uploaded_files = st.file_uploader(
    "Upload AutoChem Excel/CSV files (Multiple allowed for comparison):", 
    type=["xlsx", "xls", "csv"], 
    accept_multiple_files=True
)

def parse_autochem_file(file):
    """Parses raw AutoChem files by scanning for TCD Signal vs Temperature table."""
    try:
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

        file.seek(0)
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        temp_col, tcd_col = None, None
        for col in df.columns:
            col_str = str(col).lower()
            if "temp" in col_str and temp_col is None:
                temp_col = col
            elif ("tcd" in col_str or "signal" in col_str) and tcd_col is None:
                tcd_col = col

        if temp_col is None: temp_col = df.columns[0]
        if tcd_col is None: tcd_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]

        return pd.DataFrame({
            "Temperature": pd.to_numeric(df[temp_col], errors='coerce'),
            "TCD Signal": pd.to_numeric(df[tcd_col], errors='coerce')
        }).dropna().reset_index(drop=True)

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
        st.success(f"Successfully loaded {len(datasets)} dataset(s).")
        
        # ---------------------------------------------------------
        # Sample Composition, Legend Labels & Dispersion Parameters
        # ---------------------------------------------------------
        st.subheader("Sample Parameters, Custom Legend Labels & Metal Dispersion")
        sample_params = {}
        
        for idx, name in enumerate(datasets.keys()):
            with st.expander(f"Sample Settings: {name}", expanded=True):
                c1, c2, c3, c4 = st.columns([2, 1.5, 1.5, 1])
                with c1:
                    custom_legend_name = st.text_input("Custom Legend Label:", value=name.rsplit('.', 1)[0], key=f"leg_{name}")
                    components = st.text_input("Components Description:", value="Cu/ZnO/Al2O3", key=f"comp_{name}")
                with c2:
                    mass = st.number_input("Catalyst Mass (g):", min_value=0.0001, value=0.0500, format="%.4f", key=f"mass_{name}")
                    metal_wt_pct = st.number_input("Metal (wt%):", min_value=0.0, max_value=100.0, value=10.0, step=0.5, key=f"wt_{name}")
                with c3:
                    metal_mw = st.number_input("Metal MW (g/mol):", min_value=1.0, value=63.55, step=0.1, key=f"mw_{name}")
                    stoich_factor = st.number_input("Stoichiometry (H₂/Metal):", min_value=0.1, value=1.0, step=0.1, key=f"st_{name}")
                with c4:
                    default_colors = ["#d9534f", "#0275d8", "#5cb85c", "#f0ad4e", "#6f42c1", "#17a2b8"]
                    line_color = st.color_picker("Plot Color:", value=default_colors[idx % len(default_colors)], key=f"col_{name}")
                
                sample_params[name] = {
                    "legend_label": custom_legend_name,
                    "mass": mass,
                    "components": components,
                    "metal_wt_pct": metal_wt_pct,
                    "metal_mw": metal_mw,
                    "stoich_factor": stoich_factor,
                    "color": line_color
                }

        # Global Axis Setup
        all_min_temp = min([df["Temperature"].min() for df in datasets.values()])
        all_max_temp = max([df["Temperature"].max() for df in datasets.values()])
        
        st.subheader("Temperature Integration Range")
        x_min, x_max = st.slider(
            "Temperature Range (°C):", 
            float(all_min_temp), 
            float(all_max_temp), 
            (float(all_min_temp), float(all_max_temp))
        )

        # ---------------------------------------------------------
        # Plotting & Processing Engine
        # ---------------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 5))
        report_data = []

        for idx, (name, df) in enumerate(datasets.items()):
            df_sub = df[(df["Temperature"] >= x_min) & (df["Temperature"] <= x_max)].sort_values(by="Temperature")
            
            x_vals = df_sub["Temperature"].values
            y_raw = df_sub["TCD Signal"].values
            
            if len(x_vals) < 2:
                continue

            # Baseline subtraction
            baseline = np.linspace(y_raw[0], y_raw[-1], len(y_raw))
            y_corr = np.maximum(y_raw - baseline, 0)

            # Apply vertical stacking offset if enabled
            offset_val = idx * y_offset
            y_plot = y_corr + offset_val

            # Quantification
            integrated_area = trapezoid(y_corr, x_vals)
            p = sample_params[name]
            mass = p["mass"]
            
            total_mmol_h2 = integrated_area / calib_factor if calib_factor > 0 else 0
            h2_per_g = total_mmol_h2 / mass if mass > 0 else 0
            
            # Metal Stoichiometry & Dispersion Calculations
            mmol_metal_per_g = (p["metal_wt_pct"] / 100.0) * (1000.0 / p["metal_mw"])
            h2_to_metal_ratio = h2_per_g / mmol_metal_per_g if mmol_metal_per_g > 0 else 0
            
            theoretical_h2 = mmol_metal_per_g * p["stoich_factor"]
            reductibility = (h2_per_g / theoretical_h2 * 100.0) if theoretical_h2 > 0 else 0
            dispersion = (h2_per_g / (mmol_metal_per_g * p["stoich_factor"])) * 100.0 if mmol_metal_per_g > 0 else 0

            max_idx = np.argmax(y_corr) if len(y_corr) > 0 else 0
            t_max_val = x_vals[max_idx] if len(x_vals) > 0 else 0
            y_max_val = y_plot[max_idx] if len(y_plot) > 0 else 0

            # Render Line & Area with Custom Legend Label
            color = p["color"]
            label_str = p["legend_label"]
            ax.plot(x_vals, y_plot, label=label_str, color=color, linewidth=2)
            
            if fill_peaks:
                ax.fill_between(x_vals, offset_val, y_plot, color=color, alpha=fill_alpha)

            # Annotate Peak Maximum directly on plot
            if show_peak_labels and len(x_vals) > 0:
                ax.annotate(
                    f"{t_max_val:.1f}°C",
                    xy=(t_max_val, y_max_val),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    fontsize=font_size_ticks,
                    fontweight='bold',
                    color=color
                )

            report_data.append({
                "Filename": name,
                "Legend Label": label_str,
                "Mass (g)": mass,
                "Components": p["components"],
                "Metal (wt%)": p["metal_wt_pct"],
                "Peak T_max (°C)": round(t_max_val, 2),
                "Integrated Area": f"{integrated_area:.3e}",
                "H2 Consumption (mmol/g)": round(h2_per_g, 4),
                "H2 / Metal Ratio": round(h2_to_metal_ratio, 3),
                "Reductibility (%)": round(reductibility, 2),
                "Metal Dispersion (%)": round(dispersion, 2)
            })

        # Apply Graph Styling & Spines
        ax.set_title(custom_title, fontsize=font_size_labels+2, fontweight=label_weight)
        ax.set_xlabel(custom_xlabel, fontsize=font_size_labels, fontweight=label_weight)
        ax.set_ylabel(custom_ylabel, fontsize=font_size_labels, fontweight=label_weight)
        ax.set_xlim(x_min, x_max)

        # Axis Ticks and Labels Visibility
        if show_x_ticks:
            ax.tick_params(axis='x', which='both', bottom=True, labelbottom=True, labelsize=font_size_ticks)
        else:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

        if show_y_ticks:
            ax.tick_params(axis='y', which='both', left=True, labelleft=True, labelsize=font_size_ticks)
        else:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # Spine Styling
        for spine in ax.spines.values():
            spine.set_color(spine_color)
            spine.set_linewidth(spine_width)

        if show_grid:
            ax.grid(True, linestyle=grid_style, alpha=grid_alpha)
            
        # Legend Customization Engine
        if show_legend:
            title_arg = legend_title if legend_title.strip() != "" else None
            if legend_loc == "outside":
                ax.legend(
                    bbox_to_anchor=(1.05, 1), 
                    loc='upper left', 
                    fontsize=legend_fontsize, 
                    title=title_arg, 
                    ncol=legend_ncols, 
                    frameon=legend_frame
                )
            else:
                ax.legend(
                    loc=legend_loc, 
                    fontsize=legend_fontsize, 
                    title=title_arg, 
                    ncol=legend_ncols, 
                    frameon=legend_frame
                )

        st.pyplot(fig)

        # ---------------------------------------------------------
        # Multi-Format Figure Export
        # ---------------------------------------------------------
        st.subheader("Publication Figure Export")
        c_exp1, c_exp2, c_exp3, c_exp4 = st.columns(4)

        buf_png = io.BytesIO()
        fig.savefig(buf_png, format="png", dpi=600, bbox_inches="tight")
        buf_png.seek(0)
        c_exp1.download_button("Download PNG (600 DPI)", buf_png, "tpr_figure.png", "image/png")

        buf_svg = io.BytesIO()
        fig.savefig(buf_svg, format="svg", bbox_inches="tight")
        buf_svg.seek(0)
        c_exp2.download_button("Download Vector SVG", buf_svg, "tpr_figure.svg", "image/svg+xml")

        buf_pdf = io.BytesIO()
        fig.savefig(buf_pdf, format="pdf", bbox_inches="tight")
        buf_pdf.seek(0)
        c_exp3.download_button("Download Publication PDF", buf_pdf, "tpr_figure.pdf", "application/pdf")

        buf_eps = io.BytesIO()
        fig.savefig(buf_eps, format="eps", bbox_inches="tight")
        buf_eps.seek(0)
        c_exp4.download_button("Download Vector EPS", buf_eps, "tpr_figure.eps", "application/postscript")

        # ---------------------------------------------------------
        # Quantitative Results & Text Report
        # ---------------------------------------------------------
        st.subheader("Calculated Quantitative Summary")
        results_df = pd.DataFrame(report_data)
        st.dataframe(results_df)

        report_text = "=========================================\n"
        report_text += "     TPR-H2 QUANTITATIVE & DISPERSION REPORT \n"
        report_text += "=========================================\n\n"
        report_text += f"Calibration Factor: {calib_factor:.6e}\n"
        report_text += f"Integration Range: {x_min:.1f} °C to {x_max:.1f} °C\n\n"
        
        for r in report_data:
            report_text += f"--- Sample: {r['Filename']} ({r['Legend Label']}) ---\n"
            report_text += f"  - Mass of Catalyst: {r['Mass (g)']} g\n"
            report_text += f"  - Active Metal Content: {r['Metal (wt%)']} wt%\n"
            report_text += f"  - Components: {r['Components']}\n"
            report_text += f"  - Peak Temperature (T_max): {r['Peak T_max (°C)']} °C\n"
            report_text += f"  - Specific H2 Consumption: {r['H2 Consumption (mmol/g)']} mmol/g_cat\n"
            report_text += f"  - H2 / Metal Molar Ratio: {r['H2 / Metal Ratio']}\n"
            report_text += f"  - Degree of Reductibility: {r['Reductibility (%)']} %\n"
            report_text += f"  - Calculated Metal Dispersion: {r['Metal Dispersion (%)']} %\n\n"

        st.download_button(
            label="Download Full Analysis & Dispersion Report (TXT)",
            data=report_text,
            file_name="TPR_Comprehensive_Report.txt",
            mime="text/plain"
        )
