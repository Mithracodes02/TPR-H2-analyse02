import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="TPR-H2 Analysis", layout="wide")

st.title("TPR-H2 Data Analysis")
st.write("Upload your data file to visualize temperature-programmed reduction profiles.")

uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader("Raw Data Preview")
        st.dataframe(df.head())

        columns = df.columns.tolist()
        x_col = st.selectbox("Select Temperature Axis (X):", columns, index=0)
        y_col = st.selectbox("Select Signal/TCD Axis (Y):", columns, index=1 if len(columns) > 1 else 0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df[x_col], df[y_col], color="crimson", linewidth=1.5)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title("TPR Profile")
        ax.grid(True, linestyle="--", alpha=0.6)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading file: {e}")
