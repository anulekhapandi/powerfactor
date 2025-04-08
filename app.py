import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# === Page config ===
st.set_page_config(
    page_title="Power Factor Anomaly Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_model():
    return joblib.load(r"D:\powerfactor\model\isolation_forest_pf_model.pkl")

model = load_model()

# === Styling ===
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f6fa;
        font-family: 'Segoe UI', sans-serif;
        color: #2f3542;
    }
    h1, h2, h3 {
        color: #3742fa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Power Factor Anomaly Detector")
st.subheader("Detect phase-wise abnormal power factor behavior with contextual reasoning.")

# === File Upload ===
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Combine Date and Time into DateTime
    if 'Date' in df.columns and 'Time' in df.columns:
        df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], errors='coerce')
    else:
        st.error("CSV must include 'Date' and 'Time' columns.")
        st.stop()

    st.markdown("### Data Preview")
    st.dataframe(df.head(), use_container_width=True)

    selected_columns = [
        "Power Factor AN Avg",
        "Power Factor BN Avg",
        "Power Factor CN Avg",
        "Power Factor Total Avg"
    ]

    df_selected = df[selected_columns].copy()
    predictions = model.predict(df_selected)
    df["Anomaly"] = predictions

    # Count anomalies
    anomaly_count = (df["Anomaly"] == -1).sum()
    st.success(f"Analysis complete. Detected {anomaly_count} anomalies.")

    # === Compute Reasons ===
    def get_reason(value, mean, std):
        z = abs((value - mean) / std) if std != 0 else 0
        if z > 3:
            return "Extreme deviation from norm"
        elif z > 2:
            return "Moderately unusual value"
        elif z > 1:
            return "Slightly unusual, borderline anomaly"
        else:
            return "Normal range"

    for col in selected_columns:
        mean = df[col].mean()
        std = df[col].std()
        reason_col = col + " Reason"
        df[reason_col] = df[col].apply(lambda x: get_reason(x, mean, std))

    # === Visualization ===
    st.markdown("### Visualize Power Factor & Anomalies")
    column_to_plot = st.selectbox("Choose a phase to visualize", selected_columns)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['DateTime'],
        y=df[column_to_plot],
        mode='lines',
        name='Power Factor'
    ))

    anomalies = df[df["Anomaly"] == -1]
    fig.add_trace(go.Scatter(
        x=anomalies['DateTime'],
        y=anomalies[column_to_plot],
        mode='markers',
        marker=dict(color='red', size=8),
        name='Anomaly'
    ))

    fig.update_layout(
        title=f"{column_to_plot} - Anomaly Detection Timeline",
        xaxis_title="Date & Time",
        yaxis_title="Power Factor",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color="#2f3542"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # === Anomaly Table with Reasons ===
    st.markdown("### Detailed Anomalies with Explanation")
    reason_col_name = column_to_plot + " Reason"
    display_cols = ['Date', 'Time', column_to_plot, reason_col_name]
    st.dataframe(anomalies[display_cols], use_container_width=True)

else:
    st.info("Please upload a CSV file to detect and interpret anomalies.")
