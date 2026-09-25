"""
Comprehensive Streamlit Dashboard for Federated Loan Approval System
"""

import json
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Federated Learning - Loan Approval",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration
SERVER_HOST = os.getenv("SERVER_HOST", "federated-server")
SERVER_PORT = os.getenv("SERVER_PORT", "5000")
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"
METRICS_DIR = Path("/app/logs")
NUM_BANKS = 3

# Custom CSS
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
    }
    .success-metric {
        border-left-color: #28a745;
    }
    .warning-metric {
        border-left-color: #ffc107;
    }
    .danger-metric {
        border-left-color: #dc3545;
    }
</style>
""",
    unsafe_allow_html=True,
)


# Helper functions
@st.cache_data(ttl=5)
def fetch_server_status():
    """Fetch federated server status"""
    try:
        response = requests.get(f"{SERVER_URL}/status", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


@st.cache_data(ttl=5)
def fetch_bank_metrics(bank_id):
    """Fetch metrics for a specific bank"""
    try:
        metrics_file = METRICS_DIR / f"bank{bank_id}" / f"bank{bank_id}_metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                return json.load(f)
        return None
    except Exception:
        return None


def get_time_until_next_round(last_aggregation_time):
    """Calculate time until next training round (1 hour after last aggregation)"""
    try:
        if not last_aggregation_time:
            return "Unknown"

        last_time = datetime.fromisoformat(last_aggregation_time.replace("Z", "+00:00"))
        next_time = last_time + timedelta(hours=1)
        now = datetime.now(last_time.tzinfo)

        if now >= next_time:
            return "Retraining in progress..."

        remaining = next_time - now
        hours = remaining.seconds // 3600
        minutes = (remaining.seconds % 3600) // 60
        seconds = remaining.seconds % 60

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    except Exception:
        return "Unknown"


def create_metrics_comparison_chart(banks_data, federated_metrics=None):
    """Create comparison chart for model metrics"""
    metrics_names = ["Accuracy", "AUC", "F1 Score", "Precision", "Recall"]

    fig = go.Figure()

    # Add bank metrics
    for bank_id in range(1, NUM_BANKS + 1):
        bank_data = banks_data.get(bank_id)
        if bank_data:
            values = [
                bank_data.get("accuracy", 0),
                bank_data.get("auc", 0),
                bank_data.get("f1", 0),
                bank_data.get("precision", 0),
                bank_data.get("recall", 0),
            ]
            fig.add_trace(
                go.Bar(
                    name=f"Bank {bank_id}",
                    x=metrics_names,
                    y=values,
                    text=[f"{v:.3f}" for v in values],
                    textposition="auto",
                )
            )

    # Add federated model if available
    if federated_metrics:
        values = [
            federated_metrics.get("accuracy", 0),
            federated_metrics.get("auc", 0),
            federated_metrics.get("f1", 0),
            federated_metrics.get("precision", 0),
            federated_metrics.get("recall", 0),
        ]
        fig.add_trace(
            go.Bar(
                name="Federated Model",
                x=metrics_names,
                y=values,
                text=[f"{v:.3f}" for v in values],
                textposition="auto",
                marker_color="gold",
            )
        )

    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Metrics",
        yaxis_title="Score",
        barmode="group",
        height=400,
        yaxis_range=[0, 1],
    )

    return fig


def create_loan_flow_chart(banks_data):
    """Create chart showing loan approval flow for all banks"""
    banks = []
    approved = []
    denied = []

    for bank_id in range(1, NUM_BANKS + 1):
        bank_data = banks_data.get(bank_id)
        if bank_data:
            banks.append(f"Bank {bank_id}")
            approved.append(bank_data.get("loans_approved", 0))
            denied.append(bank_data.get("loans_denied", 0))

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Approved",
            x=banks,
            y=approved,
            marker_color="green",
            text=approved,
            textposition="auto",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Denied",
            x=banks,
            y=denied,
            marker_color="red",
            text=denied,
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Loan Decisions by Bank",
        xaxis_title="Bank",
        yaxis_title="Number of Loans",
        barmode="stack",
        height=350,
    )

    return fig


def create_dataset_size_chart(banks_data):
    """Create chart showing dataset sizes - FIXED"""
    banks = []
    sizes = []

    for bank_id in range(1, NUM_BANKS + 1):
        bank_data = banks_data.get(bank_id)
        if bank_data:
            banks.append(f"Bank {bank_id}")
            sizes.append(bank_data.get("dataset_size", 0))

    # Create DataFrame for px.bar
    df = pd.DataFrame({"Bank": banks, "Dataset Size": sizes})

    fig = px.bar(
        df,
        x="Bank",
        y="Dataset Size",
        title="Dataset Size by Bank",
        color="Dataset Size",
        color_continuous_scale="Blues",
        text="Dataset Size",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=300)

    return fig


# Main app
def main():
    # Header
    st.markdown(
        '<div class="main-header">🏦 Federated Learning Dashboard - Loan Approval System</div>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Dashboard Controls")
        auto_refresh = st.checkbox("Auto-refresh (5s)", value=True)

        if st.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.rerun()

        st.divider()
        st.subheader("📊 System Info")
        server_status = fetch_server_status()
        if server_status:
            st.success("✅ Server Online")
            st.metric("Training Round", server_status.get("training_round", 0))
            st.metric("Registered Banks", len(server_status.get("clients_registered", [])))
        else:
            st.error("❌ Server Offline")

        st.divider()
        st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Fetch all data
    server_status = fetch_server_status()
    banks_data = {}
    for bank_id in range(1, NUM_BANKS + 1):
        banks_data[bank_id] = fetch_bank_metrics(bank_id)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: FEDERATED MODEL
    # ═══════════════════════════════════════════════════════════════

    st.header("🌐 Federated Model")

    if server_status:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Training Round",
                server_status.get("training_round", 0),
                delta="Active" if server_status.get("global_model_available") else "Pending",
            )

        with col2:
            st.metric(
                "Models Received",
                f"{server_status.get('models_received', 0)}/{NUM_BANKS}",
            )

        with col3:
            aggregation_count = server_status.get("training_round", 0)
            st.metric("Aggregations Completed", aggregation_count)

        with col4:
            time_remaining = get_time_until_next_round(server_status.get("last_aggregation"))
            st.metric("Next Round In", time_remaining)

        # Federated model details
        if server_status.get("global_model_available"):
            st.success("✅ Global model is available and active")

            # Show aggregation timestamp
            if server_status.get("last_aggregation"):
                st.info(f"📅 Last aggregation: {server_status.get('last_aggregation')}")

            # Storage stats
            storage_stats = server_status.get("disk_storage", {})
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Global Models", storage_stats.get("global_models", 0))
            with col2:
                st.metric("Local Models", storage_stats.get("local_models", 0))
            with col3:
                st.metric("Total Models", storage_stats.get("total_models", 0))
        else:
            st.warning("⏳ Global model not yet available - waiting for initial training")
    else:
        st.error("❌ Cannot connect to federated server")

    st.divider()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: BANKS
    # ═══════════════════════════════════════════════════════════════

    st.header("🏦 Banks Performance")

    # Create tabs for Model and Kafka sections
    tab1, tab2 = st.tabs(["📊 Model Metrics", "📡 Kafka Data Flow"])

    # ─────────────────────────────────────────────────────────────
    # TAB 1: MODEL METRICS
    # ─────────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Model Performance Comparison")

        # Metrics comparison chart
        try:
            st.plotly_chart(create_metrics_comparison_chart(banks_data), use_container_width=True)
        except Exception as e:
            st.error(f"Error creating metrics chart: {e}")

        # Individual bank metrics
        st.subheader("Individual Bank Metrics")
        cols = st.columns(NUM_BANKS)

        for idx, bank_id in enumerate(range(1, NUM_BANKS + 1)):
            with cols[idx]:
                bank_data = banks_data.get(bank_id)

                if bank_data:
                    st.markdown(f"### Bank {bank_id}")

                    # Metrics in a nice layout
                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:
                        st.metric("Accuracy", f"{bank_data.get('accuracy', 0):.4f}")
                        st.metric("F1 Score", f"{bank_data.get('f1', 0):.4f}")
                        st.metric("Precision", f"{bank_data.get('precision', 0):.4f}")

                    with metric_col2:
                        st.metric("AUC", f"{bank_data.get('auc', 0):.4f}")
                        st.metric("Recall", f"{bank_data.get('recall', 0):.4f}")
                        st.metric("Dataset Size", f"{bank_data.get('dataset_size', 0):,}")

                    # Last update
                    if bank_data.get("last_updated"):
                        update_time = bank_data["last_updated"].split("T")[1].split(".")[0]
                        st.caption(f"Updated: {update_time}")
                else:
                    st.markdown(f"### Bank {bank_id}")
                    st.warning("⏳ No data available")

        # Dataset size comparison
        st.divider()
        try:
            st.plotly_chart(create_dataset_size_chart(banks_data), use_container_width=True)
        except Exception as e:
            st.error(f"Error creating dataset chart: {e}")

    # ─────────────────────────────────────────────────────────────
    # TAB 2: KAFKA DATA FLOW
    # ─────────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Real-time Loan Processing")

        # Loan flow chart
        try:
            st.plotly_chart(create_loan_flow_chart(banks_data), use_container_width=True)
        except Exception as e:
            st.error(f"Error creating loan flow chart: {e}")

        # Detailed bank statistics
        st.subheader("Bank Statistics")

        for bank_id in range(1, NUM_BANKS + 1):
            bank_data = banks_data.get(bank_id)

            if bank_data:
                with st.expander(f"🏦 Bank {bank_id} - Detailed Stats", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric(
                            "Total Predictions",
                            bank_data.get("predictions_made", 0),
                            help="Number of loan applications processed",
                        )

                    with col2:
                        approved = bank_data.get("loans_approved", 0)
                        st.metric(
                            "Loans Approved",
                            approved,
                            delta=f"{(approved / max(bank_data.get('predictions_made', 1), 1) * 100):.1f}%",
                        )

                    with col3:
                        denied = bank_data.get("loans_denied", 0)
                        st.metric(
                            "Loans Denied",
                            denied,
                            delta=f"{(denied / max(bank_data.get('predictions_made', 1), 1) * 100):.1f}%",
                            delta_color="inverse",
                        )

                    with col4:
                        st.metric(
                            "Total Transactions",
                            bank_data.get("total_transactions", 0),
                            help="Total transactions processed",
                        )

                    # Approval rate gauge
                    if bank_data.get("predictions_made", 0) > 0:
                        approval_rate = (
                            bank_data.get("loans_approved", 0) / bank_data.get("predictions_made", 1)
                        ) * 100

                        fig = go.Figure(
                            go.Indicator(
                                mode="gauge+number",
                                value=approval_rate,
                                title={"text": f"Bank {bank_id} Approval Rate"},
                                gauge={
                                    "axis": {"range": [None, 100]},
                                    "bar": {"color": "darkgreen"},
                                    "steps": [
                                        {"range": [0, 33], "color": "lightcoral"},
                                        {"range": [33, 66], "color": "lightyellow"},
                                        {"range": [66, 100], "color": "lightgreen"},
                                    ],
                                    "threshold": {
                                        "line": {"color": "red", "width": 4},
                                        "thickness": 0.75,
                                        "value": 50,
                                    },
                                },
                            )
                        )
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, use_container_width=True)
            else:
                with st.expander(f"🏦 Bank {bank_id} - No Data"):
                    st.warning("⏳ Waiting for data...")

    # ═══════════════════════════════════════════════════════════════
    # FOOTER
    # ═══════════════════════════════════════════════════════════════

    st.divider()
    st.caption("🔄 Dashboard auto-refreshes every 5 seconds • Built with Streamlit & Plotly")

    # ═══════════════════════════════════════════════════════════════
    # AUTO-REFRESH - MOVED TO END (AFTER ALL CONTENT IS RENDERED)
    # ═══════════════════════════════════════════════════════════════
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
