"""Teacher dashboard for the Student Early-Warning System (Week 4).

Run from the project folder:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import streamlit as st  # noqa: E402

from src.config import load_config  # noqa: E402
from src.dashboard_data import (filter_predictions, flagged_by_module, reasons_frame,  # noqa: E402
                                risk_band, student_table, summary)
from src.interventions import generate_message, ollama_available  # noqa: E402
from src.plot_style import BLUE, ORANGE, apply_style  # noqa: E402
from src.store import latest_message, load_predictions, save_message  # noqa: E402

st.set_page_config(page_title="Student Early-Warning", layout="wide")
CONFIG = load_config()
PATHS = CONFIG["paths"]


@st.cache_resource
def get_model_info() -> dict:
    path = PATHS["models_dir"] / "model.joblib"
    if not path.exists():
        return {}
    bundle = joblib.load(path)
    return {k: bundle[k] for k in ("model_name", "threshold", "cutoff_day", "trained_at", "data_dir")}


@st.cache_data(ttl=60)
def get_predictions():
    if not PATHS["database"].exists():
        return None
    frame = load_predictions(PATHS["database"])
    return frame if not frame.empty else None


@st.cache_data(ttl=30)
def llm_ready() -> bool:
    return ollama_available(CONFIG["llm"])


st.title("Student Early-Warning Dashboard")
model_info = get_model_info()
predictions = get_predictions()

if not model_info or predictions is None:
    st.warning("No predictions yet. In the terminal run `python -m src.train` and then "
               "`python -m src.predict`, then refresh this page.")
    st.stop()

st.caption(f"Model: {model_info['model_name']} · predictions made on course day "
           f"{model_info['cutoff_day']} using only data from before that day · "
           f"explanations: {predictions['explain_method'].iat[0]}")
if "sample" in Path(model_info["data_dir"]).name:
    st.error("This model was trained on SYNTHETIC sample data. The numbers only show "
             "that the system works - they are not real results.")

# ---------------- Sidebar filters ----------------
with st.sidebar:
    st.header("Filters")
    presentations = sorted(predictions["code_presentation"].unique())
    presentation = st.selectbox("Course run", presentations, index=len(presentations) - 1)
    modules_available = sorted(predictions.loc[predictions["code_presentation"] == presentation,
                                               "code_module"].unique())
    modules = st.multiselect("Modules", modules_available, default=modules_available)
    only_flagged = st.checkbox("Show only flagged students", value=True)
    student_query = st.text_input("Search student ID")
    st.divider()
    st.write("LLM status:", "🟢 Ollama ready" if llm_ready() else
             "⚪ Ollama off - template messages will be used")

try:
    view = filter_predictions(predictions, presentation, modules, only_flagged, student_query)
except ValueError as error:
    st.error(str(error))
    st.stop()

course_view = filter_predictions(predictions, presentation, modules)
stats = summary(course_view)
cols = st.columns(4)
cols[0].metric("Students", f"{stats['students']:,}")
cols[1].metric("Flagged for a check-in", f"{stats['flagged']:,}", f"{stats['flagged_pct']}%",
               delta_color="off")
if "recall_pct" in stats:
    cols[2].metric("Historical check: at-risk students caught", f"{stats['recall_pct']}%")
    cols[3].metric("Historical check: flags that were correct", f"{stats['precision_pct']}%")

students_tab, analytics_tab = st.tabs(["Students", "Analytics"])

# ---------------- Students tab ----------------
with students_tab:
    if view.empty:
        st.info("No students match these filters.")
    else:
        st.dataframe(student_table(view), hide_index=True, height=320)
        options = view["id_student"].astype(str).tolist()
        chosen = st.selectbox("Open a student", options[:500])
        row = view[view["id_student"].astype(str) == chosen].iloc[0]
        key = row[["code_module", "code_presentation", "id_student", "cutoff_day"]].to_dict()

        left, right = st.columns([1, 1])
        with left:
            st.subheader(f"Student {chosen} · {row['code_module']}")
            st.metric("Risk of failing or withdrawing",
                      f"{row['risk_score'] * 100:.0f}%", risk_band(row["risk_score"]),
                      delta_color="off")
            st.dataframe(reasons_frame(row["reasons"]), hide_index=True)
            if isinstance(row["final_result"], str):
                st.caption(f"Actual final result (historical data only): {row['final_result']}")
        with right:
            st.subheader("Check-in message")
            saved = latest_message(PATHS["database"], key)
            if st.button("Write a new message", type="primary"):
                with st.spinner("Writing message..."):
                    message = generate_message(row["code_module"], row["reasons"]["raises"],
                                               CONFIG["llm"])
                save_message(PATHS["database"], key, message.text, message.source, message.note)
                saved = latest_message(PATHS["database"], key)
            if saved:
                st.text_area("Review and edit before sending", saved["message"], height=260)
                note = f" ({saved['note']})" if saved["note"] else ""
                st.caption(f"Written by: {saved['source']}{note} · {saved['created_at']} UTC")
            else:
                st.info("No message yet. Click the button to draft one.")
            st.caption("A tutor always reviews the message before it is sent.")

# ---------------- Analytics tab ----------------
with analytics_tab:
    apply_style()
    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        ax.hist(course_view["risk_score"], bins=20, range=(0, 1), color=BLUE,
                edgecolor="#fcfcfb", linewidth=1)
        ax.axvline(model_info["threshold"], color=ORANGE, linestyle="--", linewidth=1.5)
        ax.annotate("flag threshold", (model_info["threshold"], ax.get_ylim()[1] * 0.9),
                    xytext=(4, 0), textcoords="offset points", fontsize=9)
        ax.set_title("How risk scores are spread")
        ax.set_xlabel("Predicted risk")
        ax.set_ylabel("Students")
        st.pyplot(fig)
        plt.close(fig)
    with chart_right:
        st.markdown("**Flagged students by module**")
        st.dataframe(flagged_by_module(course_view), hide_index=True)

    figures = sorted(PATHS["figures_dir"].glob("model_*.png"))
    if figures:
        st.markdown("**Model evaluation figures** (from `python -m src.train`)")
        for column, figure in zip(st.columns(len(figures)), figures):
            column.image(str(figure))
