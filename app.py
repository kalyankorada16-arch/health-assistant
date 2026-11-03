# pip install -r requirements.txt
# streamlit run app.py
import streamlit as st

import config
import pipeline
from document_loader import load_chunks
from retriever import Retriever

st.set_page_config(page_title="AI Health Symptoms Assistant", page_icon="🩺", layout="wide")


@st.cache_resource(show_spinner=False)
def load_index():
    chunks, docs, problems = load_chunks(config.KNOWLEDGE_DIR)
    return Retriever(chunks), docs, problems


retriever, documents, load_problems = load_index()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
history = st.session_state.chat_history

st.title("🩺 AI Health Symptoms Assistant")
st.warning(config.DISCLAIMER)
st.caption(
    f"Emergency: call {config.EMERGENCY_NUMBER} ({config.REGION}). "
    "Emergency screening works even when the AI service is unavailable."
)
if not config.get_api_key():
    st.info("Answer generation is not configured. Emergency screening and reference lookup still work.")
if not documents:
    st.info("No approved reference documents are loaded, so questions will be answered as 'not covered'.")
for problem in load_problems:
    st.warning(f"Skipped reference document: {problem}")

left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("Patient details")
    st.caption("Supported: children 12 months and older, and adults.")
    age_value = st.number_input("Age", min_value=0.0, max_value=120.0, value=25.0, step=1.0)
    age_unit = st.selectbox("Age unit", list(pipeline.AGE_UNITS))
    sex = st.selectbox("Sex", ["Prefer not to say", "Male", "Female", "Other"])
    duration = st.selectbox("Symptom duration", ["Less than 1 day", "1-3 days", "4-7 days", "1-2 weeks", "More than 2 weeks"])
    severity = st.slider("Severity (1 = mild, 10 = severe)", 1, 10, 4)
    st.caption("Each message is answered with these details plus your last few messages.")
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    st.caption(f"Reference library: {len(documents)} document(s).")

with right_col:
    chat_container = st.container(height=520, border=True)
    with chat_container:
        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message["role"] == "user" and message.get("profile"):
                    st.caption(pipeline.Profile.from_dict(message["profile"]).describe())
                for s in message.get("sources", []):
                    dates = f"published {s['published']}" if s["published"] else f"reviewed {s['reviewed']}"
                    st.caption(f"**[{s['label']}]** [{s['title']}]({s['url']}) · {s['publisher']} · version {s['version']} · {dates}")

    question = st.chat_input("Type your symptoms, e.g. 'fever and sore throat for 2 days'")
    if question:
        profile = pipeline.Profile(age_value, age_unit, sex, duration, severity)
        with st.spinner("Checking..."):
            reply = pipeline.answer_question(
                question, profile, history, retriever,
                api_key=config.get_api_key(), model=config.get_model_name(),
            )
        if reply:
            history.append({"role": "user", "content": question.strip(), "profile": profile.to_dict()})
            history.append({"role": "assistant", "content": reply.content, "kind": reply.kind,
                            "sources": reply.sources, "profile": profile.to_dict()})
            del history[:-config.MAX_STORED_MESSAGES]
        st.rerun()
