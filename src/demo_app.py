import streamlit as st
from agent import run_agent

st.set_page_config(page_title="AmericanAir Support Agent Demo", layout="wide")

st.title("✈️ AmericanAir Support Agent — Live Demo")
st.caption(
    "Paste a customer tweet below. The agent will classify intent, draft a retrieval-grounded "
    "reply, and decide whether to auto-handle or escalate - live, using the real pipeline."
)

EXAMPLES = [
    "@AmericanAir my flight AA1234 has been delayed 3 hours and nobody will tell me why",
    "Huge shoutout to the crew on flight 592 today, they were amazing!",
    "@AmericanAir this is the 3rd time this month my bag has been lost. I've had enough.",
]

col1, col2 = st.columns([3, 1])
with col1:
    customer_text = st.text_area("Customer message", height=100, placeholder="Paste a tweet here...")
with col2:
    st.markdown("**Try an example:**")
    for ex in EXAMPLES:
        if st.button(ex[:40] + "...", key=ex, use_container_width=True):
            customer_text = ex
            st.session_state["prefill"] = ex

if "prefill" in st.session_state:
    customer_text = st.session_state["prefill"]

run_clicked = st.button("🚀 Run Agent", type="primary", disabled=not customer_text)

if run_clicked and customer_text:
    with st.spinner("Classifying intent, retrieving historical cases, drafting reply, deciding escalation..."):
        try:
            result = run_agent(customer_text)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.stop()

    st.markdown("---")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Predicted Intent", result["intent"].replace("_", " ").title())
        st.caption(f"Confidence: {result['confidence']:.2f}")
    with c2:
        escalate_label = "🚩 ESCALATE" if result["escalate"] == "yes" else "✅ Auto-handle"
        st.metric("Decision", escalate_label)
    with c3:
        st.metric("Retrieved Similar Cases", len(result["retrieved_examples"]))

    st.markdown("### Classification reasoning")
    st.write(result["classify_reasoning"])

    st.markdown("### Drafted Reply")
    st.success(result["draft_reply"])
    st.caption(f"Grounding: {result['grounding_note']}")

    st.markdown("### Escalation Decision")
    if result["escalate"] == "yes":
        st.warning(f"**Escalate to human.** Reason: {result['escalate_reason']}")
    else:
        st.info(f"**Auto-handle.** Reason: {result['escalate_reason']}")

    with st.expander("See retrieved historical cases used for grounding"):
        for i, m in enumerate(result["retrieved_examples"], 1):
            st.markdown(f"**Match {i}** (similarity: {m['similarity']:.2f})")
            st.write(f"Customer: {m['customer_text']}")
            st.write(f"AmericanAir: {m['americanair_reply']}")
            st.markdown("---")