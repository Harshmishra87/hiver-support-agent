import pandas as pd
import streamlit as st
import os

CANDIDATES_PATH = "eval/golden/golden_candidates.csv"
LABELED_PATH = "eval/golden/golden_labeled.csv"

INTENTS = [
    "flight_disruption",
    "baggage_issue",
    "billing_fee_dispute",
    "booking_seating_assistance",
    "loyalty_aadvantage_inquiry",
    "onboard_technical_issue",
    "service_staff_complaint",
    "praise_non_actionable",
]

st.set_page_config(page_title="Golden Set Labeler", layout="wide")


@st.cache_data
def load_candidates():
    return pd.read_csv(CANDIDATES_PATH)


def load_working_df():
    if os.path.exists(LABELED_PATH):
        return pd.read_csv(LABELED_PATH)
    df = load_candidates().copy()
    df["intent_label"] = ""
    df["escalate"] = ""
    df["escalate_reason"] = ""
    return df


if "df" not in st.session_state:
    st.session_state.df = load_working_df()
if "idx" not in st.session_state:
    unlabeled = st.session_state.df[st.session_state.df["intent_label"] == ""]
    st.session_state.idx = unlabeled.index[0] if len(unlabeled) > 0 else 0

df = st.session_state.df
total = len(df)
labeled_count = (df["intent_label"] != "").sum()

st.title("Golden Set Labeler — AmericanAir")
st.progress(labeled_count / total)
st.caption(f"{labeled_count} / {total} labeled")

idx = st.session_state.idx
row = df.loc[idx]

st.markdown(f"**Row {idx + 1} of {total}** (tweet_id: {row['tweet_id']})")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Customer message:**")
    st.info(row["customer_text"])
with col2:
    st.markdown("**AmericanAir reply (historical):**")
    st.success(row["americanair_reply"])

st.markdown("---")

esc_col, reason_col = st.columns([1, 3])
with esc_col:
    escalate_flag = st.checkbox("🚩 Escalate this one", key=f"esc_{idx}")
with reason_col:
    reason = ""
    if escalate_flag:
        reason = st.text_input("Why? (brief)", key=f"reason_{idx}")

st.caption("Click the matching intent below — this saves the row and moves to the next one automatically.")

def save_and_advance(label):
    df.at[idx, "intent_label"] = label
    df.at[idx, "escalate"] = "yes" if escalate_flag else "no"
    df.at[idx, "escalate_reason"] = reason
    df.to_csv(LABELED_PATH, index=False)
    if idx < total - 1:
        st.session_state.idx += 1
    st.rerun()

row1 = st.columns(4)
row2 = st.columns(4)
buttons = row1 + row2
for btn_col, label in zip(buttons, INTENTS):
    with btn_col:
        if st.button(label.replace("_", " "), use_container_width=True, key=f"btn_{label}_{idx}"):
            save_and_advance(label)

st.markdown("---")

nav1, nav2, nav3 = st.columns(3)
with nav1:
    if st.button("⬅ Previous", use_container_width=True) and idx > 0:
        st.session_state.idx -= 1
        st.rerun()
with nav2:
    if st.button("⏭ Skip (leave blank)", use_container_width=True):
        if idx < total - 1:
            st.session_state.idx += 1
        st.rerun()
with nav3:
    jump_to = st.number_input("Jump to row #", min_value=1, max_value=total, value=idx + 1, step=1, label_visibility="collapsed")
    if st.button("Go to row", use_container_width=True):
        st.session_state.idx = jump_to - 1
        st.rerun()

st.markdown("---")
st.caption("Progress saves automatically to eval/golden/golden_labeled.csv after every click. Safe to close and resume anytime.")

with st.expander("Intent definitions (reference)"):
    st.markdown("""
    - **flight_disruption** — delays, cancellations, diversions, tarmac holds, missed connections
    - **baggage_issue** — lost, damaged, delayed bags, or baggage fee disputes
    - **billing_fee_dispute** — unauthorized charges, refund requests, seat/fee complaints
    - **booking_seating_assistance** — standby, seat reassignment, rebooking not caused by disruption
    - **loyalty_aadvantage_inquiry** — miles, status match, upgrade eligibility
    - **onboard_technical_issue** — broken IFE, no power, food quality, app bugs
    - **service_staff_complaint** — rude staff, poor communication, general dissatisfaction
    - **praise_non_actionable** — compliments, shoutouts, chit-chat, general non-complaint questions

    **Rule for ambiguous cases:** classify by the actionable ask, not the emotional framing.
    """)

with st.expander("Category counts so far"):
    st.write(df[df["intent_label"] != ""]["intent_label"].value_counts())