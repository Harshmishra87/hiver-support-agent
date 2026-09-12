import pandas as pd
import streamlit as st
import os
import random

GOLDEN_PATH = "eval/golden/golden_labeled.csv"
DRAFTS_PATH = "eval/results/reply_drafts.csv"
OUTPUT_PATH = "eval/results/human_judge_scores.csv"
N_TO_SCORE = 40  # target for judge-vs-human agreement (within your stated 30-50 range)

st.set_page_config(page_title="Human Reply Rater", layout="wide")


@st.cache_data
def load_sample():
    golden = pd.read_csv(GOLDEN_PATH)
    drafts = pd.read_csv(DRAFTS_PATH).dropna(subset=["draft_reply"])
    df = golden.merge(drafts, on="tweet_id", how="inner")
    return df.sample(n=min(N_TO_SCORE, len(df)), random_state=555).reset_index(drop=True)


def load_working_df():
    if os.path.exists(OUTPUT_PATH):
        return pd.read_csv(OUTPUT_PATH)
    df = load_sample().copy()
    df["groundedness"] = None
    df["tone_appropriateness"] = None
    df["resolves_the_ask"] = None
    df["overall"] = None
    return df


if "df" not in st.session_state:
    st.session_state.df = load_working_df()
if "idx" not in st.session_state:
    unlabeled = st.session_state.df[st.session_state.df["overall"].isna()]
    st.session_state.idx = unlabeled.index[0] if len(unlabeled) > 0 else 0

df = st.session_state.df
total = len(df)
scored_count = df["overall"].notna().sum()

st.title("Human Reply Rater — Judge Validation Set")
st.caption("Score these blind to the LLM judge's scores - don't look at judge_scores.csv while doing this.")
st.progress(scored_count / total)
st.caption(f"{scored_count} / {total} scored")

idx = st.session_state.idx
row = df.loc[idx]

st.markdown(f"**Item {idx + 1} of {total}** (tweet_id: {row['tweet_id']})")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Customer message:**")
    st.info(row["customer_text"])
    st.markdown(f"**Intent:** {row['intent_label']}")
with col2:
    st.markdown("**Drafted reply:**")
    st.success(row["draft_reply"])

st.markdown("---")

groundedness = st.slider("Groundedness (1=fabricates facts, 5=fully grounded)", 1, 5, 3, key=f"g_{idx}")
tone = st.slider("Tone appropriateness (1=wrong tone, 5=perfect tone)", 1, 5, 3, key=f"t_{idx}")
resolves = st.slider("Resolves the ask (1=doesn't address it, 5=fully addresses it)", 1, 5, 3, key=f"r_{idx}")
overall = st.slider("Overall quality (1=poor, 5=excellent)", 1, 5, 3, key=f"o_{idx}")

col_a, col_b = st.columns(2)
with col_a:
    if st.button("⬅ Previous", use_container_width=True) and idx > 0:
        st.session_state.idx -= 1
        st.rerun()
with col_b:
    if st.button("💾 Save & Next", type="primary", use_container_width=True):
        df.at[idx, "groundedness"] = groundedness
        df.at[idx, "tone_appropriateness"] = tone
        df.at[idx, "resolves_the_ask"] = resolves
        df.at[idx, "overall"] = overall
        df.to_csv(OUTPUT_PATH, index=False)
        if idx < total - 1:
            st.session_state.idx += 1
        st.rerun()

st.caption("Saves automatically to eval/results/human_judge_scores.csv after every save. Safe to close and resume.")