# AmericanAir Twitter Support Agent

An AI customer support agent for AmericanAir, built on the [Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset (Kaggle, `thoughtvector/customer-support-on-twitter`). Built for the Hiver SDE Intern
take-home assignment.

The agent:

1. **Classifies** each customer message into one of 8 intents derived from real data (`eval/INTENT_TAXONOMY.md`)
2. **Drafts a reply**, grounded in how AmericanAir has historically resolved similar issues (TF-IDF retrieval over ~36k real historical resolutions)
3. **Decides** whether to auto-handle or escalate to a human, with a stated reason

See `REPORT.md` for the full write-up (baselines, failure analysis, business framing, what I chose not to build) and `DECISIONS.md` for the non-obvious decision log.

## Headline results

| Method                            | Accuracy  |
| --------------------------------- | --------- |
| Trivial baseline (majority class) | 21.6%     |
| Keyword baseline (rule-based)     | 44.0%     |
| **LLM intent classification**     | **79.1%** |

Escalation decisioning: 84.1% overall accuracy, 66% recall on true escalations (see `REPORT.md` for why the second number matters more).

## One-time setup

Requires Python 3.13, a Kaggle account (for dataset download), and a Gemini API key.

```powershell
git clone https://github.com/Harshmishra87/hiver-support-agent.git
cd hiver-support-agent

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Required: fix a memory allocation issue on low-RAM machines.** On machines with limited RAM
relative to CPU core count, `numpy`/`pandas` imports can fail with an OpenBLAS memory allocation
error. Fix it once per environment by adding this line to the end of `venv\Scripts\Activate.ps1`:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
```

Then deactivate and reactivate the venv (`deactivate` then `.\venv\Scripts\Activate.ps1`) so it takes effect.

**Kaggle API**: place your `kaggle.json` (or `access_token`) at `~/.kaggle/` per [Kaggle's API docs](https://www.kaggle.com/docs/api).

**Gemini API**: create a `.env` file in the project root with:

GEMINI_API_KEY=your_key_here

Download and prepare the raw data (one-time, ~2 minutes):

```powershell
kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/raw
Expand-Archive -Path "data\raw\customer-support-on-twitter.zip" -DestinationPath "data\raw" -Force
```

## Reproduce headline results in under 15 minutes

All LLM outputs (classification, drafts, escalation decisions, judge scores) are already
generated and committed under `eval/results/`. The commands below **do not re-call the API** for
already-completed work — each script checks `eval/results/` first and only evaluates existing
predictions against the golden set. This should complete in well under 15 minutes with no API costs.

```powershell
# Baselines (instant, no API calls)
python src/baseline_trivial.py
python src/baseline_keyword.py

# Re-evaluates existing LLM predictions against the golden set (no new API calls if already complete)
python src/classify_intent.py
python src/escalate.py

# Re-computes judge-vs-human agreement from existing scores (instant, no API calls)
python src/judge_agreement.py
```

**To regenerate everything from scratch** (not needed for reproduction, only if you want to
re-run the full pipeline): delete the relevant files in `eval/results/` first, then run the same
commands. Full regeneration involves ~1,000+ Gemini API calls across classification, drafting,
escalation, and judging — expect several hours across multiple days due to free-tier daily quota
limits (500 requests/day on `gemini-3.1-flash-lite`), unless billing is enabled on the API project.

## Project Structure

## Project Structure

```text
src/
├── agent.py                     # Main support agent pipeline
├── prompts.py                   # All LLM prompts
├── retrieval.py                 # TF-IDF retrieval system
├── eval_utils.py                # Shared evaluation utilities

├── explore_brands.py            # Brand selection analysis
├── look_at_brand.py             # Inspect real conversation samples
├── sample_100.py                # Sample dataset examples

├── build_golden_candidates.py   # Generate candidates for annotation
├── label_tool.py                # Streamlit labeling interface
├── check_golden_dupes.py        # Detect duplicate golden examples

├── baseline_trivial.py          # Majority-class baseline
├── baseline_keyword.py          # Rule-based baseline

├── classify_intent.py           # Intent classification evaluation
├── generate_drafts.py           # Reply generation evaluation
├── escalate.py                  # Escalation prediction evaluation

├── judge_replies.py             # LLM-as-judge evaluation
├── human_judge_tool.py          # Human evaluation interface
├── judge_agreement.py           # Judge-human agreement analysis

├── find_misclassifications.py   # Error analysis
├── debug_id_mismatch.py         # Dataset debugging utilities

├── demo_app.py                  # Interactive demo application

├── test_draft_reply.py          # Reply generation tests
└── test_gemini.py               # Gemini integration tests

__pycache__/                     # Python cache files

eval/
├── INTENT_TAXONOMY.md        # 8-intent taxonomy with definitions and boundary rules
├── golden/                  # Golden evaluation set (258 hand-labeled examples)
└── results/                 # All evaluation outputs (baselines, classification, judge scores)

data/
├── raw/                     # Downloaded Kaggle dataset (git-ignored)
└── processed/               # Retrieval corpus, sample pairs

.github/
└── workflows/
    └── eval-gate.yml        # CI: reruns baseline eval on prompt/retrieval changes

DECISIONS.md                 # Non-obvious decision log
REPORT.md                    # Full report (baselines, failure analysis, business framing)
```

## Notes on the dataset

Used under the dataset's CC-BY-NC-SA-4.0 license, for non-commercial academic/evaluation use.
